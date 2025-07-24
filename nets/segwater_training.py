import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ================================================================= #
# 辅助函数：学习率调度和权重初始化
# ================================================================= #

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters):
    """
    根据衰减类型获取学习率调度函数。
    Args:
        lr_decay_type (str): 学习率衰减类型 ('cos', 'step', 'poly'等，这里实现'cos')
        lr (float): 初始学习率
        min_lr (float): 最小学习率
        total_iters (int): 总迭代次数或总epoch数 (用于计算衰减进度)
    Returns:
        function: 一个接受当前迭代/epoch作为参数并返回学习率的函数。
    """
    if lr_decay_type == 'cos':
        # Cosine annealing scheduler
        def cos_lr(current_iter):
            return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * current_iter / total_iters))
        return cos_lr
    else:
        raise ValueError(f"Unsupported lr_decay_type: {lr_decay_type}. Only 'cos' is implemented.")

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    """
    根据学习率调度函数设置优化器的学习率。
    Args:
        optimizer (torch.optim.Optimizer): 优化器实例。
        lr_scheduler_func (function): 学习率调度函数，接受当前epoch作为参数。
        epoch (int): 当前epoch。
    """
    new_lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def weights_init(net, init_type='normal', init_gain=0.02):
    """
    初始化神经网络的权重。
    Args:
        net (torch.nn.Module): 神经网络模型。
        init_type (str): 初始化类型 ('normal', 'xavier', 'kaiming', 'orthogonal')。
        init_gain (float): 标准差或增益。
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    
    # 递归地对所有子模块应用初始化函数
    print('Initializing network with %s type' % init_type)
    net.apply(init_func)


# ================================================================= #
# 损失函数
# ================================================================= #

def CE_Loss(inputs, target, cls_weights, num_classes=21):
    """
    Standard Cross-Entropy Loss.
    inputs: [B, C, H, W] (logits)
    target: [B, H, W] (class indices, LongTensor)
    cls_weights: [C] (weights for each class, FloatTensor)
    num_classes: total number of classes (used as ignore_index for F.cross_entropy).
    """
    target = target.long() # Ensure target is LongTensor

    # Handle class weights
    if cls_weights is not None and isinstance(cls_weights, torch.Tensor):
        # Ensure weights are float and on the same device as inputs
        weight_tensor = cls_weights.float().to(inputs.device)
    else:
        weight_tensor = None

    # F.cross_entropy expects inputs [N, C, H, W] and target [N, H, W] for class indices
    loss = F.cross_entropy(inputs, target, weight=weight_tensor, ignore_index=num_classes)
    return loss

def Focal_Loss(inputs, target, cls_weights, num_classes=21, gamma=2, alpha=0.25):
    """
    Focal Loss based on Cross-Entropy.
    inputs: [B, C, H, W] (logits)
    target: [B, H, W] (class indices, LongTensor)
    cls_weights: [C] (weights for each class, FloatTensor)
    num_classes: total number of classes (used as ignore_index for F.cross_entropy).
    gamma, alpha: Focal Loss parameters.
    """
    # Ensure target is 3D (B, H, W) for F.cross_entropy
    if target.dim() == 4:
        if target.shape[1] == 1:
            target = target.squeeze(1)
        else:
            raise ValueError(f"Target dimension mismatch in Focal_Loss. Expected 3 or 4 (with channel 1), "
                             f"but got {target.dim()}D with shape {target.shape}. "
                             f"This indicates a potential issue in input data or parameter passing.")
    
    target = target.long() # Ensure target is LongTensor

    # Compute element-wise Cross-Entropy loss (negative log probability)
    # This gives -log(p_t) for each pixel where p_t is the probability of the true class
    logpt = F.log_softmax(inputs, dim=1)
    # Gather log probabilities of the true classes based on target indices
    logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1) # [B, H, W]

    pt = torch.exp(logpt) # [B, H, W]

    # Initialize alpha_tensor (alpha balancing factor)
    alpha_tensor = torch.full_like(target, alpha, dtype=torch.float, device=inputs.device)
    
    # If class weights are provided, adjust alpha_tensor based on the true class of each pixel
    if cls_weights is not None and isinstance(cls_weights, torch.Tensor):
        # Ensure weights are float and on the same device as inputs
        weight_tensor = cls_weights.float().to(inputs.device)
        # Apply class weights: for each pixel, pick weight based on its true class
        # Ensure target values are valid indices for weight_tensor before indexing
        valid_mask = (target >= 0) & (target < num_classes) # Exclude ignore_index
        alpha_tensor[valid_mask] = weight_tensor[target[valid_mask]] # Apply provided weights

    # Compute focal loss for each pixel
    # Focal Loss = -alpha * (1 - pt)^gamma * log(pt)
    focal_loss = -alpha_tensor * ((1 - pt)**gamma) * logpt
    
    # Apply ignore_index masking: pixels with target == num_classes should not contribute to loss
    mask = (target != num_classes) # Assuming num_classes is the ignore index value
    focal_loss = focal_loss * mask.float()

    return focal_loss.mean()


def Dice_loss(inputs, target, smooth=1e-5):
    """
    Dice Loss for multi-class segmentation.
    inputs: [B, C, H, W] (logits)
    target: [B, H, W, C_num_classes_plus_1] (one-hot from dataloader, for Dice)
    C_num_classes_plus_1 includes background and classes, potentially also ignore_index.
    """
    # Convert inputs to probabilities
    inputs = torch.softmax(inputs, dim=1) # inputs: [B, C, H, W]

    # Adjust target dimensions from [B, H, W, C] to [B, C, H, W]
    # The last dimension of target is (num_classes + 1) which might include an 'ignore' channel.
    # We should only consider the actual class channels for Dice.
    # Assuming the 'ignore' channel is the last one if present.
    # If target.shape[-1] > inputs.shape[1], we trim target.
    num_actual_classes = inputs.shape[1]
    if target.dim() == 4:
        if target.shape[-1] > num_actual_classes: # Target might include an extra ignore channel
            target = target[:, :, :, :num_actual_classes] # Trim to match actual classes
        target = target.permute(0, 3, 1, 2) # Permute to [B, C, H, W]
    else:
        raise ValueError(f"Dice_loss: Expected target to be 4D (B,H,W,C_or_C+1), "
                         f"but got {target.dim()}D with shape {target.shape}.")

    # Flatten inputs and target for calculation
    # inputs and target should now both be [B, C, H, W] after softmax and permute
    inputs_flat = inputs.contiguous().view(inputs.size(0), inputs.size(1), -1) # [B, C, H*W]
    target_flat = target.contiguous().view(target.size(0), target.size(1), -1).float() # [B, C, H*W]

    # Intersection and Union calculation across spatial dimensions for each class and batch
    intersection = (inputs_flat * target_flat).sum(dim=2) # [B, C]
    union = inputs_flat.sum(dim=2) + target_flat.sum(dim=2) # [B, C]

    # Dice score for each class and batch
    dice_per_class = (2. * intersection + smooth) / (union + smooth) # [B, C]
    
    # Average Dice score over classes and batches
    # Exclude background class if desired, or average over all C classes.
    # For common semantic segmentation, usually average over all classes including background.
    return 1 - dice_per_class.mean() # Return 1 - mean Dice as loss


def ConLoss(pred_con, gt_con):
    """
    Connectivity Loss, typically MSE Loss.
    pred_con: [B, C_neighbor, H, W] (predictions for connectivity map from model)
    gt_con: [B, H, W, C_neighbor] (ground truth connectivity map from dataloader)
    """
    # Need to permute gt_con to [B, C_neighbor, H, W] to match pred_con
    if gt_con.dim() == 4 and gt_con.shape[-1] == pred_con.shape[1]:
        gt_con = gt_con.permute(0, 3, 1, 2)
    elif gt_con.dim() != 4:
        raise ValueError(f"ConLoss: Expected gt_con to be 4D (B,H,W,C_neighbor), "
                         f"but got {gt_con.dim()}D with shape {gt_con.shape}")

    # Assuming pred_con and gt_con are now both [B, C_neighbor, H, W]
    # Use MSELoss
    loss_fn = nn.MSELoss()
    loss = loss_fn(pred_con, gt_con.float()) # Ensure gt_con is float
    return loss