# 导入项目所需的标准库和模块
import datetime  # 用于生成带时间戳的日志目录
import os        # 用于处理文件和目录路径
import math      # 在此脚本中未使用，但在 segwater_training 中可能用到
import numpy as np # 用于处理数组，特别是类别权重
import torch     # PyTorch 核心库
import torch.backends.cudnn as cudnn # 优化CUDNN性能
import torch.optim as optim          # 包含各种优化器
from torch.utils.data import DataLoader # 用于高效加载数据

# 导入项目内部的自定义模块
# 确保导入与您的文件和类名匹配
from nets.segwater import SegFormer, DASCIModule # 导入主模型 SegFormer 和其子模块
from nets.segwater_training import get_lr_scheduler, set_optimizer_lr, weights_init # 导入训练相关的辅助函数
from utils.callbacks import LossHistory, EvalCallback # 导入回调函数，用于记录损失和在验证集上评估
from utils.dataloader import SegmentationDataset, seg_dataset_collate # 导入自定义的数据集类和数据整理函数
from utils.utils import show_config # 导入显示配置信息的函数
from utils.utils_fit import fit_one_epoch # 导入核心的训练/验证循环函数

if __name__ == "__main__":
    # ============================================================================================ #
    #                                    1. 训练超参数和配置设置
    # ============================================================================================ #
    
    # ----------------------------------- 硬件与环境配置 ----------------------------------- #
    Cuda = True             # 是否使用 CUDA (GPU) 进行训练
    distributed = False     # 是否使用分布式训练 (DDP)，在此脚本中未完全实现，设为 False
    sync_bn = False         # 是否使用同步批量归一化 (SyncBatchNorm)，在DDP中常用，此处设为 False
    fp16 = False            # 是否使用混合精度训练 (FP16)，可加速训练并减少显存占用
    num_workers = 4         # 数据加载时使用的子进程数量
    
    # ----------------------------------- 模型相关配置 ------------------------------------- #
    num_classes = 2         # 分割任务的类别数 (例如，1个前景类 + 1个背景类 = 2)
    mit_variant_name = 'b2' # 使用的 MiT (Mix Transformer) 主干网络变体，如 'b0', 'b1', 'b2' 等
    # 预训练权重路径，通常是在大数据集（如ImageNet或VOC）上训练好的模型
    pretrained_full_model_path = "model_data\segformer_b2_weights_voc.pth" 
    # 是否在 ASPP 模块中使用可变形卷积 (DCN)。需要正确安装 mmcv-full
    use_dcn_in_aspp = False
    # 是否在特征融合后使用 DASCI 模块
    use_dasci_post_fusion = True 
    # 控制在 MiT 主干网络的不同阶段输出后是否应用 DASCI 模块。这允许对特定层级的特征进行增强
    dasci_individual_mlp_stages = {'c1': False, 'c2': False, 'c3': False, 'c4': True}
    # 在 SPCII 注意力模块中，1D卷积的组数。大于1时使用分组卷积
    num_groups_1d_conv_in_spcii = 1 
    # 模型断点续训路径。如果此路径下的文件存在，会从该检查点恢复训练
    model_resume_path = '' 
    # 是否使用梯度检查点 (Gradient Checkpointing)。这是一种以计算时间换取显存的技术，适用于大模型
    use_gradient_checkpointing = False

    # ----------------------------------- 数据与训练周期配置 --------------------------------- #
    input_shape = [512, 512] # 模型期望的输入图像尺寸 [H, W]
    Init_Epoch = 0           # 训练起始的 epoch
    Freeze_Epoch = 20        # 冻结训练阶段的 epoch 数。此脚本中未使用冻结训练逻辑，但保留了变量
    UnFreeze_Epoch = 100     # 总的训练 epoch 数
    batch_size = 4           # 每个批次中的样本数量
    Freeze_Train = True      # 是否进行冻结训练。此脚本中该标志未实际控制冻结流程，主要由 epoch 循环控制
    
    # ----------------------------------- 优化器与学习率配置 --------------------------------- #
    Init_lr = 1e-4          # 初始学习率
    Min_lr = Init_lr * 0.01 # 学习率衰减的最小值
    optimizer_type = "adamw"# 优化器类型，可选 'adam', 'adamw', 'sgd'
    momentum = 0.9          # 优化器动量 (主要用于 SGD)
    weight_decay = 0.05     # 权重衰减系数，用于防止过拟合
    lr_decay_type = 'cos'   # 学习率衰减策略，'cos' 表示余弦退火
    
    # ----------------------------------- 损失函数配置 ------------------------------------- #
    dice_loss_enabled = True # 是否启用 Dice Loss
    focal_loss_enabled = False # 是否启用 Focal Loss (与 CE Loss 互斥)
    # 类别权重，用于处理类别不平衡问题。例如，给样本少的类别更高的权重
    cls_weights_np = np.array([0.7, 1.3], dtype=np.float32) if num_classes == 2 else np.ones([num_classes], dtype=np.float32)
    # Lovasz-Softmax Loss 的权重因子。最终损失 = (1-alpha)*CE_Loss + alpha*Lovasz_Loss
    lovasz_alpha = 0.1
    # 是否启用深度监督。如果为True，模型中间层的输出也会被用来计算损失
    using_deep_supervision_flag = True # 注意：当前 SegFormer 模型结构并不返回中间监督所需的输出
    aux_loss_weight = 0.4    # 深度监督的辅助损失的权重
    gradient_clip_val = 1.0  # 梯度裁剪的阈值，防止梯度爆炸
    dice_smooth_coeff = 1e-5 # Dice Loss 中的平滑系数，防止分母为零
    
    # ----------------------------------- 存储与评估配置 ----------------------------------- #
    save_period = 5         # 每隔多少个 epoch 保存一次模型权重
    save_dir = '' # 保存模型权重、日志等文件的目录
    eval_flag = True        # 是否在训练过程中进行评估 (计算 mIoU)
    eval_period = 1         # 每隔多少个 epoch 进行一次评估
    
    # ----------------------------------- 数据集路径配置 ----------------------------------- #
    # 数据集根目录，应包含 VOC2007/ImageSets/Segmentation/train.txt 等文件结构
    VOCdevkit_path = '' 

    # ============================================================================================ #
    #                                2. 初始化环境和模型
    # ============================================================================================ #
    
    # 设置设备 (GPU 或 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() and Cuda else 'cpu')
    # 在分布式训练中，local_rank 表示当前进程在当前节点上的ID
    local_rank = 0 
    rank = 0 # 在分布式训练中，rank 表示全局进程ID

    # 实例化 SegFormer 模型
    model = SegFormer(
        num_classes=num_classes,
        phi=mit_variant_name,
        pretrained=False, # 此处的 pretrained 指的是 MiT 主干网络是否加载 ImageNet 预训练权重，由主干网络内部实现
        mit_weights_path=None, # 可以为 MiT 主干指定一个本地权重文件
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_dcn_in_aspp=use_dcn_in_aspp,
        use_dasci_post_fusion=use_dasci_post_fusion,
        dasci_individual_mlp_stages=dasci_individual_mlp_stages,
        num_groups_1d_conv_in_spcii=num_groups_1d_conv_in_spcii
    )

    # --- 权重加载逻辑 ---
    # 优先级: 断点续训 > VOC预训练 > 随机初始化
    if model_resume_path != "" and os.path.exists(model_resume_path):
        # 1. 尝试从断点续训路径加载
        if local_rank == 0: print(f'从断点续训权重加载: {model_resume_path}')
        model_dict = model.state_dict()
        try:
            pretrained_dict = torch.load(model_resume_path, map_location=device)
            # 兼容不同保存格式的检查点
            if isinstance(pretrained_dict, dict) and 'model_state_dict' in pretrained_dict:
                weights_to_load = pretrained_dict['model_state_dict']
                if local_rank == 0 and 'epoch' in pretrained_dict:
                    print(f"检查点保存于 epoch 结束时: {pretrained_dict['epoch']}")
            else:
                weights_to_load = pretrained_dict
            
            # 去除 'module.' 前缀 (当权重由 DataParallel 保存时会带有这个前缀)
            load_key = {k.replace('module.', ''): v for k, v in weights_to_load.items()}
            # 只加载键名和形状都匹配的权重
            matched_dict = {k: v for k, v in load_key.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(matched_dict)
            missing, unexpected = model.load_state_dict(model_dict, strict=False) # 非严格加载
            if local_rank == 0:
                if missing: print("缺失的键 (续训):", missing)
                if unexpected: print("意外的键 (续训):", unexpected)
                print("成功加载模型权重用于断点续训。")
        except Exception as e:
            if local_rank == 0: print(f"加载断点续训检查点错误: {e}. 模型将以其他方式初始化。")
            # 如果断点续训失败，则尝试加载 VOC 预训练权重
            if pretrained_full_model_path != "" and os.path.exists(pretrained_full_model_path):
                # ... (加载 VOC 权重的逻辑，与下面的 else 分支相同) ...
                pass # 代码省略，因为逻辑与下一个分支重复
            else:
                if local_rank == 0: print("续训失败后无 VOC 路径。从头开始初始化模型。")
                weights_init(model)

    elif pretrained_full_model_path != "" and os.path.exists(pretrained_full_model_path):
        # 2. 如果没有断点续训路径，则尝试加载 VOC 预训练权重
        if local_rank == 0: print(f'从 VOC 预训练权重加载: {pretrained_full_model_path}')
        try:
            checkpoint = torch.load(pretrained_full_model_path, map_location=device)
            # 兼容不同的 checkpoint 格式
            pretrained_voc_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
            
            encoder_weights_to_load = {}
            decoder_weights_to_load = {}
            # 遍历预训练权重字典，分离编码器(backbone)和解码头(decode_head)的权重
            for k, v in pretrained_voc_dict.items():
                k_no_module = k.replace('module.', '')
                if k_no_module.startswith("backbone."):
                    new_k = k_no_module.replace("backbone.", "")
                    if new_k in model.encoder.state_dict() and model.encoder.state_dict()[new_k].shape == v.shape:
                        encoder_weights_to_load[new_k] = v
                elif k_no_module.startswith("decode_head."):
                    new_k = k_no_module.replace("decode_head.", "")
                    if new_k in model.decode_head.state_dict() and model.decode_head.state_dict()[new_k].shape == v.shape:
                        decoder_weights_to_load[new_k] = v

            if encoder_weights_to_load:
                if local_rank == 0: print(f"从 VOC 中找到 {len(encoder_weights_to_load)} 个 ENCODER 键。")
                model.encoder.load_state_dict(encoder_weights_to_load, strict=False)
            if decoder_weights_to_load:
                if local_rank == 0: print(f"从 VOC 中找到 {len(decoder_weights_to_load)} 个 DECODER_HEAD 键。")
                model.decode_head.load_state_dict(decoder_weights_to_load, strict=False)
            
            if not encoder_weights_to_load and not decoder_weights_to_load and local_rank == 0:
                print("警告: 没有匹配到 'backbone.' 或 'decode_head.' 键。VOC 权重可能未加载。")
                weights_init(model) # 如果没加载成功，则随机初始化
            elif local_rank == 0: print("已处理 VOC 预训练权重。")
        except Exception as e:
            if local_rank == 0: print(f"加载 VOC 检查点错误: {e}. 从头开始初始化。")
            weights_init(model)
    else:
        # 3. 如果以上两种权重都没有，则随机初始化模型权重
        if local_rank == 0: print("无续训/VOC路径。从头开始初始化模型。")
        weights_init(model)

    # --- 日志和配置显示 ---
    if local_rank == 0:
        # 生成带时间戳的日志目录
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        os.makedirs(log_dir, exist_ok=True)
        
        # 整理并显示所有重要的训练配置
        config_to_show = {
            "num_classes": num_classes, "mit_variant": mit_variant_name,
            "pretrained_full_model_path": pretrained_full_model_path if pretrained_full_model_path and os.path.exists(pretrained_full_model_path) else "N/A",
            "model_resume_path": model_resume_path if model_resume_path else "N/A",
            "head_embedding_dim": model.embedding_dim,
            "input_shape": input_shape,
            "Init_Epoch": Init_Epoch, "Freeze_Epoch": Freeze_Epoch, "UnFreeze_Epoch": UnFreeze_Epoch,
            "batch_size": batch_size, "Freeze_Train": Freeze_Train,
            "Init_lr": Init_lr, "Min_lr": Min_lr, "optimizer_type": optimizer_type,
            "weight_decay": weight_decay, "lr_decay_type": lr_decay_type,
            "save_period": save_period, "save_dir": save_dir, "num_workers": num_workers,
            "dice_loss_enabled": dice_loss_enabled, "focal_loss_enabled": focal_loss_enabled,
            "cls_weights_preview": list(cls_weights_np), "lovasz_alpha": lovasz_alpha,
            "using_deep_supervision_flag": using_deep_supervision_flag,
            "aux_loss_weight": aux_loss_weight if using_deep_supervision_flag else "N/A",
            "use_dcn_in_aspp": use_dcn_in_aspp,
            "use_dasci_post_fusion": use_dasci_post_fusion,
            "dasci_individual_mlp_stages": dasci_individual_mlp_stages,
            "num_groups_1d_conv_in_spcii": num_groups_1d_conv_in_spcii,
            "gradient_checkpointing": use_gradient_checkpointing, "fp16": fp16,
            "actual_dcn_in_aspp": ( # 检查DCN是否实际生效
                model.decode_head.post_fuse_dasci_module.aspp_module.use_dcn_effective
                if hasattr(model.decode_head, 'post_fuse_dasci_module')
                and isinstance(model.decode_head.post_fuse_dasci_module, DASCIModule)
                else "N/A"
            ),
        }
        show_config(**config_to_show)
        # 初始化 LossHistory 对象，用于记录和绘制损失曲线
        loss_history = LossHistory(log_dir, model, input_shape=(3, input_shape[0], input_shape[1]))
    else:
        loss_history = None

    # 初始化混合精度训练的 GradScaler
    scaler = torch.cuda.amp.GradScaler() if fp16 and torch.cuda.is_available() else None
    
    # 将模型设置为训练模式
    model_train = model.train()

    # --- 模型并行化处理 ---
    if Cuda and torch.cuda.is_available():
        model_train = model.to(device)
        # 如果有多于一个GPU并且未使用分布式训练，则使用 DataParallel
        if torch.cuda.device_count() > 1 and not distributed:
            if local_rank == 0: print(f"在 {torch.cuda.device_count()} 个 GPU 上使用 DataParallel。")
            model_train = torch.nn.DataParallel(model)
        # 开启 cudnn.benchmark 可以让CUDNN自动寻找最适合当前配置的高效算法
        cudnn.benchmark = True
    else:
        model_train = model.to(device)
        if local_rank == 0: print("在 CPU 上训练。")

    # ============================================================================================ #
    #                                  3. 准备数据和优化器
    # ============================================================================================ #

    # --- 加载数据集文件列表 ---
    try:
        with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
            train_lines = f.readlines()
        with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
            val_lines = f.readlines()
    except FileNotFoundError as e:
        if local_rank == 0: print(f"数据集列表文件未找到: {e}"); exit(1)
    num_train, num_val = len(train_lines), len(val_lines)

    # --- 定义优化器 ---
    # pg (parameter group) 是所有需要计算梯度的模型参数
    pg = [p for p in model.parameters() if p.requires_grad]
    if optimizer_type == "adam":
        optimizer = optim.Adam(pg, Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(pg, Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(pg, Init_lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)

    # --- 定义学习率调度器 ---
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, UnFreeze_Epoch)

    # --- 计算每个epoch的步数 ---
    epoch_step = num_train // batch_size if num_train > 0 else 1
    epoch_step_val = num_val // batch_size if num_val > 0 else 1
    if epoch_step == 0 or epoch_step_val == 0:
        if local_rank == 0: print(f"数据集太小或 batch_size 太大。步数: {epoch_step}, {epoch_step_val}"); exit(1)

    # --- 创建数据集和数据加载器实例 ---
    train_dataset = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
    val_dataset   = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    train_loader  = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=seg_dataset_collate, sampler=None)
    val_loader    = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=seg_dataset_collate, sampler=None)

    # --- 初始化评估回调函数 ---
    if local_rank == 0:
        # 如果使用了DataParallel，需要传入 .module 来获取原始模型
        eval_model_cb = model.module if isinstance(model_train, torch.nn.DataParallel) else model
        eval_callback = EvalCallback(net=eval_model_cb, input_shape=tuple(input_shape), num_classes=num_classes,
                                     image_ids=[line.split()[0] for line in val_lines], dataset_path=VOCdevkit_path,
                                     log_dir=log_dir, cuda=Cuda, eval_flag=eval_flag, period=eval_period, letterbox_image=True)
    else:
        eval_callback = None

    # ============================================================================================ #
    #                                      4. 开始训练
    # ============================================================================================ #
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        # 设置当前 epoch 的学习率
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        
        # 调用 fit_one_epoch 函数执行一个完整的训练和验证周期
        fit_one_epoch(
            model_train, model, loss_history, eval_callback, optimizer, epoch,
            epoch_step, epoch_step_val, train_loader, val_loader, UnFreeze_Epoch,
            (Cuda and torch.cuda.is_available()),
            dice_loss_enabled, focal_loss_enabled, cls_weights_np, num_classes,
            (fp16 and torch.cuda.is_available()), scaler,
            save_period, save_dir, lovasz_alpha, using_deep_supervision_flag, aux_loss_weight,
            local_rank,
            dice_smooth_coeff,
            gradient_clip_val,
            num_expected_model_outputs=3 # SegFormer在训练时返回3个输出(seg, con0, con1)
        )

    # 训练结束后，如果是在主进程中，关闭 TensorBoard writer
    if local_rank == 0 and loss_history is not None:
        loss_history.writer.close()
        print("训练完成。TensorBoard writer 已关闭。")