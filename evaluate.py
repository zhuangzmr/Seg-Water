import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse # 导入 argparse 用于解析命令行参数
import os # 用于路径操作
import ast # 用于解析字符串形式的字典

# --- 从你的项目中导入必要的模块 ---
# 请确保这些路径相对于你运行 evaluate.py 的位置是正确的
try:
    from utils.dataloader import SegmentationDataset, seg_dataset_collate
    from nets.segwater import SegFormer # 导入 SegFormer
except ImportError as e:
    print(f"导入自定义模块失败: {e}")
    print("请确保 'utils' 和 'nets' 文件夹及其内容与 'evaluate.py' 在同一级别，")
    print("或者这些文件夹在你的 PYTHONPATH 中。")
    exit(1)


class SegmentationEvaluator:
    def __init__(self, num_classes=2, device='cuda'):
        self.num_classes = num_classes
        self.device = torch.device(device) 
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        
    def _reset_metrics(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        
    def _update_class_metrics(self, pred, target):
        # pred 和 target 预期已经是 numpy 数组，且形状为 (B, H, W)
        
        # 将预测和标签展平
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # 排除掉 ignore_index 的像素
        # 根据 dataloader.py 中的逻辑，png_array[png_array >= self.num_classes] = self.num_classes
        # 所以 self.num_classes 这个值在标签中代表了 ignore_index
        valid_mask = (target_flat < self.num_classes)
        
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]

        if len(target_valid) == 0:
            # 没有有效像素，跳过此批次
            return

        current_confusion = confusion_matrix(
            target_valid, pred_valid, 
            labels=np.arange(self.num_classes) # 指定所有可能的类别标签，确保混淆矩阵完整
        )
        self.confusion += current_confusion

        
    def evaluate(self, model, dataloader):
        model.eval()
        self._reset_metrics()
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Evaluating"):
                # batch_data 预期包含 (images, png_idx_map, con1_target, con3_target, seg_labels_one_hot)
                images = batch_data[0].to(self.device)
                labels_idx_map = batch_data[1] # 这是 (B, H, W) 的索引图，在CPU

                outputs = model(images)
                # SegFormer 在 eval 模式下通常只返回主分割预测 (seg_pred)
                # outputs 形状是 [B, num_classes, H_out, W_out]
                
                preds_indices = torch.argmax(outputs, dim=1) # [B, H_out, W_out]
                
                # labels_idx_map 从 dataloader 返回已经是 (B, H, W) 的索引图
                targets_indices = labels_idx_map 
                
                # 尺寸对齐 (如果模型输出尺寸与标签不一致)
                if preds_indices.shape[-2:] != targets_indices.shape[-2:]:
                    preds_indices = F.interpolate(
                        preds_indices.unsqueeze(1).float(), # [B, 1, H_out, W_out]
                        size=targets_indices.shape[-2:],
                        mode='nearest'
                    ).squeeze(1).long() # [B, H_target, W_target]
                
                # 转换为 numpy 用于 confusion_matrix
                preds_np = preds_indices.cpu().numpy()
                targets_np = targets_indices.cpu().numpy()
                
                self._update_class_metrics(preds_np, targets_np)
        
        # --- 指标计算 ---
        hist = self.confusion
        eps = 1e-10 # 增加一个非常小的数以避免除以零

        # IoU per class
        intersection = np.diag(hist)
        ground_truth_set = hist.sum(axis=1) # 真实标签中每个类别的总数
        predicted_set = hist.sum(axis=0)  # 预测标签中每个类别的总数
        union = ground_truth_set + predicted_set - intersection
        
        iou_per_class = intersection / (union + eps)
        mIoU = np.nanmean(iou_per_class) * 100 # 平均IoU，转换为百分比
        
        # Recall per class (Pixel Accuracy per class)
        pa_per_class = intersection / (ground_truth_set + eps) 
        mRecall = np.nanmean(pa_per_class) * 100 # 平均召回率，转换为百分比
        
        # Precision per class
        precision_per_class = intersection / (predicted_set + eps)
        mPrecision = np.nanmean(precision_per_class) * 100 # 平均精确率，转换为百分比
        
        # F1 Score per class
        # 注意：这里 Precision 和 Recall 已经是百分比了，在计算F1时需要先转回0-1范围
        f1_per_class = 2 * ((precision_per_class / 100) * (pa_per_class / 100)) / ((precision_per_class / 100) + (pa_per_class / 100) + eps)
        mF1 = np.nanmean(f1_per_class) * 100 # 平均F1，转换为百分比
        
        # Overall Accuracy (aAcc)
        overall_accuracy = np.sum(intersection) / (np.sum(hist) + eps) * 100 # 整体准确率，转换为百分比
        
        final_metrics = {
            'mIoU': mIoU,
            'mPrecision': mPrecision, 
            'mRecall': mRecall,
            'mF1': mF1,
            'aAcc': overall_accuracy 
        }
        
        # 打印每个类别的指标
        print("\n--- Per-class Metrics (in %) ---")
        for i in range(self.num_classes):
            print(f"Class {i}:")
            print(f"  IoU:       {iou_per_class[i]*100:.2f}%")
            print(f"  Precision: {precision_per_class[i]*100:.2f}%")
            print(f"  Recall:    {pa_per_class[i]*100:.2f}%")
            print(f"  F1-Score:  {f1_per_class[i]*100:.2f}%")
            
        return final_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SegFormer model")
    
    # 默认路径和参数设置
    parser.add_argument("--model_path", type=str, 
                        default="模型地址", 
                        help="Path to the trained model weights (.pth). Default: %(default)s")

    parser.add_argument("--dataset_root_dir", type=str, 
                        default="", #VOCdevkit的父目录
                        help="Root directory of the VOCdevkit dataset. Default: %(default)s")
    
    parser.add_argument("--val_split_file", type=str, 
                        default="VOC2007/ImageSets/Segmentation/val.txt", 
                        help="Path to the validation split file, relative to dataset_root_dir. Default: %(default)s")
    
    parser.add_argument("--phi", type=str, default="b2", help="SegFormer backbone type. Must match training. Default: %(default)s")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes. Default: %(default)s")
    parser.add_argument("--model_input_shape", type=str, default="512,512", help="Model input image size 'height,width'. Default: %(default)s")
    
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for DataLoader. Default: %(default)s")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader. Default: %(default)s")
    
    parser.add_argument("--cuda", action='store_true', default=True, 
                        help="Use CUDA if available. Default: %(default)s")
    parser.add_argument("--no_cuda", action='store_false', dest='cuda',
                        help="Do not use CUDA, force CPU.")

    # 模型架构参数，必须与训练时完全一致
    parser.add_argument("--use_dcn_in_aspp", action='store_true', default=True, # 与train.py配置一致
                        help="Whether to use Deformable Convolution in ASPP module. Must match training. Default: %(default)s")
    parser.add_argument("--use_dasci_post_fusion", action='store_true', default=True, # 与train.py配置一致
                        help="Whether to use DASCI module after feature fusion. Must match training. Default: %(default)s")
    # dasci_individual_mlp_stages 字典的字符串表示
    parser.add_argument("--dasci_individual_mlp_stages_str", type=str, 
                        default="{'c1': False, 'c2': False, 'c3': False, 'c4': True}", # 与train.py配置一致
                        help="Dictionary string for individual DASCI modules per MLP stage. Must match training. Default: %(default)s")
    parser.add_argument("--num_groups_1d_conv_in_spcii", type=int, default=1, # 与train.py配置一致
                        help="Number of groups for 1D convolution in SPCII module. Must match training. Default: %(default)s")

    args = parser.parse_args()

    device_name = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    device = torch.device(device_name)
    print(f"Using device: {device}")
    
    # 1. 模型加载
    print(f"Loading model: {args.model_path}")
    print(f"  - Backbone (phi): {args.phi}")
    print(f"  - Num Classes: {args.num_classes}")

    # 解析 dasci_individual_mlp_stages 字符串
    try:
        dasci_stages_dict = ast.literal_eval(args.dasci_individual_mlp_stages_str)
        if not isinstance(dasci_stages_dict, dict):
            raise ValueError
    except (ValueError, SyntaxError):
        print(f"错误: dasci_individual_mlp_stages_str 参数格式不正确。应为Python字典的字符串表示。")
        exit(1)


    model = SegFormer(
        num_classes=args.num_classes, 
        phi=args.phi,
        pretrained=False, # 加载自定义权重
        use_dcn_in_aspp=args.use_dcn_in_aspp,
        use_dasci_post_fusion=args.use_dasci_post_fusion,
        dasci_individual_mlp_stages=dasci_stages_dict,
        num_groups_1d_conv_in_spcii=args.num_groups_1d_conv_in_spcii
    )
    
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        # 处理 DataParallel 带来的 'module.' 前缀
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # 使用 strict=True 严格加载，确保模型结构完全匹配
        model.load_state_dict(state_dict, strict=True) 
        print("Model weights loaded successfully.")
    except Exception as e: # 更通用的异常捕获
        print(f"Error loading model weights: {e}")
        print("--------------------------------------------------------------------------------------")
        print("重要提示：请检查模型架构参数 (phi, use_dcn_in_aspp, use_dasci_post_fusion, dasci_individual_mlp_stages_str, num_groups_1d_conv_in_spcii) 是否与训练时完全一致。")
        print("  如果训练时因'mmcv'未安装导致'use_dcn_in_aspp'实际未启用，请将此脚本中的'use_dcn_in_aspp'改为'False'。")
        print("--------------------------------------------------------------------------------------")
        exit(1) # 加载失败则退出
    model = model.to(device)
    
    # 2. 数据集和 DataLoader 设置
    val_split_file_path = os.path.join(args.dataset_root_dir, args.val_split_file)
    if not os.path.exists(val_split_file_path):
        print(f"Error: Validation split file not found at {val_split_file_path}")
        exit(1) # 文件不存在则退出
        
    with open(val_split_file_path, 'r') as f:
        val_lines = f.readlines()
    
    if not val_lines:
        print(f"Error: Validation split file {val_split_file_path} is empty.")
        exit(1) # 文件为空则退出

    try:
        h_str, w_str = args.model_input_shape.split(',')
        model_input_shape_tuple = (int(h_str), int(w_str))
    except ValueError:
        print(f"Error: model_input_shape format should be 'height,width'. Got: {args.model_input_shape}")
        exit(1) # 格式错误则退出

    val_dataset = SegmentationDataset(
        annotation_lines=val_lines, 
        input_shape=model_input_shape_tuple, # (H, W)
        num_classes=args.num_classes,
        train=False, # 评估模式
        dataset_path=args.dataset_root_dir # 传递数据集根目录
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True if device_name == 'cuda' else False,
        collate_fn=seg_dataset_collate
    )
    
    if len(val_loader) == 0:
        print("Error: DataLoader is empty. Check dataset path, split file, or batch size.")
        exit(1) # DataLoader 为空则退出

    # 3. 评估器初始化和执行
    evaluator = SegmentationEvaluator(num_classes=args.num_classes, device=device_name)
    
    print("\n====== Starting Evaluation ======")
    accuracy_metrics = evaluator.evaluate(model, val_loader)
    
    print("\n====== Evaluation Results (Averages) ======")
    for k, v in accuracy_metrics.items():
        print(f"{k}: {v:.4f}")