import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

# 导入SegFormer模型和辅助函数
# 请确保这些路径相对于你运行 inference.py 的位置是正确的
try:
    from nets.segwater import SegFormer, DASCIModule 
    from utils.utils import cvtColor, preprocess_input
except ImportError as e:
    print(f"导入自定义模块失败: {e}")
    print("请确保 'nets' 和 'utils' 文件夹及其内容与 'inference.py' 在同一级别，")
    print("或者这些文件夹在你的 PYTHONPATH 中。")
    exit(1)

def get_model_inference(model_path, phi='b2', num_classes=2, cuda=True, input_shape=(512, 512),
                        use_dcn_in_aspp=True, # 与train.py配置一致
                        use_dasci_post_fusion=True, # 与train.py配置一致
                        dasci_individual_mlp_stages={'c1': False, 'c2': False, 'c3': False, 'c4': True}, # 与train.py配置一致
                        num_groups_1d_conv_in_spcii=1): # 与train.py配置一致
    """
    加载模型进行推理。
    """
    device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')

    model = SegFormer(
        num_classes=num_classes,
        phi=phi,
        pretrained=False, # 我们加载自己的权重，而非MiT的预训练权重
        mit_weights_path=None, # 不加载MiT的预训练权重
        use_gradient_checkpointing=False, # 推理时不需要
        use_dcn_in_aspp=use_dcn_in_aspp,
        use_dasci_post_fusion=use_dasci_post_fusion,
        dasci_individual_mlp_stages=dasci_individual_mlp_stages,
        num_groups_1d_conv_in_spcii=num_groups_1d_conv_in_spcii
    )

    print(f"尝试从 '{model_path}' 加载模型权重...")
    try:
        # 加载权重，处理DataParallel带来的'module.'前缀
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            # 如果保存的是整个训练状态（如在train.py中），提取模型状态
            loaded_weights = state_dict['model_state_dict']
            print(f"检测到检查点保存于 epoch {state_dict.get('epoch', 'N/A')}")
        else:
            # 如果直接保存的是model.state_dict()
            loaded_weights = state_dict

        # 移除DataParallel在键名中添加的'module.'前缀
        new_state_dict = {}
        for k, v in loaded_weights.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # 使用 strict=True 严格加载，确保模型结构完全匹配
        model.load_state_dict(new_state_dict, strict=True)
        print("模型权重加载成功。")
    except Exception as e:
        print(f"加载模型权重失败: {e}")
        print("--------------------------------------------------------------------------------------")
        print("重要提示：请确保模型架构参数与训练时完全一致！")
        print(f"  当前模型参数: phi={phi}, use_dcn_in_aspp={use_dcn_in_aspp}, use_dasci_post_fusion={use_dasci_post_fusion},")
        print(f"  dasci_individual_mlp_stages={dasci_individual_mlp_stages}, num_groups_1d_conv_in_spcii={num_groups_1d_conv_in_spcii}")
        print("  如果训练时因'mmcv'未安装导致'use_dcn_in_aspp'实际未启用，请将此脚本中的'use_dcn_in_aspp'改为'False'。")
        print("--------------------------------------------------------------------------------------")
        raise e

    model.eval() # 设置为评估模式
    if cuda:
        model = model.to(device)
    print(f"模型已加载到 {device}。")
    return model, input_shape # 返回模型和其期望的输入尺寸

def letterbox_image(image, size):
    """
    将图像调整大小并填充到指定尺寸，保持图像比例不变。
    这个函数是从callbacks.py中的letterbox方法修改而来。
    """
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image, nw, nh

def predict_and_save(model, input_path, output_path, input_shape, threshold, cuda, letterbox_resize=True):
    """
    对单张图片进行预测并保存结果。
    """
    try:
        image = Image.open(input_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
    except Exception as e:
        print(f"无法打开或处理图片 {input_path}: {e}")
        return

    original_w, original_h = image.size

    # --- 预处理图像 ---
    if letterbox_resize:
        input_image, nw, nh = letterbox_image(image, (input_shape[1], input_shape[0])) # input_shape是(H,W), letterbox需要(W,H)
    else:
        input_image = image.resize((input_shape[1], input_shape[0]), Image.BICUBIC)
        nw, nh = input_shape[1], input_shape[0]

    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(input_image, np.float32)), (2, 0, 1)), 0)
    
    with torch.no_grad():
        images = torch.from_numpy(image_data)
        if cuda:
            images = images.cuda()
        
        # 模型在eval模式下只返回seg_pred
        outputs = model(images) 
        
        # outputs的形状是 (B, C, H, W)，这里 B=1
        # 我们需要获取前景（水体）的概率，通常是类别1
        # pr 此时是 (num_classes, H, W)
        pr = F.softmax(outputs.squeeze(0), dim=0) # 移除batch维度，并对类别维度进行softmax

        # 获取前景（类别1）的概率图
        # 确保num_classes是2，则类别0是背景，类别1是前景
        if pr.shape[0] == 2:
            foreground_prob_map = pr[1, :, :].cpu().numpy() # 提取类别1的概率图
        else:
            print(f"警告: 模型输出类别数非2 ({pr.shape[0]}), 无法确定前景类别。将使用argmax结果。")
            foreground_prob_map = pr.argmax(dim=0).cpu().numpy() # 如果不是二分类，就直接取argmax

        # --- 后处理，缩放回原始尺寸并应用阈值 ---
        if letterbox_resize:
            # 裁剪letterbox填充部分
            offset_h = (input_shape[0] - nh) // 2
            offset_w = (input_shape[1] - nw) // 2
            foreground_prob_map = foreground_prob_map[offset_h : offset_h + nh, offset_w : offset_w + nw]

        # 缩放回原始尺寸
        foreground_prob_map = Image.fromarray((foreground_prob_map * 255).astype(np.uint8))
        foreground_prob_map = foreground_prob_map.resize((original_w, original_h), Image.BICUBIC)
        foreground_prob_map = np.array(foreground_prob_map) / 255.0 # 再次归一化到0-1

        # 应用阈值，生成二值化结果
        binary_mask = (foreground_prob_map > threshold).astype(np.uint8) * 255 # 水体为白色 (255), 非水体为黑色 (0)
        
        # 保存结果
        result_image = Image.fromarray(binary_mask)
        result_image.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="SegFormer Water Segmentation Inference Script")

    # 默认路径和参数设置
    parser.add_argument('--model_path', type=str, 
                        default="", #模型地址
                        help='Path to the trained model weights. Default: %(default)s')
    parser.add_argument('--input_folder', type=str, 
                        default="", # 示例输入图片文件夹
                        help='Path to the folder containing input images for segmentation. Default: %(default)s')
    parser.add_argument('--output_folder', type=str, 
                        default="kaggle识gaofen",
                        help='Path to the folder where segmented images will be saved. Default: %(default)s')

    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary segmentation. Default: %(default)s')
    parser.add_argument('--phi', type=str, default='b2',
                        help='MiT backbone variant. Must match training. Default: %(default)s')
    parser.add_argument('--input_size', type=int, nargs=2, default=(512, 512),
                        help='Input image size for the model (H W). Must match training. Default: %(default)s')
    parser.add_argument('--cuda', action='store_true', default=True, 
                        help='Use CUDA if available. Default: %(default)s')
    parser.add_argument('--no_cuda', action='store_false', dest='cuda',
                        help='Do not use CUDA, force CPU.')
    parser.add_argument('--letterbox_resize', action='store_true', default=True,
                        help='Use letterbox resizing during preprocessing. Default: %(default)s')
    parser.add_argument('--no_letterbox_resize', action='store_false', dest='letterbox_resize',
                        help='Do not use letterbox resizing.')

    # 模型架构参数，必须与训练时完全一致
    parser.add_argument('--use_dcn_in_aspp', action='store_true', default=True, # 与train.py配置一致
                        help='Whether to use Deformable Convolution in ASPP module. Must match training. Default: %(default)s')
    parser.add_argument('--use_dasci_post_fusion', action='store_true', default=True, # 与train.py配置一致
                        help='Whether to use DASCI module after feature fusion. Must match training. Default: %(default)s')
    # dasci_individual_mlp_stages 字典的字符串表示
    parser.add_argument('--dasci_individual_mlp_stages_str', type=str, 
                        default="{'c1': False, 'c2': False, 'c3': False, 'c4': True}", # 与train.py配置一致
                        help='Dictionary string for individual DASCI modules per MLP stage. Must match training. Default: %(default)s')
    parser.add_argument('--num_groups_1d_conv_in_spcii', type=int, default=1, # 与train.py配置一致
                        help='Number of groups for 1D convolution in SPCII module. Must match training. Default: %(default)s')

    args = parser.parse_args()

    # 将字符串转换为字典
    import ast
    try:
        args.dasci_individual_mlp_stages = ast.literal_eval(args.dasci_individual_mlp_stages_str)
        if not isinstance(args.dasci_individual_mlp_stages, dict):
            raise ValueError
    except (ValueError, SyntaxError):
        print(f"错误: dasci_individual_mlp_stages_str 参数格式不正确。应为Python字典的字符串表示。")
        exit(1)

    # 确保输出文件夹存在
    os.makedirs(args.output_folder, exist_ok=True)

    # 1. 加载模型
    model, input_shape = get_model_inference(
        model_path=args.model_path,
        phi=args.phi,
        num_classes=2, # 水体分割是二分类
        cuda=args.cuda,
        input_shape=tuple(args.input_size),
        use_dcn_in_aspp=args.use_dcn_in_aspp,
        use_dasci_post_fusion=args.use_dasci_post_fusion,
        dasci_individual_mlp_stages=args.dasci_individual_mlp_stages,
        num_groups_1d_conv_in_spcii=args.num_groups_1d_conv_in_spcii
    )

    # 2. 获取所有图片文件
    image_files = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    for root, _, files in os.walk(args.input_folder):
        for file in files:
            if file.lower().endswith(supported_extensions):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print(f"在 '{args.input_folder}' 中未找到任何支持的图片文件。")
        return

    print(f"找到 {len(image_files)} 张图片进行推理...")

    # 3. 逐一进行预测并保存
    for image_path in tqdm(image_files, desc="Processing Images"):
        file_name = os.path.basename(image_path)
        # 为输出文件添加 "_seg" 后缀，避免覆盖原始图片
        name_without_ext, ext = os.path.splitext(file_name)
        output_image_path = os.path.join(args.output_folder, f"{name_without_ext}{ext}") 
        
        predict_and_save(model, image_path, output_image_path, input_shape, args.threshold, args.cuda, args.letterbox_resize)

    print(f"所有分割结果已保存到 '{args.output_folder}'。")

if __name__ == "__main__":
    main()