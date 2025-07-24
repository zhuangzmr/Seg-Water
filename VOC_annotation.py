import os
import shutil
import random
from PIL import Image
import numpy as np
from tqdm import tqdm

#============================== 配置参数 ==============================
raw_data_path = ""  # 原始数据集路径
voc_output_path = ""  # 输出VOC格式路径
train_ratio = 0.9  # 训练集比例
val_ratio = 0.1    # 验证集比例
random_seed = 2023  # 随机种子
convert_to_png = True  # 是否将图像转换为PNG格式
#======================================================================
def create_voc_structure():
    """创建VOC目录结构"""
    dirs = [
        "VOC2007/JPEGImages",
        "VOC2007/SegmentationClass",
        "VOC2007/ImageSets/Segmentation"
    ]
    for d in dirs:
        os.makedirs(os.path.join(voc_output_path, d), exist_ok=True)

def process_images_and_labels():
    """处理图像和标签文件"""
    print("\n正在处理图像和标签...")

    # 获取所有原始图像
    all_files = [f for f in os.listdir(raw_data_path) if f.endswith(".jpg") and not f.endswith("_mask.png")]
    total = len(all_files)

    # 进度条设置
    pbar = tqdm(total=total, desc="处理进度")
    error_count = 0
    valid_samples = []

    for img_file in all_files:
        try:
            # 解析文件名
            base_name = os.path.splitext(img_file)[0]
            img_num = base_name

            # 原始文件路径
            raw_img_path = os.path.join(raw_data_path, f"{img_num}.jpg")
            raw_label_path = os.path.join(raw_data_path, f"{img_num}.png")

            # 校验标签文件存在性
            if not os.path.exists(raw_label_path):
                print(f"\n警告：{img_num}.png 标签文件缺失")
                error_count += 1
                continue

            # 处理图像文件
            img_output_path = os.path.join(voc_output_path, f"VOC2007/JPEGImages/{img_num}.{'png' if convert_to_png else 'jpg'}")
            if convert_to_png:
                img = Image.open(raw_img_path)
                img.save(img_output_path)
            else:
                shutil.copy(raw_img_path, img_output_path)

            # 处理标签文件
            label = Image.open(raw_label_path)
            label_arr = np.array(label)

            # 验证标签格式
            if len(label_arr.shape) > 2:
                print(f"\n错误：{img_num}.png 是多通道图像")
                error_count += 1
                continue

            # 转换为二值标签（如果有必要）
            unique_values = np.unique(label_arr)
            if not set(unique_values).issubset({0, 1}):
                print(f"\n警告：{img_num}.png 检测到非0/1值，正在自动转换")
                label_arr = np.where(label_arr > 0, 1, 0).astype(np.uint8)

            # 保存标签
            label_output_path = os.path.join(voc_output_path, f"VOC2007/SegmentationClass/{img_num}.png")
            Image.fromarray(label_arr).save(label_output_path)

            valid_samples.append(img_num)
            pbar.update(1)

        except Exception as e:
            print(f"\n处理 {img_file} 时发生错误：{str(e)}")
            error_count += 1
            continue

    pbar.close()
    print(f"\n处理完成！成功处理 {len(valid_samples)} 个样本，失败 {error_count} 个")
    return valid_samples

def split_dataset(valid_samples):
    """划分数据集"""
    print("\n正在划分数据集...")
    random.seed(random_seed)
    random.shuffle(valid_samples)

    total = len(valid_samples)
    train_num = int(total * train_ratio)

    train_set = valid_samples[:train_num]
    val_set = valid_samples[train_num:]

    # 写入文件
    with open(os.path.join(voc_output_path, "VOC2007/ImageSets/Segmentation/train.txt"), "w") as f:
        f.write("\n".join(train_set))

    with open(os.path.join(voc_output_path, "VOC2007/ImageSets/Segmentation/val.txt"), "w") as f:
        f.write("\n".join(val_set))

    print(f"数据集划分结果：")
    print(f"训练集：{len(train_set)} 个样本")
    print(f"验证集：{len(val_set)} 个样本")

def verify_dataset():
    """验证数据集完整性"""
    print("\n正在验证数据集...")
    jpeg_dir = os.path.join(voc_output_path, "VOC2007/JPEGImages")
    label_dir = os.path.join(voc_output_path, "VOC2007/SegmentationClass")

    # 检查数量匹配
    jpeg_files = set([os.path.splitext(f)[0] for f in os.listdir(jpeg_dir)])
    label_files = set([os.path.splitext(f)[0] for f in os.listdir(label_dir)])

    missing_jpeg = label_files - jpeg_files
    missing_label = jpeg_files - label_files

    if missing_jpeg:
        print(f"错误：缺少 {len(missing_jpeg)} 个对应的图像文件")
    if missing_label:
        print(f"错误：缺少 {len(missing_label)} 个对应的标签文件")

    # 随机抽样检查标签
    sample_files = random.sample(list(jpeg_files), min(5, len(jpeg_files)))
    for file in sample_files:
        label_path = os.path.join(label_dir, file + ".png")
        label = np.array(Image.open(label_path))
        unique = np.unique(label)
        print(f"样本 {file} 标签值分布：{unique}")

def main():
    # 创建目录结构
    create_voc_structure()

    # 处理图像和标签
    valid_samples = process_images_and_labels()

    if not valid_samples:
        print("错误：没有有效样本可供处理！")
        return

    # 划分数据集
    split_dataset(valid_samples)

    # 验证数据集
    verify_dataset()

    print("\n全部处理完成！请检查输出目录：")
    print(voc_output_path)

if __name__ == "__main__":
    main()
