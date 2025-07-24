# 文件: dataloader.py

import os
import time # 用于临时调试

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import preprocess_input, cvtColor
from numba import jit # 确保导入 numba

# 将 get_con_1 和 get_con_3 移到类的外部，作为独立的JIT函数
@jit(nopython=True) # 启用 Numba JIT
def get_con_1_static(png_array): # 输入是 numpy 数组，去掉 self
    # 对于二分类，png_array 可能是 0 或 1，或者 0 或 255。
    # 这里统一处理为 0 或 1 的前景背景判断。
    img = np.where(png_array > 0, 1, 0) # 基于 png_array (语义标签) 判断前景
    shp = img.shape

    img_pad = np.zeros((shp[0] + 4, shp[1] + 4), dtype=img.dtype)
    img_pad[2:-2, 2:-2] = img
    dir_array0 = np.zeros((shp[0], shp[1], 9), dtype=np.float32)

    for i in range(shp[0]):
        for j in range(shp[1]):
            if img[i, j] == 0: # 只处理前景像素
                continue
            dir_array0[i, j, 0] = img_pad[i, j]
            dir_array0[i, j, 1] = img_pad[i, j + 2]
            dir_array0[i, j, 2] = img_pad[i, j + 4]
            dir_array0[i, j, 3] = img_pad[i + 2, j]
            dir_array0[i, j, 4] = img_pad[i + 2, j + 2]
            dir_array0[i, j, 5] = img_pad[i + 2, j + 4]
            dir_array0[i, j, 6] = img_pad[i + 4, j]
            dir_array0[i, j, 7] = img_pad[i + 4, j + 2]
            dir_array0[i, j, 8] = img_pad[i + 4, j + 4]
    return dir_array0

@jit(nopython=True) # 启用 Numba JIT
def get_con_3_static(png_array): # 输入是 numpy 数组，去掉 self
    img = np.where(png_array > 0, 1, 0)
    shp = img.shape

    img_pad = np.zeros((shp[0] + 8, shp[1] + 8), dtype=img.dtype)
    img_pad[4:-4, 4:-4] = img
    dir_array0 = np.zeros((shp[0], shp[1], 9), dtype=np.float32)

    for i in range(shp[0]):
        for j in range(shp[1]):
            if img[i, j] == 0: # 只处理前景像素
                continue
            dir_array0[i, j, 0] = img_pad[i, j]
            dir_array0[i, j, 1] = img_pad[i, j + 4]
            dir_array0[i, j, 2] = img_pad[i, j + 8]
            dir_array0[i, j, 3] = img_pad[i + 4, j]
            dir_array0[i, j, 4] = img_pad[i + 4, j + 4]
            dir_array0[i, j, 5] = img_pad[i + 4, j + 8]
            dir_array0[i, j, 6] = img_pad[i + 8, j]
            dir_array0[i, j, 7] = img_pad[i + 8, j + 4]
            dir_array0[i, j, 8] = img_pad[i + 8, j + 8]
    return dir_array0


class SegmentationDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(SegmentationDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape # 期望 (H, W)
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        # 调整为 VOC2007 路径结构
        jpg_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages", name + ".jpg")
        png_path = os.path.join(self.dataset_path, "VOC2007/SegmentationClass", name + ".png")

        try:
            jpg         = Image.open(jpg_path)
            png         = Image.open(png_path)
        except FileNotFoundError:
            print(f"错误: 文件未找到 for name {name}. JPG: {jpg_path}, PNG: {png_path}")
            # 记录文件名到日志或跳过，而不是直接raise，取决于训练策略
            # raise # 如果要严格检查，则保留
            # 为了继续训练，可以返回 None 或一个空批次（但这会使 collate_fn 复杂化）
            # 最简单的是确保数据路径正确或跳过此项
            print(f"Skipping index {index} due to missing file.")
            # 返回一个可以被 collate_fn 过滤掉的特殊值，或者直接跳过
            # 在多进程Dataloader中直接raise会崩溃，最好在创建DataLoader前检查文件存在性
            # 或者在 collate_fn 中处理 None 值。
            # 这里保持raise以避免静默错误，但在实际应用中可能需要更健壮的错误处理。
            raise FileNotFoundError(f"文件未找到: {jpg_path} 或 {png_path}")

        jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)

        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        
        png_array   = np.array(png)
        # 确保标签值在有效范围内 [0, num_classes-1] 或 num_classes 作为 ignore_index
        # 通常 num_classes 是背景+所有前景类的数量。
        # 如果标签图中有超出 num_classes-1 的值（如 VOC 的 255 表示 ignore），应映射到 num_classes
        png_array[png_array >= self.num_classes] = self.num_classes

        # seg_labels 是 One-Hot 编码的标签，形状 (H, W, num_classes + 1)
        # num_classes + 1 是为了包含背景类和前景类，以及可能的 ignore_index
        seg_labels  = np.eye(self.num_classes + 1)[png_array.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        # 调用独立的 JIT 函数
        con1_data = get_con_1_static(png_array)
        con3_data = get_con_3_static(png_array)

        return jpg, png_array, con1_data, con3_data, seg_labels

    def rand(self, a=0, b=1): # 这个函数没有被JIT装饰，所以它可以是类方法
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.5, hue=.15, sat=0.8, val=0.4, random=True):
        image   = cvtColor(image)
        label   = label.convert('L')
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w, h), (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', (w, h), (0)) #背景用0填充，对于分割任务通常是合理的
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        # 随机改变宽高比和缩放
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.4, 2.2) 
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        
        # 随机水平翻转
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 随机放置
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0)) # 背景用0填充
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label_pil = new_label # PIL Image 格式的标签，用于后续可能的 warpAffine

        image_data      = np.array(image, np.uint8)
        
        # 随机高斯模糊
        blur = self.rand() < 0.30 
        if blur: 
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        # 随机旋转
        rotate = self.rand() < 0.30 
        if rotate: 
            center      = (w // 2, h // 2)
            rotation    = np.random.randint(-15, 16) 
            M           = cv2.getRotationMatrix2D(center, -rotation, scale=1) # 保持scale=1，因为前面已经有缩放增强
            image_data  = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
            # 对标签图应用同样的旋转，使用最近邻插值，边界填充0
            label_np    = cv2.warpAffine(np.array(label_pil, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))
            label_pil   = Image.fromarray(label_np) # 更新PIL Image格式的标签

        # 色彩抖动 (HSV空间)
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue_val, sat_val, val_val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype) # Hue 范围是 0-179 in OpenCV
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue_val, lut_hue), cv2.LUT(sat_val, lut_sat), cv2.LUT(val_val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return Image.fromarray(image_data), label_pil 


def seg_dataset_collate(batch):
    images      = []
    pngs        = [] # 存储处理后的标签图 (numpy array, index map)
    con_1_list  = []
    con_3_list  = []
    seg_labels_one_hot_list  = [] # 存储one-hot编码的标签

    for img, png_array, c1, c3, labels_one_hot in batch:
        images.append(img)
        pngs.append(png_array)
        con_1_list.append(c1)
        con_3_list.append(c3)
        seg_labels_one_hot_list.append(labels_one_hot)

    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long() # 标签图应为 LongTensor (B, H, W)
    con_1       = torch.from_numpy(np.array(con_1_list)).type(torch.FloatTensor) # (B, H, W, 9)
    con_3       = torch.from_numpy(np.array(con_3_list)).type(torch.FloatTensor) # (B, H, W, 9)
    seg_labels_one_hot  = torch.from_numpy(np.array(seg_labels_one_hot_list)).type(torch.FloatTensor) # (B, H, W, num_classes+1)
    
    return images, pngs, con_1, con_3, seg_labels_one_hot