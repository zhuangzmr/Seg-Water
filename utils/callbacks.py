# callbacks.py

import os
import datetime
import cv2 # Make sure to install opencv-python
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from tqdm import tqdm # For progress bar

# Assuming utils_metrics.py is in the same directory or accessible
try:
    from .utils_metrics import compute_mIoU #, show_results
except ImportError:
    from utils_metrics import compute_mIoU #, show_results


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            # input_shape 预期是 (C, H, W) 或 (H,W) (如果 C 固定, 例如 3)
            # 如果 input_shape 是 (H,W)，则 dummy_input 需要明确的 C
            if len(input_shape) == 2: # 假设是 (H,W) 并且 C=3
                dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            elif len(input_shape) == 3: # 假设是 (C,H,W)
                dummy_input = torch.randn(2, input_shape[0], input_shape[1], input_shape[2])
            else:
                raise ValueError(f"不支持的 input_shape 格式: {input_shape}")
            
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
            # self.writer.add_graph(model, dummy_input) # 可能有问题
            print("TensorBoard: 尝试添加计算图 (如果出错则跳过).")
        except Exception as e: # 如果可能，捕获特定的错误
            print(f"TensorBoard: 未能为模型添加计算图. 错误: {e}")
            # import traceback
            # traceback.print_exc()

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
            
        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
            miou_out_path=".temp_miou_out", eval_flag=True, period=1, letterbox_image=True): # Added letterbox_image
        super(EvalCallback, self).__init__()
        
        self.net                = net
        # self.input_shape 期望从 train.py 传入 (H,W) 或 (C,H,W)
        # 我们将在 get_miou_png 中处理这个问题
        self.input_shape        = input_shape 
        self.num_classes        = num_classes
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.miou_out_path      = miou_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        self.letterbox_image    = letterbox_image # Store letterbox_image flag
        
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        self.mious      = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def letterbox(self, image, size): # size 是 (W,H) for PIL
        iw, ih  = image.size
        w, h    = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image, nw, nh


    def get_miou_png(self, image): # Renamed from get_miou_png to get_pred_png_for_miou
        image       = image.convert('RGB')
        image_old_w, image_old_h = image.size
        
        # 根据 self.input_shape 的长度确定 target_h, target_w
        if len(self.input_shape) == 2: # 假设是 (H, W)
            target_h, target_w = self.input_shape[0], self.input_shape[1]
        elif len(self.input_shape) == 3: # 假设是 (C, H, W)
            target_h, target_w = self.input_shape[1], self.input_shape[2]
        else:
            raise ValueError(f"EvalCallback: 不支持的 input_shape 格式: {self.input_shape}. 期望 (H,W) 或 (C,H,W).")

        if self.letterbox_image:
            image_data, nw, nh  = self.letterbox(image, (target_w, target_h)) 
        else:
            image_data  = image.resize((target_w, target_h), Image.BICUBIC)
            nw, nh = target_w, target_h 


        image_data  = np.expand_dims(np.transpose(np.array(image_data, np.float32) / 255.0, (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            outputs = self.net(images)
            # outputs 可能是列表 (来自 DataParallel) 或元组 (seg, r1, r3)
            # 我们需要分割图 pr
            if isinstance(outputs, list) and len(outputs) > 0:
                # DataParallel 返回一个列表，每个元素是对应GPU的输出
                # 如果模型本身返回元组，那么 outputs[0] 也是元组
                if isinstance(outputs[0], tuple):
                    pr = outputs[0][0]  # 取第一个GPU输出元组中的第一个元素（分割图）
                else:
                    pr = outputs[0] # 假设列表的第一个元素就是分割张量
            elif isinstance(outputs, tuple):
                pr = outputs[0] # 假设元组的第一个元素是分割图
            else:
                pr = outputs # 假设 outputs 直接是分割张量
            
            # 确保 pr 是正确的分割张量形状，例如 [1, num_classes, H, W] 或 [num_classes, H, W]
            if pr.dim() == 4 and pr.shape[0] == 1: # 如果 batch size 是 1
                 pr = pr.squeeze(0) # 移除 batch 维度, pr 变为 [num_classes, H, W]
            elif pr.dim() != 3: # 如果不是 [C,H,W] (在 squeeze(0)之后)
                raise ValueError(f"模型输出中提取的pr形状不符合预期: {pr.shape}")
            
            # pr 此时应为 [num_classes, H, W]
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy() # pr 变为 [H, W, num_classes]
            
            if self.letterbox_image:
                # letterbox 缩放后的实际内容区域是在原始图片中的投影
                # 注意 nw, nh 是 letterbox 函数返回的 letterbox 内部图像的尺寸
                # 我们需要基于原始图像 old_w, old_h 和目标尺寸 target_w, target_h 计算偏移
                # 这里 nw, nh 已经是 image_data（输入到网络的图像）中有效内容的尺寸
                # 而 image_data 的尺寸是 target_w, target_h
                
                # 计算 image_data 中有效内容区域的边界
                offset_h = (target_h - nh) // 2
                offset_w = (target_w - nw) // 2
                
                # 从预测结果 pr (尺寸为 target_h, target_w) 中裁剪出有效区域
                pr = pr[offset_h : offset_h + nh, offset_w : offset_w + nw]

            # 将裁剪后的 pr 缩放到原始图像尺寸
            pr = cv2.resize(pr, (image_old_w, image_old_h), interpolation = cv2.INTER_LINEAR)
            # pr 现在是 (image_old_h, image_old_w, num_classes)
            pr = pr.argmax(axis=-1) # pr 现在是 (image_old_h, image_old_w) 的类别索引图
    
        # 为了mIoU计算，我们返回单通道的类别索引图 (NumPy array)
        # 将 NumPy array 转换为 PIL Image 对象进行保存
        image_out_indexed = Image.fromarray(pr.astype(np.uint8))
        return image_out_indexed

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            
            # --- 修正路径 ---
            gt_dir      = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")
            jpeg_dir    = os.path.join(self.dataset_path, "VOC2007/JPEGImages/") 
            # --- 修正路径结束 ---

            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                image_path  = os.path.join(jpeg_dir, image_id + ".jpg") # 使用修正后的 jpeg_dir
                try:
                    image = Image.open(image_path)
                except FileNotFoundError:
                    print(f"警告: 图片 {image_path} 未找到, 跳过.")
                    continue
                
                # get_pred_png_for_miou 返回的是单通道索引图 (PIL Image)
                pred_indexed_img = self.get_miou_png(image)
                pred_indexed_img.save(os.path.join(pred_dir, image_id + ".png"))
                        
            print("Calculate miou.")
            _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None) 
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='val miou') 

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou (%)') 
            plt.title('Validation Miou Curve') 
            plt.legend(loc="lower right") 

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print(f"Epoch: {epoch}, Validation mIoU: {temp_miou:.2f}%")
            # show_results 在训练过程中可能太慢或占用太多资源，通常在训练结束后单独评估
            # show_results(self.miou_out_path, gt_dir, pred_dir, self.image_ids, self.num_classes)