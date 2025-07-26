# Seg-Water: A Hybrid Transformer-CNN with Deformable Attention for Precise Water Body Segmentation

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.8+](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is the official PyTorch implementation of the paper **"Multi-Scale Feature Fusion in a Transformer-CNN Framework for Water Body Delineation from Remote Sensing Data"**.

We propose a novel hybrid Transformer-CNN framework named **Seg-Water**, designed for high-accuracy and high-robustness water body segmentation from remote sensing imagery. Seg-Water effectively combines the global context-capturing capabilities of Transformers with the local detail extraction advantages of CNNs, achieving state-of-the-art performance on multiple public water body segmentation benchmark datasets.

---

### **Figure 1: Visualization of Segmentation Results**
![alt text](<img/Figure 1.png>)
((a) original remote sensing input, (b) reference ground truth mask, (c) Seg-Water model’s output, (d) HRNet’s output, (e) SegFormer’s output, (f) PSPNet’s output, (g) DANet’s output, (h) DeepLabV3’s output)
---

## Core Innovations and Model Architecture

At the heart of Seg-Water is a sophisticated encoder-decoder architecture designed to overcome common challenges in water body segmentation, such as irregular boundaries, multi-scale variations, and background interference.

### 1. Hybrid Encoder-Decoder Structure

-   **Encoder (MiT-B2)**: We employ the **MixVisionTransformer (MiT)** from the SegFormer series as the encoder. It efficiently extracts multi-scale features by progressively reducing the spatial resolution of feature maps while expanding the channel dimensions. Its self-attention mechanism is particularly adept at capturing long-range dependencies, which is crucial for understanding the global context of large water bodies.

-   **Decoder (Innovative CNN-based Modules)**: Unlike the simple MLP decoder in SegFormer, our main innovations are concentrated in a powerful CNN-based decoder that refines and fuses features with high fidelity.

### 2. DASCI Module: The Core Innovation

The **Deformable Attention with Spatial and Channel Interaction (DASCI)** module is the cornerstone of our decoder. It is designed to perform adaptive geometric feature extraction and enhance key spatial information.

-   **Deformable Context Pyramid (DCP)**: This sub-module, within an ASPP-like multi-branch structure, replaces standard convolutions with **Deformable Convolution Networks (DCN)**. By learning sampling point offsets from the data itself, it can flexibly adapt its receptive field to the natural, irregular shapes of water bodies, significantly improving boundary segmentation accuracy.
-   **Spatial Perception and Channel Information Interaction (SPCII)**: This is a novel attention module that efficiently models dependencies in both spatial and channel dimensions. It utilizes adaptive 1D convolutions to capture cross-channel correlations and generates decoupled spatial attention maps (`S_h` and `S_w`) to recalibrate feature importance along the height and width dimensions, ultimately enhancing key features while suppressing noise.

### 3. Connect Module

To further enhance the structural integrity and spatial continuity of the segmentation results, we introduce the **Connect Module** at the final stage of the decoder. This module employs parallel branches with standard and dilated convolutions to predict adjacency relationships between pixels at different distances.

### 4. Composite Loss Function

To train the model effectively, we designed a multi-component composite loss function to synergistically optimize both pixel-level classification accuracy and macro-level structural awareness:
-   **Segmentation Loss (`L_seg`)**: A combination of **Weighted Cross-Entropy (WCE)**, for handling class imbalance, and **Lovász-Softmax Loss**, for directly optimizing the IoU metric.
-   **Connectivity Loss (`L_con`)**: This is a novel loss term derived from our Connect Module. It encourages the model to generate smoother and more realistically connected water bodies by penalizing discontinuities and ruptures in the predicted masks.
-   **Dice Loss (`L_dice`)**: An optional auxiliary loss term that also helps to mitigate severe foreground-background class imbalance issues.

---

### **Figure 2: Overall Architecture of the Seg-Water Model**
![alt text](<img/Figure 2.png>)
Overall structure of the Seg-Water model
---

## Performance

Seg-Water consistently surpasses current mainstream segmentation models on two public datasets. Experiments demonstrate that the integration of the DASCI and Connect modules is crucial for achieving superior accuracy and robustness.

### Quantitative Results

**Table 1: Performance on the Satellite Images of Water Bodies Dataset**

| Model | MIoU | Precision | Recall | F1-score | aAcc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DeepLabV3  | 81.55% | 89.91% | 89.78% | 89.84% | 91.30% |
| DANet  | 82.10% | 90.23% | 90.08% | 90.15% | 91.75% |
| PSPNet  | 82.85% | 90.66% | 90.58% | 90.62% | 92.13% |
| SegFormer | 84.21% | 91.50% | 91.36% | 91.43% | 92.88% |
| HRNet  | 84.93% | 91.88% | 91.84% | 91.86% | 93.10% |
| **Seg-Water (Ours)**| **85.91%**| **92.33%**| **92.36%**| **92.34%**| **93.45%**|

**Table 2: Performance on the 2020 Gaofen Challenge Dataset**

| Model | MIoU | Precision | Recall | F1-score | aAcc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DeepLabV3  | 83.24% | 92.52% | 88.61% | 90.42% | 95.39% |
| DANet  | 83.81% | 92.03% | 89.67% | 90.80% | 95.49% |
| PSPNet  | 84.11% | 92.75% | 89.42% | 90.98% | 95.63% |
| SegFormer | 85.24% | 92.61% | 90.87% | 91.71% | 95.90% |
| HRNet  | 85.56% | 92.56% | 91.30% | 91.92% | 95.98% |
| **Seg-Water (Ours)**| **86.67%**| **92.88%**| **93.39%**| **92.61%**| **96.21%**|

### Ablation Study

Extensive ablation studies prove that each of our proposed innovative modules (DASCI, DCN, SPCII) makes an indispensable contribution to the final performance of the model.

---

### **Figure 3: Ablation Study or Feature Map Visualization**
![alt text](<img/Figure 3.png>)
((a) original remote sensing input, (b) reference ground truth mask, (c) the segmentation result produced by the full Seg-Water network, (d) the segmentation result after removing the SPCII attention, (e) the segmentation result after removing the DCP module, (f) the segmentation result after removing the DASCI module)
---

## Installation and Configuration

### 1. Environment Requirements
- Python 3.8+
- PyTorch 1.8+
- CUDA 10.2+ (for GPU usage)

### 2. Clone Repository
```bash
git clone https://github.com/zhuangzmr/Seg-Water.git
cd SegWater
