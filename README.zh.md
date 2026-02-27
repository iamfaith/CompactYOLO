# Compact YOLO: 面向低资源环境的轻量化目标检测

[English README](README.md)

Compact YOLO 是一个轻量化的目标检测模型，专为 **CPU 和边缘设备部署**而设计。它在保持检测精度的同时，大幅降低了计算量和内存占用，适用于实时监控与低功耗场景。

## ✨ 特点
- **轻量化结构**：采用 GhostConv 与 C3Ghost 模块，减少 FLOPs 与参数量。  
- **高效激活函数**：用 ReLU 替代 SiLU，提升 CPU 推理速度。  
- **推理优化**：引入 Post-Sigmoid Filter 与 External Reshape/Transpose，降低延迟并增强量化兼容性。  
- **实测性能**：在桌面 CPU 上平均推理仅需 **3.34 ms/张**，在 Raspberry Pi 3B 上约 **90.47 ms/张**。  

## 📊 性能对比
在 COCO 人体检测子集上的实验表明，Compact YOLO 在 **mAP@0.5** 上优于其他轻量化 YOLO 变体，同时保持更快的推理速度。  

| 模型 | Precision | Recall | mAP@0.5 | CPU Latency |
|------|-----------|--------|---------|-------------|
| YOLOv5n6 (640) | 0.518 | 0.540 | 0.492 | 20.29 ms |
| YOLOv5n6 (192) | 0.627 | 0.277 | 0.455 | 3.09 ms |
| FastestDet (352) | 0.461 | 0.285 | 0.460 | 7.33 ms |
| **Compact YOLO (640)** | **0.587** | **0.429** | **0.533** | **3.34 ms** |

## 📂 模型文件说明
- `best.onnx` → **最终优化版模型**（推荐使用，论文中的主结果）。  
- `best.pt` → PyTorch 格式权重。  
- `full_best.onnx` → **未使用 Post-Sigmoid Filter** 的版本（论文中的 *w/o Post-Sigmoid Filter* 对照实验）。  
- `transpose_best.onnx` → **Post-Sigmoid Filter + 内部 Reshape/Transpose** 的版本（论文中的另一对照实验）。  
- `last.pt` → 训练过程中的最后一次权重保存。  

## 🚀 快速开始
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("best.onnx")
outputs = session.run(None, {"images": input_tensor})
