
---

## 📄 README.md （English）


# Compact YOLO: Optimizing Object Detection for Low-Resource Environments

[中文说明](README.zh.md)

Compact YOLO is a lightweight object detection model designed for **CPU and edge deployment**. It reduces FLOPs and memory usage while maintaining detection accuracy, making it practical for real-time monitoring and low-power scenarios.

## ✨ Features
- **Lightweight structure**: GhostConv and C3Ghost modules reduce FLOPs and parameters.  
- **Efficient activation**: ReLU replaces SiLU for faster CPU inference.  
- **Inference optimization**: Post-Sigmoid Filter and External Reshape/Transpose reduce latency and improve quantization compatibility.  
- **Benchmark results**: Runs at **3.34 ms per image** on desktop CPU and **90.47 ms per image** on Raspberry Pi 3B.  

## 📊 Performance Comparison
Compact YOLO achieves higher **mAP@0.5** than other lightweight YOLO variants while keeping faster inference speed.  

| Model | Precision | Recall | mAP@0.5 | CPU Latency |
|-------|-----------|--------|---------|-------------|
| YOLOv5n6 (640) | 0.518 | 0.540 | 0.492 | 20.29 ms |
| YOLOv5n6 (192) | 0.627 | 0.277 | 0.455 | 3.09 ms |
| FastestDet (352) | 0.461 | 0.285 | 0.460 | 7.33 ms |
| **Compact YOLO (640)** | **0.587** | **0.429** | **0.533** | **3.34 ms** |

## 📂 Model Files
- `best.onnx` → **Final optimized model** (recommended, main result in paper).  
- `best.pt` → PyTorch weights.  
- `full_best.onnx` → **Without Post-Sigmoid Filter** (baseline experiment in paper).  
- `transpose_best.onnx` → **Post-Sigmoid Filter + Internal Reshape/Transpose** (alternative experiment).  
- `last.pt` → Last checkpoint during training.  

## 🚀 Quick Start
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("best.onnx")
outputs = session.run(None, {"images": input_tensor})
