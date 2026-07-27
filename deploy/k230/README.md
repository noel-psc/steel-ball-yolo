# K230 部署

本目录遵循嘉楠 K230 SDK 的两部分流程：主机使用 `nncase` 将 ONNX
量化为 `kmodel`，设备端基于 SDK 的
[`object_detect_yolov8n`](https://github.com/kendryte/k230_sdk/tree/main/src/reference/ai_poc/object_detect_yolov8n)
C++ 示例获取摄像头画面、调用 KPU 并显示结果。

## 转换

安装与 K230 SDK 匹配的 nncase 后，在项目根目录运行：

```bash
uv pip install nncase nncase-kpu onnxsim scikit-learn
bash deploy/k230/convert_kmodel.sh \
  artifacts/steel-ball-yolo26n.onnx \
  data/yolo/images/train \
  artifacts/steel-ball-yolo26n.kmodel 100
```

转换脚本固定输入为 RGB、`float32`、NCHW、`1x3x640x640`，且使用 100 张
校准图片。校准图片必须来自实机环境且不应包含验证集图片。

## 设备端接入

YOLO26 默认 End-to-End ONNX 的输出行为是 `(1, 300, 6)`，每行格式：

```text
[x1, y1, x2, y2, confidence, class_id]
```

只有 `class_id == 0` 且置信度达到阈值的行表示 `steel_ball`；无需 NMS。
`steel_ball_detect.py` 是这一约定的可测试参考实现。将该逻辑移植进 K230
SDK 示例的后处理后，选择最高置信度框并输出其中心点。

若 nncase 不能转换 YOLO26，请以同一数据集重新训练 `yolo11n.pt`，导出
`--fallback-yolo11` ONNX，并在官方 YOLOv8n 示例的 NMS 后处理基础上适配
单类别输出。YOLO26 权重不能直接转为 YOLO11。

## 验收

同一张静态图应先在 `validate_onnx.py` 验证，再与 K230 结果对比钢球是否
检出、类别、中心点和置信度。摄像头模式稳定后再接入小车控制逻辑。
