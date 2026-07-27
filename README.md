# steel-ball-yolo

面向 K230 电赛小车的单类别钢球检测训练与部署流程。默认模型为
YOLO26n 的 End-to-End ONNX 输出；若 nncase 转换不兼容，可切换至
YOLO11n 的传统 NMS 路径。

## 当前范围

- 类别：`steel_ball`（ID 为 `0`）
- 训练：CUDA 主机上的 Ultralytics YOLO26n，固定 `640x640`
- 部署：ONNX → nncase → K230 `kmodel`
- 输出：是否检测到钢球、置信度、钢球框中心坐标

公开图片仅用于验证流程。电赛使用前，必须用小车实机摄像头采集并标注
不同地面、反光、光照、遮挡和运动模糊条件下的数据。

## 安装

```bash
uv sync --group dev
```

随后使用 `uv run` 执行项目命令，例如 `uv run pytest -v`。`requirements.txt`
保留给不使用 uv 的环境。

## 数据和标注

下载可追溯来源的图片清单后，人工确认钢球并以 YOLO 格式标注：

```bash
python scripts/download_open_images.py --manifest sources.jsonl
python scripts/prepare_dataset.py \
  --images-dir data/raw/approved-images \
  --labels-dir data/raw/labels \
  --output-dir data/yolo
```

每张图片对应一个同名 `.txt` 标签文件；每行格式为：

```text
0 center_x center_y width height
```

坐标必须归一化到 `0..1`。

## CUDA 训练

```bash
python scripts/train.py --data data/yolo/dataset.yaml --epochs 100 --device 0
```

默认从 `yolo26n.pt` 微调。先确认 `nvidia-smi` 可用，并在训练输出中确认
脚本检测到了 CUDA GPU。

## 导出和验证 ONNX

```bash
python scripts/export_onnx.py \
  --weights runs/detect/train/weights/best.pt \
  --output artifacts/steel-ball-yolo26n.onnx

python scripts/validate_onnx.py \
  --model artifacts/steel-ball-yolo26n.onnx \
  --image data/yolo/images/val/example.jpg
```

若 K230 的 nncase 不能转换 YOLO26 End-to-End ONNX：

```bash
python scripts/export_onnx.py \
  --weights runs/detect/train/weights/best.pt \
  --output artifacts/steel-ball-yolo11n.onnx \
  --fallback-yolo11
```

## K230

部署转换与设备端运行约定见 [deploy/k230/README.md](deploy/k230/README.md)。

## 验证

```bash
python -m pytest -v
python -m compileall scripts deploy/k230
```

## 许可证

本项目采用 [GPL-3.0-only](LICENSE) 许可证。
K230 steel ball detection training and deployment workflow
