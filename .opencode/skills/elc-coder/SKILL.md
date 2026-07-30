---
name: elc-coder
description: 车载平衡滚球运动控制系统 (ELC 2026 H题) 项目专用开发技能。使用OpenCV/Numpy实现实时视觉跟踪。当需要修改 camera, processor, tuner, main, demo, serial 模块，或编写图像处理、串口通信、配置管理相关代码时自动触发。Use ONLY when editing files under src/ or config/.
---

# ELC Coder — 车载平衡滚球运动控制系统

项目根目录：`/home/glennade/elc_2026/`

## 上下文（预加载，减少 token 消耗）

```python
import cv2           # 不要用 from cv2 import ...
import numpy as np   # 不要用 from numpy import ...
import yaml
```

---

## 项目结构

| 文件 | 职责 |
|---|---|
| `src/main.py` | `class Application` — 主控，编排 Camera + BallDetector + Tuner + Visualizer |
| `src/camera.py` | `class Camera` — cv2.VideoCapture + V4L2，支持内参加载与去畸变 |
| `src/processor.py` | `class BallDetector`, `class ROIManager`, `class Visualizer` — 核心检测算法+ROI裁剪+可视化 |
| `src/tuner.py` | `class Tuner` — cv2.createTrackbar 实时调参，从 config.yaml 读默认值 |
| `src/serial.py` | 空占位文件，预留串口通信（发给MCU） |
| `demo.py` | 独立单文件 demo，无依赖 config 文件 |
| `config/config.yaml` | 相机参数(device_id=2, 640x480, 120fps)、bar_length_cm=25.0、trackbar默认值 |
| `config/camera_matrix.yaml` | 相机内参矩阵+畸变系数（当前为空） |

---

## 检测算法 Pipeline（核心逻辑）

1. **背景减除** `cv2.absdiff(gray_roi, bg)`
2. **二值化** `cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)`
3. **形态学闭运算** `cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_ellipse)`
4. **垂直投影** `np.sum(binary, axis=0)` → 1D 信号
5. **峰值定位** `np.argmax(projection)` → X像素坐标
6. **换算厘米** `(x_pixel - center) * (bar_length_cm / roi_width)`

`demo.py` 中包含独立可运行的完整算法实现，可作为参考。

---

## 代码规范（强制执行）

### 风格
- **注释用中文**，变量/函数/类名用英文
- **4空格缩进**，不用 Tab
- **f-string** 格式化字符串
- **类架构** 在 src/ 中，过程式在 demo.py 中

### 命名
- 类：`PascalCase`（例：`BallDetector`, `ROIManager`）
- 函数/方法：`snake_case`（例：`get_frame`, `load_config`）
- 常量：`UPPER_SNAKE_CASE`
- 私有方法：`_leading_underscore`

### 静态方法
工具类（`ROIManager`, `Visualizer`, `Tuner`）使用 `@staticmethod`，不维护实例状态。

---

## 性能约束（实时系统）

- 相机目标帧率：**120 FPS**（640x480）
- 每帧处理耗时必须控制在 **8ms 以内**
- 禁止逐像素 Python 循环，所有图像操作必须用 **cv2 或 numpy 向量化**
- 背景减除前先 `cv2.cvtColor` 转灰度，避免三通道计算
- ROI 裁剪后再处理，减少无效计算面积
- 避免每帧创建/销毁大对象（预分配 numpy buffer）

---

## 当前问题（修改前检查）

1. `src/main.py:29` — 硬编码的 config 绝对路径，应在类初始化时接受相对路径参数
2. `src/camera.py` — 加载 camera_matrix.yaml 失败时仅 print 警告，没有 fallback 参数
3. `BallDetector.detect()` 未对 `roi` 为 None 或空图像做防御性检查
4. `demo.py` 与 `src/processor.py` 算法逻辑重复，修改检测算法时需要同步两个文件
5. 无 `.gitignore`，`__pycache__/` 被误提交
6. `config/camera_matrix.yaml` 为空，需要补充标定数据或添加示例模板

---

## 新增代码要求

### 必需
- 中文注释说明关键步骤
- 防御性编程：对相机断连、无效帧、空 ROI 做 if 判断
- 变量名自解释（如 `binary_mask`, `projection_1d` 而非 `b`, `p`）
- 使用项目已有工具类（`ROIManager`, `Visualizer`），不重复造轮子

### 禁止
- 引入新第三方依赖（项目当前仅依赖 opencv-python, numpy, pyyaml）
- `// ...` 或 `# TODO` 占位符——必须给出完整可运行代码
- 硬编码绝对路径（现有的除外，新增代码必须用相对路径或参数传入）

### 测试
- 项目当前无测试框架。编写测试时自行创建 `tests/` 目录
- 优先用 `pytest`（非强制，取决于后续是否引入 pytest 依赖）
- 覆盖：正常帧、空帧、极端光照（全黑/全白）下的检测输出

---

## 最小化 Token 消耗

回答时遵循：
1. **直接给代码**，不解释算法原理（除非用户明确提问）
2. **一次给出完整函数/类**，不分段输出
3. **不输出项目已有的代码**，只展示改动部分或用 `edit` 工具直接修改
4. 回答控制在 **4行以内**（不含代码块），不寒暄不总结
