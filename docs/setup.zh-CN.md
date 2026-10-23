# ChessRobot 部署与操作指南

**简体中文** · [English](setup.en.md) · [日本語](setup.ja.md) · [项目首页](../README.md)

本指南针对当前第三代入口，需要实体机器人及其专用标定。仓库未提供整机纯仿真开关。第四代训练属于独立研发任务，见[技术说明](technical-notes.zh-CN.md)。

## 环境

- Windows 10/11、64 位 Python 3.10 或更高版本。
- 可使用 CUDA 的 PyTorch 及兼容 NVIDIA 环境。当前视觉直接调用 `model.to('cuda')`，没有 CPU 回退。
- USB 摄像头、WCH CH347 驱动、匹配的 64 位 `CH347DLLA64.DLL`、PCA9685、四路舵机及电磁铁/继电器组件。

`requirements.txt` 给出 PyTorch、Ultralytics、OpenCV、NumPy、PyQt5 的版本下限，并非验证过的环境锁定文件。复现时应记录实际可用版本。历史视觉脚本还可能需要 Pillow 和字体文件。

在 PowerShell 中：

```powershell
git clone https://github.com/TJS-Git-Hub/ChessRobot.git
cd ChessRobot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import torch, cv2; print(torch.__version__); print(torch.cuda.is_available()); print(cv2.__version__)"
```

若 CUDA 不可用，先解决 PyTorch/驱动环境，再启动当前视觉程序。安装依赖成功不代表硬件运行已验证。

## 外部文件与模型

| 文件 | 路径 | 说明 |
| :--- | :--- | :--- |
| WCH DLL | 项目根目录 `CH347DLLA64.DLL` | 使用匹配的厂商库和驱动。DLL 被 Git 忽略，干净克隆不提供此文件。 |
| 棋子检测器 | `recognition/models/best.pt` | 仓库包含权重，但类别映射需匹配所选视觉版本，不能仅凭文件名替换为根目录 `yolo11n.pt`。 |
| 可选走法权重 | `game/aaa.pth` | 当前目录树不包含；缺少时决策模块随机选择候选。 |
| 标定数组 | `recognition/models/H_matrix.npy`、`ROI_corners.npy` | 早期脚本使用；当前主入口仍交互标定。 |
| YOLO 训练数据 | `recognition/data/data.yaml` 及其引用目录 | 原图与训练输出被 Git 排除；训练前核对路径和标签定义。 |

第三代检测器用于定位，棋子身份来自标准开局和历史状态。文件名不能证明模型为 14 类、经过量化或达到某项成绩。

## 硬件与标定

| 接口或参数 | 代码依据 |
| :--- | :--- |
| CH347 设备索引 0，PCA9685 地址 `0x40` | `hardware.py` |
| 通道 0 / 4 / 8 / 12 对应底座 / 肩 / 肘 / 腕 | `config.py` |
| GPIO2 掩码 `0x04` 控制继电器 | `hardware.py` |
| 各轴参考脉宽和上下限 | `config.py` 的 `SERVO_CONFIG` |
| 棋盘原点、间距、盘面及安全高度 | `config.py` |
| 横向、半径、高度经验补偿 | `kinematics.py` |

根据实际装配确认棋盘、舵机供电、公共参考及继电器接线，确保棋盘和弃子区位于已验证的运动范围。控制器构造时调用 `go_to_wait()`，会立即发出运动指令。

以下工具的作用不同：

```powershell
python tools/calibrate_servo.py
python tools/test_electromagnet.py
python tools/calc_reach.py
```

- `calibrate_servo.py` 会驱动舵机，支持粗细脉宽调整，最后打印应写入 `config.py` 的数值，不自动保存。
- `test_electromagnet.py` 会实际控制继电器和电磁铁。
- `calc_reach.py` 复用当前 `board_to_physical` 和 `solve_ik`，不打开硬件；当前配置有11/90个目标被拒绝，存在拒绝时以非零状态退出。几何通过也不认证舵机行程、碰撞或抓放效果。

尝试全盘对弈前，先统一几何参数，并使用活动逆解逐点检查目标。当前远端点可能无法通过可达性检查，仓库参数也不应直接当作另一台机器的标定值。

## CLI 操作

1. 准备检测权重、DLL、驱动与标定后的硬件。当前 `VideoStream` 默认摄像头索引 `1`，请求 1280 × 720；采集脚本使用索引 `0`，应按实际设备统一。
2. 按标准开局摆好 32 枚棋子，机械臂位于黑方底线后方。
3. 启动：

   ```powershell
   python main.py
   ```

4. 在 OpenCV 窗口按左上、右上、右下、左下顺时针点击棋盘四角。
5. 必要时手动指定朝向：`1` 表示黑方在图像远端/上侧，`2` 表示黑方在近端/下侧。亮度比例已按单通道像素数修正；真实光照下的自动判向仍需实测。
6. 等待初始化。人类执红先行；走棋后移开手，聚焦视觉窗口，按 **Space** 提交盘面。
7. 系统选择应手，执行动作，返回待命并重设视觉基准。吃子流程先把被吃棋子转移到盘外弃子区。
8. 在视觉窗口按 `q` 或中断 CLI 结束。关闭流程请求停止并关闭接口，不再发起返回待命动作；软件停止不等同于硬件断电急停。

锁定基准时不要改变棋子数量或位置。吃子引发 ID 混淆时，程序可能提示短暂拿起进攻棋子、放回后重新确认。若实体与软件盘面失去一致性，应恢复一致盘面后重新初始化；目前不能完全自主恢复。

## GUI 原型

```powershell
python ui_main.py
```

界面提供“开始对弈”、软件停止、日志和 Space/1/2 快捷键。点击开始会构造同一套硬件调度器。

GUI 现在接收标定及追踪画面，按实际帧、等比例缩放和留白换算点击位置；棋盘使用加锁快照并解析 `(side, kind)` 元组。停止请求会结束视频流及插补，关闭时不再发起回零动作。仍需实机检查 Qt/OpenCV 窗口协同、摄像头后端和标定精度。

## 常见问题

| 现象 | 检查项 |
| :--- | :--- |
| DLL 加载或 CH347 打开失败 | Windows/Python/DLL 位数、厂商驱动、库位置、USB 连接和设备索引。 |
| 摄像头打不开或画面错误 | 索引、设备占用、连接及实际分辨率。 |
| CUDA 错误 | PyTorch 构建、NVIDIA 环境和 CUDA 可用性，当前视觉没有自动 CPU 路径。 |
| 无朝向或基准盘面 | 手动 1/2、角点顺序、检测模型、完整开局和检测稳定性。 |
| 人类走棋后无动作 | 窗口焦点、空格确认、棋子数量、ID 和上一基准。 |
| 走法近似随机或不合理 | `game/aaa.pth` 是否存在及兼容，规则检查也未达到完整竞赛裁定。 |
| GUI 视频空白或棋盘报错 | 确认采集帧、标定点击映射与日志；回调和元组棋盘接口已修复，Qt/相机联调仍需实测。 |
| 不可达或落子偏移 | 活动几何、棋盘原点与高度、逆解补偿、舵机标定和脉宽限制；运行当前 `tools/calc_reach.py`，并结合实际量测。 |

## 记录复现结果

记录代码版本、Python/依赖版本、检测与决策权重身份、相机设置、棋盘几何、补偿参数及实际使用的入口。落子试验注明样本量及中心覆盖/距离判据；识别评测使用独立实拍数据，并区分交点准确率和整盘准确率。已有记录及后续节点见[技术说明](technical-notes.zh-CN.md)。

2026-10-24：移位前预检起点与终点的落点、8 mm悬停点和安全高度；吃子还预检弃子区及己方起点，已知不可达时在首次移动或吸取前拒绝。每段插补仍单独预检，未证明整条抓放路径无碰撞。9项离线回归通过；标定参数未改，11/90目标仍被当前IK拒绝，需实测后才能修正。
