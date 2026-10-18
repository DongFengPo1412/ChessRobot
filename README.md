<div align="center">

# 基于机器视觉与深度卷积神经网络的四自由度象棋对弈机器人系统
### Full-Stack Autonomous 4-DOF Chinese Chess Robotic Arm System with Deep Vision & Kinematics

[ English ](README_EN.md) | [ 简体中文 ](README.md) | [ 日本語 ](README_JA.md)

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B%20(CUDA%20Accelerated)-ee4c2c.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Vision%20Perception%20V11-00a8ff.svg?style=for-the-badge)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B%20Homography-5c3ee8.svg?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Kinematics](https://img.shields.io/badge/Kinematics-4--DOF%20Analytical%20IK-success.svg?style=for-the-badge)](https://github.com/)
[![Bilibili: Live Demo](https://img.shields.io/badge/Bilibili-Live%20Demo-fb7299?style=for-the-badge&logo=bilibili&logoColor=white)](https://www.bilibili.com/video/BV1aXLn6HEQd)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<p align="center">
  本项目是一套工业级、硬实时闭环的<b>全栈智能中国象棋人机对弈机器人系统</b>（代号“智弈·灵手”）。<br/>
  系统深度融合了 <b>YOLO 深度卷积神经网络目标检测</b>、<b>单应性矩阵（Homography）空间几何透视校正</b>、<br/>
  <b>PyTorch ChessNet 深度卷积博弈决策网络</b> 与 <b>四自由度（4-DOF）解析空间逆运动学（IK）控制律</b>，<br/>
  构建了从物理视觉感知、意图识别、棋局推理到电磁铁微距防碰撞机械拾取落子的完整信息物理闭环。
</p>

</div>

---

## 目录
- [1. 工程背景与团队职责分工](#1-工程背景与团队职责分工)
- [2. 硬件实机演示与全链路验证](#2-硬件实机演示与全链路验证)
  - [2.1 2K 高清实机动态对弈过程展示](#21-2k-高清实机动态对弈过程展示)
  - [2.2 信息物理系统（CPS）全栈架构与控制台](#22-信息物理系统cps全栈架构与控制台)
- [3. 算法演进历程与早期版本溯源](#3-算法演进历程与早期版本溯源)
- [4. 核心数学模型与运动学控制律](#4-核心数学模型与运动学控制律)
  - [4.1 单应性透视投影变换几何校正模型](#41-单应性透视投影变换几何校正模型)
  - [4.2 四自由度关节型机械臂解析逆运动学（IK）求解](#42-四自由度关节型机械臂解析逆运动学ik求解)
  - [4.3 微距空间三段式防刮蹭碰撞避免控制律](#43-微距空间三段式防刮蹭碰撞避免控制律)
  - [4.4 滑动窗口时序多帧共识投票稳态判定](#44-滑动窗口时序多帧共识投票稳态判定)
  - [4.5 PCA9685 12-Bit 高精度 PWM 脉宽映射模型](#45-pca9685-12-bit-高精度-pwm-脉宽映射模型)
- [5. 硬件电气连接与总线定义](#5-硬件电气连接与总线定义)
- [6. 仓库代码目录规范](#6-仓库代码目录规范)
- [7. 快速复现与部署运行指南](#7-快速复现与部署运行指南)
- [8. 系统操作指南（对弈交互与标定）](#8-系统操作指南对弈交互与标定)
- [9. 开源许可与致谢](#9-开源许可与致谢)

---

## 1. 工程背景与团队职责分工

本项目为大学《智能控制算法设计 / 机器人控制工程》综合课程设计重点课题（第七组）。针对传统棋类机器人感知易受环境光干扰、机械臂拾放易碰倒相邻棋子以及多线程视控延迟大等核心工程瓶颈，团队从零构建了一体化闭环实物对弈平台。

### 团队工程职责分工

| 开发者 | 核心工程角色 | 核心研发模块与技术落地 |
| :--- | :--- | :--- |
| **田峻硕 (Tian Junshuo)** | **团队组长 / 机械与运动控制负责人** | <ul><li>四自由度机械臂硬件结构装配与各关节调谐（RDS3235 35kg·cm 高扭矩金属数码舵机）</li><li>WCH CH347 高速 USB-I2C 桥接与 PCA9685 12-bit PWM 硬件底层驱动实现（`hardware.py`）</li><li>空间解析逆运动学（Analytical IK）推导与前馈重力补偿模型架构（`kinematics.py`）</li><li>微距空间三段式防刮蹭无碰撞抓放轨迹规划控制律（`control.py`）</li></ul> |
| **刘超然 (Liu Chaoran)** | **视觉感知与系统集成负责人 / 答辩主讲人** | <ul><li>OpenCV 单应性变换（$3 	imes 3$ Homography）四点交互标定与毫米级物理坐标映射（`localization/find_chess_new.py`）</li><li>轻量级 YOLO 目标检测模型部署与 LAB 色彩空间亮度（L 通道）黑红棋子快速判定引擎</li><li>35 帧滑动窗口时序共识抗遮挡滤波与 32 颗棋子吃子死锁防御机制</li><li>基于 `move_queue` 的端到端多线程解耦闭环架构构建与 PyQt5 交互控制台（`ui_main.py`）</li><li>项目答辩演示文稿设计、技术设计报告统稿与答辩现场汇报</li></ul> |
| **张子琛 (Zhang Zichen)** | **博弈决策与规则引擎负责人** | <ul><li>中国象棋 10x9 完整棋盘规则引擎构建、合法走法生成与历史回滚机制（`game/board.py`）</li><li>基于 PyTorch 深度卷积神经网络（`ChessNet`，14通道输入/8100维输出）的局面评估与走子决策模型研发（`game/ai.py`）</li><li>棋盘物理网格空间映射与初始阵型标准常量规范定义（`game/constants.py`）</li></ul> |
| **邢艺龙 (Xing Yilong)** | **工装搭建与数据集工程负责人** | <ul><li>多光照与多角度真实棋盘图像数据集采集与标注（`recognition/collect_real_data.py`）</li><li>定制亚克力导磁棋盘制作、棋子导磁金属贴片工艺与电磁吸盘末端工装搭建</li><li>电气配线、硬件测试台架搭建与多轴实物联调支持（`tools/`）</li></ul> |

---

## 2. 硬件实机演示与全链路验证

以下展示从 2K 60fps 实机对弈演示视频（`电子科技大学_智能控制算法设计“四轴机械臂”_答辩.mp4`）中截取的真实动态对弈过程，以及系统的底层信息物理系统（CPS）架构与交互控制台。

* 完整 2K 60fps 实机人机对弈全闭环动态演示视频已发布在 Bilibili：  
  👉 **[点击观看实机演示视频：四自由度象棋对弈机器人端到端闭环实战](https://www.bilibili.com/video/BV1aXLn6HEQd)**  
  *(完整演示人类落子视觉感知、单应性透视校正、ChessNet 深度学习决策解算、4 轴解析逆运动学规划及电磁防碰撞精准拾放全流程)*

### 2.1 2K 高清实机动态对弈过程展示

<div align="center">

| 1. 人类走子意图捕获与抗遮挡稳态感知 | 2. 机械臂空间逆解下探与电磁精准吸附 | 3. 防碰撞轨迹转移、落子与归位就绪 |
| :---: | :---: | :---: |
| <img src="docs/images/demo_live_human_move.jpg" width="310px" alt="人类走子识别"/> | <img src="docs/images/demo_live_arm_pickup.jpg" width="310px" alt="机械臂精准吸取棋子"/> | <img src="docs/images/demo_live_arm_place.jpg" width="310px" alt="机械臂精准落子"/> |
| 工业镜头 30 FPS 连续追踪，35 帧滑动窗口彻底消除手部遮挡噪声，准确捕获红方移子 | ChessNet 完成走法解算，解析逆运动学驱动 4 轴舵机垂直切入，电磁吸铁牢固吸附 | 强制提升至 70mm 安全巡航平面平移，结合 8mm 微距纯垂直缓冲避让相邻棋子，垂直平稳落子 |

</div>

### 2.2 信息物理系统（CPS）全栈架构与控制台

<div align="center">
  <img src="docs/images/system_architecture.png" width="900px" alt="端到端信息物理系统全栈微架构拓扑图"/>
  <p><b>图 1：四自由度象棋对弈机器人端到端信息物理多线程解耦架构</b></p>
</div>

<div align="center">

| 完整对弈实物测试台架全貌 (`demo_system_cover.png`) | 单应性空间透视几何校正 (`demo_homography_transform.png`) | YOLO 目标检测卷积神经网络 (`demo_yolo_detection.png`) |
| :---: | :---: | :---: |
| <img src="docs/images/demo_system_cover.png" width="300px" alt="实物台架全貌"/> | <img src="docs/images/demo_homography_transform.png" width="300px" alt="单应性变换几何校正"/> | <img src="docs/images/demo_yolo_detection.png" width="300px" alt="YOLO 目标检测"/> |
| 4-DOF 铝合金高强度机械臂、顶部俯视工业广角相机与定制亚克力导磁象棋盘 | 消除相机倾角投影畸变，实现像素坐标向物理世界坐标系（毫米）的精准映射 | 实时推理识别 14 类红黑棋子标签与二维空间网格，实时推理识别棋子位置，单帧推理耗时仅 12ms，辅以 LAB 亮度通道高速分拣红黑阵营 |

</div>

<div align="center">
  <img src="docs/images/demo_ui_console.png" width="880px" alt="PyQt5 异步人机交互控制台"/>
  <p><b>图 2：PyQt5 异步多线程交互仪表盘（包含实时视频遥测、虚拟数字棋盘、单步日志与手动标定模式）</b></p>
</div>

---

## 3. 算法演进历程与早期版本溯源

本项目在研发过程中经历了四代迭代演进，每一步都针对上一代的真实物理缺陷进行了针对性攻坚：

1. **第一代原型 — 经典霍夫圆检测与色彩分割 (`find_chess_simple.py` 早期雏形)**：
   - 采用经典 OpenCV 霍夫圆变换（Hough Circles）检测棋子圆形轮廓，结合 HSV 颜色阈值分割红黑棋子。
   - **工程痛点**：室内台灯与自然采光产生的投影阴影极易造成伪圆漏检误检，反光严重时圆心偏移超过 $5\,\text{mm}$，导致机械臂吸盘频繁扑空。
2. **第二代演进 — 定制微调 YOLO 目标检测 (`find_chess.py` / V6 架构)**：
   - 引入轻量级 YOLO 目标检测模型直接回归棋子包围框与 14 类字形标签。
   - **工程痛点**：由于广角镜头边缘存在透视压缩，大角度下外观极为接近的字形（如“車”与“卒”、“馬”与“兵”）在远端容易产生类别混淆。
3. **第三代演进 — 空间解耦与工控实机部署架构 (`find_chess_new.py` / V15.4 生产架构)**：
   - **几何与色彩解耦**：YOLO 仅回归棋子几何圆心；阵营判定则提取 ROI 区域内 LAB 色彩空间 $L$ 通道亮度分布（深黑棋子 $L < 45$），实现极低算力消耗的高速二值划分。
   - **楚河汉界物理死区过滤**：针对棋盘中央“楚河 汉界”印刷字形引发 YOLO 误检的物理痛点，在透视变换坐标系中设置 $4.35 < Y_{\text{grid}} < 4.65$ 的判定死区，彻底剔除幽灵棋子。
   - **状态机与防死锁设计**：引入初始阶段自动锁定（$len \ge 10$ 帧）与对弈走棋阶段人工信号触发机制（通过空格键或 UI 按钮置位 `force_detect_move = True`），彻底杜绝人手遮挡导致的 ID 粘连跳变；并设计 `on_capture_event` 动态递减棋子总数（32 $\to$ 31），实现实战 100% 稳态闭环。
4. **第四代架构 — 全栈异步 CPS 闭环与前沿视觉科研拓展 (`main.py` & `ui_main.py` & 科研分支)**：
   - **实机全栈闭环**：基于 `move_queue` 线程安全队列重构为多线程异步解耦架构（视觉感知、ChessNet 推理、机械臂运动完全并发）；结合五阶段微距防刮蹭规划与重力前馈补偿，实现硬实时物理对弈。
   - **前沿视觉科研升华**：在此基础上，视觉团队针对任意复杂倾斜光照场景进一步研发了无人工干预的端到端视觉系统（RTMPose 四角点自动几何矫正 + Swin-Transformer2 输出 $10 \times 9 \times 16$ 棋局分类特征张量），在 17 组真实非标棋局场景下识别精度全面超越商业级 App“天天象棋”与“棋路”，形成了从边缘工控落地到前沿 SOTA 的完整技术图谱。

---

## 4. 核心数学模型与运动学控制律

### 4.1 单应性透视投影变换几何校正模型

由于摄像机安装角度存在微小倾角，原始图像存在非线性射影畸变。系统通过求解 $3 \times 3$ 单应性矩阵 $\mathbf{H}$，建立图像像素坐标 $(u, v)$ 与棋盘绝对毫米物理坐标 $(X, Y)$ 的线性射影映射：

$$
\begin{bmatrix} X' \\ Y' \\ Z' \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

绝对物理坐标通过齐次坐标去归一化解算：

$$
X = \frac{X'}{Z'} = \frac{h_{11} u + h_{12} v + h_{13}}{h_{31} u + h_{32} v + h_{33}}, \quad Y = \frac{Y'}{Z'} = \frac{h_{21} u + h_{22} v + h_{23}}{h_{31} u + h_{32} v + h_{33}}
$$

系统在标定阶段采集棋盘四个基准角点，通过奇异值分解（SVD）直接求解超定线性方程组，消除镜头透视倾角。虽然开环金属舵机传动齿隙导致远端机械重复定位误差在 $\pm 5.0\text{ mm}$ 左右，但得益于端部直径 $20\text{ mm}$ 的电磁吸盘与棋子顶部导磁贴片形成的宽裕几何容差，结合逆运动学前馈垂度补偿后，实测全场抓放综合成功率稳定达到 **92%**。

---

### 4.2 四自由度关节型机械臂解析逆运动学（IK）与前馈补偿控制律

机械臂由底座水平回转关节（Joint 1）、大臂俯仰关节（Joint 2）、小臂俯仰关节（Joint 3）与腕部姿态保持关节（Joint 4）构成。

```mermaid
graph LR
    Base["底座回转 J1 (θ₁)"] --> Shoulder["大臂俯仰 J2 (θ₂)"]
    Shoulder --> Elbow["小臂俯仰 J3 (θ₃)"]
    Elbow --> Wrist["腕部姿态 J4 (θ₄)"]
    Wrist --> End["末端电磁吸盘 (强制垂直向下 φ = -90°)"]
```

#### 1. 真实物理连杆几何参数 (`config.py`)
系统严格依据物理实机装配尺寸标定连杆长度：
- 底座回转中心高：$L_1 = 73.0\,\text{mm}$
- 大臂有效连杆长：$L_2 = 231.0\,\text{mm}$
- 小臂有效连杆长：$L_3 = 231.0\,\text{mm}$
- 腕部关节至吸盘端长：$L_4 = 66.0\,\text{mm}$

#### 2. 非线性机械形变与前馈补偿模型 (`kinematics.py`)
由于机械臂伸展半径大且采用开环金属舵机驱动，远端在重力作用下存在弹性悬臂下垂，且底座舵机传动存在轻微几何非线性。团队在理论逆解前引入了实测前馈补偿控制律：

- **X 轴横向几何比例放大前馈**（补偿边缘列回转角不足，纠正约 10% 几何内收）：
  $$X_{\text{comp}} = X \times K_x \quad (K_x = 1.125)$$
- **底座目标合成工作半径与伸展修正**：
  $$R_{\text{target}} = \sqrt{X_{\text{comp}}^2 + Y^2}$$
  $$R_{\text{comp}} = R_{\text{target}} \times (1 - K_r) \quad (K_r = 0.06)$$
- **悬臂自重弹性下垂垂度前馈补偿**（随作业半径线性放大抬升）：
  $$\text{comp}_z = Z + (R_{\text{target}} \times K_z) \quad (K_z = 0.12)$$

#### 3. 闭式几何解析逆解推导
在末端吸盘强制保持垂直向下（$\phi = -90^\circ$）的约束下，系统将空间 3D 运动解耦至俯仰平面：

1. **底座方位回转角 $\theta_1$**：
   $$\theta_1 = \text{atan2}(X_{\text{comp}}, Y)$$
   转换为舵机控制基准角：
   $$t_1^{\text{deg}} = \text{degrees}(\theta_1) + 90.0^\circ$$

2. **俯仰平面等效高度差与弦长求解**：
   $$\Delta Z = (\text{comp}_z + L_4) - L_1$$
   $$D = \sqrt{R_{\text{comp}}^2 + \Delta Z^2}$$

3. **余弦定理求解大臂俯仰内角 $\alpha$ 与小臂弯折内角 $\gamma$**：
   $$\cos(\alpha) = \frac{L_2^2 + D^2 - L_3^2}{2 L_2 D}, \quad \cos(\gamma) = \frac{L_2^2 + L_3^2 - D^2}{2 L_2 L_3}$$
   $$\alpha = \arccos\left(\text{clip}(\cos(\alpha), -1, 1)\right), \quad \gamma = \arccos\left(\text{clip}(\cos(\gamma), -1, 1)\right)$$
   $$\beta = \text{atan2}(\Delta Z, R_{\text{comp}})$$

4. **大臂、小臂绝对角与腕部自适应姿态解算**：
   $$t_2^{\text{deg}} = \text{degrees}(\beta + \alpha)$$
   $$t_3^{\text{deg}} = t_2^{\text{deg}} - (180.0^\circ - \text{degrees}(\gamma))$$
   $$t_4^{\text{relative}} = -90.0^\circ - t_3^{\text{deg}}$$

该闭式解析解在 CPU 上单次解算耗时 $< 1\,\mu\text{s}$，无雅可比矩阵奇异性风险，结合前馈补偿模型将末端定位误差显著抑制在可抓取容限内。

---

### 4.3 微距空间五阶段防刮蹭碰撞避免控制律

由于中国象棋棋子半径约 $18\,\text{mm}$，密集排列时相邻棋子间隙常不足 $10\,\text{mm}$。若机械臂直接沿斜向逼近，吸盘极易刮倒外围棋子。系统设计了严格的微距航路点控制律（`control.py`）：

$$
\mathbf{P}(t) = \begin{cases} 
(X_{\text{src}}, \, Y_{\text{src}}, \, Z_{\text{safe}}), & t \in [0, t_1) \quad (\text{快速平移至起点上方安全巡航面}) \\
(X_{\text{src}}, \, Y_{\text{src}}, \, Z_{\text{micro}}), & t \in [t_1, t_2) \quad (\text{降至棋子上方 8mm 微距缓冲层，腕部姿态调平}) \\
(X_{\text{src}}, \, Y_{\text{src}}, \, Z_{\text{board}}), & t \in [t_2, t_3) \quad (\text{慢速纯垂直贴合吸附，电磁铁通电，延时 0.5s}) \\
(X_{\text{src}}, \, Y_{\text{src}}, \, Z_{\text{micro}}), & t \in [t_3, t_4) \quad (\text{小臂主导纯垂直微抬 8mm，杜绝水平偏摆带倒邻子}) \\
(X_{\text{src}}, \, Y_{\text{src}}, \, Z_{\text{safe}}), & t \in [t_4, t_5) \quad (\text{垂直升入安全巡航高度}) \\
(X_{\text{dst}}, \, Y_{\text{dst}}, \, Z_{\text{safe}}), & t \in [t_5, t_6) \quad (\text{巡航高度无碰撞平移至目标点上方}) \\
(X_{\text{dst}}, \, Y_{\text{dst}}, \, Z_{\text{micro}}), & t \in [t_6, t_7) \quad (\text{降至目标微距缓冲层，稳住垂直姿态}) \\
(X_{\text{dst}}, \, Y_{\text{dst}}, \, Z_{\text{board}}), & t \in [t_7, t_8) \quad (\text{慢速垂直下探落子，电磁铁断电，延时 0.6s 消磁}) \\
(X_{\text{dst}}, \, Y_{\text{dst}}, \, Z_{\text{micro}}), & t \in [t_8, t_9) \quad (\text{纯垂直微抬脱离棋盘}) \\
(X_{\text{dst}}, \, Y_{\text{dst}}, \, Z_{\text{safe}}), & t \in [t_9, t_{10}) \quad (\text{垂直复位至巡航面，平滑归位至待命点})
\end{cases}
$$

系统的真实物理基准高度配置（`config.py`）：
- 安全巡航高度：$Z_{\text{safe}} = 70.0\,\text{mm}$；
- 棋盘落子作业面高度：$Z_{\text{board}} = -33.0\,\text{mm}$（末端低于机械臂基准面）；
- 微距防碰撞缓冲高度：$Z_{\text{micro}} = Z_{\text{board}} + 8.0 = -25.0\,\text{mm}$；
- 机械臂固定收容待命点：$\mathbf{P}_{\text{wait}} = (220.0, \, 50.0, \, 150.0)\,\text{mm}$；
- 棋盘外吃子丢弃区（墓地）：$\mathbf{P}_{\text{graveyard}} = (220.0, \, 120.0, \, -13.0)\,\text{mm}$。

---

### 4.4 滑动窗口时序多帧共识与人机协同状态机

在人类玩家走棋过程中，手部遮挡、棋子移动瞬态和阴影突变会导致单帧检测出现虚警。系统在时序维度建立长度为 $W = 35$ 帧（约 1.1 秒）的滑动窗口，对棋盘 $10 \times 9 = 90$ 个交叉格点进行能量投票：

$$
S_{\text{consensus}}(i, j) = \arg\max_{c \in \mathcal{C}} \sum_{t=1}^{W} \mathbb{I}(\hat{C}_t(i, j) == c)
$$

只有当某一棋子类别 $c$ 在窗口内的累计票数满足 $\text{Votes}(c) \ge \theta_{\text{votes}}$（$\theta_{\text{votes}} = 28$）时，系统才认定格点 $(i, j)$ 进入稳态。

为彻底消除人手遮挡与 ID 跳变误触，系统在实机对弈中引入**分阶段协同状态机**：
1. **阶段 A（初始盘面锁定）**：启动阶段自动缓存滑动窗口，当全盘识别棋子数稳定在 32 颗（或预期值）且窗口缓存连续平稳超过 10 帧时，自动锁定初始局面基准；
2. **阶段 B（走棋人机确认驱动）**：人类玩家走棋完毕且手移开后，通过按下**键盘空格键**（或在 PyQt5 界面点击**走法确认按钮**）触发单步检测信号（`force_detect_move = True`）。系统首先核验盘面棋子总数有效性（平移仍为 $N$ 颗，被吃子为 $N-1$ 颗），再比对基准快照精准提取唯一 `lost` 与 `gain` 坐标推入 `move_queue`；若检测到人手遮挡或 ID 粘连异常则在控制台警示，确保了物理闭环对弈 100% 动作可靠。

---

### 4.5 PCA9685 12-Bit 高精度 PWM 脉宽映射模型

PCA9685 硬件 PWM 驱动器具备 12-bit 分辨率（4096 级计数值），输出基频 $f_{\text{PWM}} = 50\,\text{Hz}$（周期 $T = 20\,\text{ms}$）。

脉宽 $T_{\text{pulse}} \in [0.5\,\text{ms}, 2.5\,\text{ms}]$ 对应计数值 $\text{Tick}$：

$$
\text{Tick} = \left\lfloor \frac{T_{\text{pulse}}(\text{ms})}{20\,\text{ms}} \times 4096 \right\rfloor
$$

RDS3235 舵机转角 $\theta \in [0^\circ, 270^\circ]$ 与脉宽映射线性方程为：

$$
T_{\text{pulse}}(\theta) = 0.5\,\text{ms} + \frac{\theta}{270^\circ} \times 2.0\,\text{ms}
$$

$$
\text{Tick}(\theta) = \left\lfloor \frac{0.5 + \frac{\theta}{270} \times 2.0}{20} \times 4096 \right\rfloor = \left\lfloor 102.4 + \theta \times 1.517 \right\rfloor
$$

配合各关节零位死区补偿表，角度控制分辨率高达 $0.066^\circ$，有力保障了末端定位精度。

---

## 5. 硬件电气连接与总线定义

系统的物理硬件电气拓扑结构如下表所示：

| 硬件模组 | 物理接口 | 驱动协议 | 宿主控制引脚 | 电气特征与控制参数 |
| :--- | :--- | :--- | :--- | :--- |
| **PCA9685 舵机驱动板** | SDA / SCL | 硬件 I2C (100kHz 标准模式) | CH347 D0(SCL) / D1(SDA) | 16 通道 12 位分辨率，基准频率 50Hz ($T = 20\,\text{ms}$) |
| **底座旋转舵机 (J1)** | PWM 通道 0 | 50Hz PWM 方波 | PCA9685 输出端 0 | RDS3235 金属数码舵机，水平方位角控制 |
| **大臂俯仰舵机 (J2)** | PWM 通道 4 | 50Hz PWM 方波 | PCA9685 输出端 4 | RDS3235 35kg·cm 高扭矩数码舵机，大臂抬升控制 |
| **小臂俯仰舵机 (J3)** | PWM 通道 8 | 50Hz PWM 方波 | PCA9685 输出端 8 | 20kg·cm 精密金属数码舵机，伸展半径与高度控制 |
| **腕部姿态舵机 (J4)** | PWM 通道 12 | 50Hz PWM 方波 | PCA9685 输出端 12 | 180° 金属舵机，强制保持末端吸盘垂直朝下 |
| **电磁拾子吸盘** | MOS 直驱信号 | CH347 原生 GPIO | CH347 GPIO2 (Pin 10 / DTR0) | 5V DC 强力电磁铁，额定吸力 $> 5\,\text{N}$，掩码 0x04 直接通断 |
| **工业俯视摄像头** | USB 2.0 / UVC | DirectShow 协议 | PC USB 3.0 高速口 | 1280×720 @ 30 FPS，定焦低畸变镜头 |
| **WCH CH347 桥接器** | USB Type-C | 480Mbps 高速 USB | PC 宿主机 USB 口 | 硬件 USB 转 I2C/GPIO 高速工业桥接器 |

---

## 6. 仓库代码目录规范

```bash
ChessRobot/
├── localization/                   # 空间视觉几何校正与棋盘标定核心模块
│   ├── find_chess_new.py           # 生产级核心视觉管道 (ChessVisionV15.4: YOLO定位 + LAB颜色判别 + 死区过滤 + 35帧滑窗共识)
│   ├── find_chess_simple_plus.py   # 异步视频流与高灵敏度检测演进原型
│   ├── find_chess_simple.py        # 早期轻量化视觉算法测试原型 (ChessVisionV7)
│   └── find_chess.py               # 经典全分类视觉引擎历史版 (ChessVisionEngineV6)
├── recognition/                    # 模型训练、数据采集与标定参数存储
│   ├── train.py                    # YOLO 目标检测模型微调训练脚本
│   ├── collect_real_data.py        # 现场实机多角度多光照图像数据集采集工具
│   ├── test_gpu.py                 # CUDA / GPU 加速与推理环境验证脚本
│   ├── yolo26n.pt                  # 迭代微调权重文件
│   └── models/                     # 标定几何参数与模型权重持久化存储 (best.pt, H_matrix.npy, ROI_corners.npy)
├── game/                           # 中国象棋博弈决策大脑与规则引擎
│   ├── board.py                    # 10x9 完整象棋物理规则基座 (合法走法生成 / 走子校验 / 局面回滚)
│   ├── ai.py                       # 基于 PyTorch 卷积神经网络的局面评估决策引擎 (ChessNet, 14通道输入/8100维输出)
│   ├── constants.py                # 初始棋局阵型、棋盘坐标映射与网格几何常量
│   └── play.py                     # 独立终端人机对弈测试主入口
├── tools/                          # 硬件调试与底层标定工具集
│   ├── calibrate_servo.py          # 4 自由度机械臂各关节舵机零位微秒脉宽标定工具 (通道 0, 4, 8, 12)
│   ├── test_electromagnet.py       # WCH CH347 原生 GPIO2 (DTR0) 电磁铁吸附通断测试脚本
│   └── calc_reach.py               # 机械臂极限作业半径与可达空间几何校验工具
├── control.py                      # 机械臂运动控制中枢 (微距空间五阶段防刮蹭抓放控制律)
├── kinematics.py                   # 4 自由度空间解析逆运动学求解器 (含 X 轴增益与重力下垂前馈模型)
├── hardware.py                     # WCH CH347 高速 USB 桥接 (100kHz I2C + GPIO2) + PCA9685 12-bit PWM 硬件驱动
├── config.py                       # 系统全局连杆物理几何尺寸 (L1-L4)、舵机通道与棋盘坐标基准
├── main.py                         # 终端命令行主程序 (基于 move_queue 的异步端到端对弈闭环)
├── ui_main.py                      # 基于 PyQt5 的异步多线程人机交互控制台 (实时视频 / 虚拟棋盘 / 标定)
├── yolo11n.pt                      # 定制微调的轻量级中国象棋检测网络权重
├── requirements.txt                # Python 依赖库环境清单
├── LICENSE                         # MIT 官方开源许可证
├── README.md                       # 简体中文系统技术规格说明书
├── README_EN.md                    # English Technical Specification
└── README_JA.md                    # 日本語技術仕様书
```

---

## 7. 快速复现与部署运行指南

### 环境依赖
- **操作系统**：Windows 10 / 11 (64-bit) 或 Ubuntu 20.04 / 22.04 LTS
- **Python 环境**：Python 3.10+ (推荐通过 Anaconda 创建隔离虚拟环境)
- **CUDA 算力**：NVIDIA GPU (推荐 GTX 1650 以上，用于 30 FPS 实时目标检测加速)
- **硬件依赖**：WCH CH347 驱动 (`CH347DLLA64.DLL`) 已内置于工程根目录

### 1. 虚拟环境构建与依赖安装
```bash
# 1. 克隆代码仓库
git clone https://github.com/TJS-Git-Hub/ChessRobot.git
cd ChessRobot

# 2. 创建并激活 Python 3.10 隔离环境
conda create -n chess_robot python=3.10 -y
conda activate chess_robot

# 3. 安装依赖库与 PyTorch (CUDA 版本)
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. 硬件连接与测试
1. 将 WCH CH347 转换器通过 USB 插入电脑，连接 PCA9685 与 4 轴机械臂供电（5V 5A 独立稳压电源）。
2. 调试机械臂各轴舵机零位脉宽偏置（基准 90° 竖直指天姿态）：
   ```bash
   python tools/calibrate_servo.py
   ```
3. 测试 WCH CH347 原生 GPIO2 电磁铁通断与吸附动作：
   ```bash
   python tools/test_electromagnet.py
   ```
4. 校验机械臂全盘物理作业半径与可达空间余量：
   ```bash
   python tools/calc_reach.py
   ```

### 3. 启动对弈系统
- **启动 PyQt5 全图形化交互仪表盘**：
  ```bash
  python ui_main.py
  ```
- **启动轻量无界面 CLI 终端对弈模式**：
  ```bash
  python main.py
  ```

---

## 8. 系统操作指南（对弈交互与标定）

1. **棋盘角点标定**：
   - 首次启动前确保摄像头俯视对准棋盘，系统进入交互标定模式。
   - 用鼠标左键在视频窗口中依次顺时针点击棋盘四个角点（1.左上 $\to$ 2.右上 $\to$ 3.右下 $\to$ 4.左下），系统自动通过 SVD 分解求解单应性矩阵 $\mathbf{H}$，持久化保存至 `recognition/models/`，完成物理坐标映射并自动确认黑方视角。
2. **摆放初始棋局**：
   - 将红黑双方 32 枚棋子摆放在标准起始位置。
   - 点击 `开始对弈`（Start Match），系统开始连续采集视频帧并在控制台回显当前稳态棋局。
3. **人类走子与人机协同触发**：
   - 人类执红棋先行走子。走完后手离开棋盘区域。
   - 按下**键盘空格键**（或在控制台点击**确认走子**按钮），系统快速校验棋子总数并比对快照，自动解析红方移动并推入 `move_queue`，ChessNet 随即解算黑方最优应手。
4. **机器人应子执行**：
   - 机械臂从待命点出发，自动提升至 $Z_{\text{safe}}=70.0\,\text{mm}$ 巡航高度，下探至 $Z_{\text{micro}}=-25.0\,\text{mm}$ 缓冲层并垂直吸取棋子，沿无碰撞三维倒 U 轨迹平移至目标点垂直落子，随后平滑归位至待命点 $\mathbf{P}_{\text{wait}}$。

---

## 9. 开源许可与致谢

本项目依据 **[MIT License](LICENSE)** 协议开源。

- **核心工程团队（第七组）**：
  - **田峻硕 (Tian Junshuo)**：机械机构组装、底层硬件驱动与逆运动学规划
  - **刘超然 (Liu Chaoran)**：视觉感知检测、单应性几何校正、多线程架构与 PyQt5 UI 控制台
  - **张子琛 (Zhang Zichen)**：博弈算法、中国象棋规则引擎与棋局推演
  - **邢艺龙 (Xing Yilong)**：工装制作、数据集标注增强与系统联调
- **致谢**：
  - 感谢《智能控制算法设计》课程教学组专家评委的悉心指导。
  - 感谢 Ultralytics YOLO、OpenCV 与 WCH 开源硬件社区的技术生态支持。

---

<div align="center">
  <b>ChessRobot — 智弈·灵手</b>，融合现代计算机视觉与多轴精密控制的智能人机对弈工程典范。
</div>
