<div align="center">

# Autonomous 4-DOF Chinese Chess Robotic Arm System with Deep Vision & Kinematics
### 基于机器视觉与深度学习的四自由度中国象棋对弈机器人系统

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
  This project is an independently developed <b>4-DOF Autonomous Chinese Chess Robotic Arm System</b> (Codename: "Zhiyi Lingshou").<br/>
  An overhead industrial camera tracks human chess moves, mapping 2D pixel coordinates to physical grid positions via homography rectification;<br/>
  upon move inference by the AI decision engine, the 4-DOF articulated manipulator executes open-loop trajectory planning,<br/>
  utilizing an electromagnetic end-effector for micro-space collision-free pickup, transfer, and precision piece placement.
</p>

</div>

---

## Table of Contents
- [1. System Overview & Physical Fabrication](#1-system-overview--physical-fabrication)
- [2. Engineering Responsibilities](#2-engineering-responsibilities)
- [3. Hardware Demonstrations & Placement Accuracy](#3-hardware-demonstrations--placement-accuracy)
- [4. Modular Architecture & Algorithmic Evolution](#4-modular-architecture--algorithmic-evolution)
  - [4.1 Mechanical & Motion Control Module (Lead: Tian Junshuo, Co-tuning: Liu Chaoran)](#41-mechanical--motion-control-module-lead-tian-junshuo-co-tuning-liu-chaoran)
  - [4.2 Vision Perception Module & 4-Gen Evolution (Lead: Liu Chaoran)](#42-vision-perception-module--4-gen-evolution-lead-liu-chaoran)
  - [4.3 Game Decision & Rule Engine Module (Lead: Zhang Zichen)](#43-game-decision--rule-engine-module-lead-zhang-zichen)
- [5. Hardware Electrical Topology & Bus Definitions](#5-hardware-electrical-topology--bus-definitions)
- [6. Repository Directory Structure](#6-repository-directory-structure)
- [7. Quickstart & Deployment Guide](#7-quickstart--deployment-guide)
- [8. Operating Guide (Interactive Calibration & Play)](#8-operating-guide-interactive-calibration--play)
- [9. Open Source License & Acknowledgments](#9-open-source-license--acknowledgments)

---

## 1. System Overview & Physical Fabrication

This platform integrates custom physical fixtures, embedded drivers, vision recognition, and robotic trajectory control.

### 1.1 Physical Chessboard & Handcrafted Magnetic Pieces
- **Beige Board**: Selected a rigid $35 \times 35\,\text{cm}$ beige Chinese chessboard with crisp grid lines and anti-glare matte coating.
- **Handmade Magnetic Pieces**:
  - Printed high-contrast red and black Chinese chess character templates (Microsoft YaHei font, bold) sized at exactly $3.0\,\text{cm}$ diameter;
  - Manually precision-cut all 32 character circles;
  - Procured $3.0\,\text{cm}$ diameter circular magnetic iron discs, manually adhering the printed labels onto the discs to create custom magnetic pieces.
- **End-Effector & Actuation**:
  - Outfitted with a 5V DC suction electromagnet driven by an onboard relay switch;
  - During pickup, the arm descends vertically over the target piece, the relay engages to energize the electromagnet, securing the magnetic piece;
  - Upon reaching the target destination, the relay opens to de-energize and smoothly place the piece.

### 1.2 Benchmark Accuracy & Experimental Verification
Core control benchmarks focus on **micro-space collision avoidance** and **grid-intersection centering coverage**.  
Across 20 continuous consecutive real-world robot moves:
- **18 moves perfectly covered** the intersection center point;
- **2 moves showed slight displacement** without losing piece grip;
- **Center-covering success rate reached 90%**, operating smoothly without tipping neighboring pieces.

---

## 2. Engineering Responsibilities

All software and hardware were developed collaboratively by four core team members. Presentation design, project documentation, and defense presentations were shared equally across technical domains.

| Member | Engineering Role | Core Deliverables & Contributions |
| :--- | :--- | :--- |
| **Tian Junshuo (田峻硕)** | **Team Lead / Mechanical & Motion Control Lead** | <ul><li>4-DOF robotic arm physical assembly, servo selection, and mechanical joint tuning;</li><li>WCH CH347 high-speed USB bridge and PCA9685 12-bit PWM hardware driver implementation (`hardware.py`);</li><li>Derivation of analytical closed-form inverse kinematics (IK) equations (`kinematics.py`);</li><li>5-phase collision-free micro-approach trajectory planning control law (`control.py`).</li></ul> |
| **Liu Chaoran (刘超然)** | **Vision Perception Lead** | <ul><li><b>Independently conducted 100% of physical dataset collection, cleaning, and annotation</b>;</li><li>Spearheaded all 4 generations of vision architecture, developing lightweight YOLO piece localization & red/black classification;</li><li>Implemented OpenCV homography ( $3 \times 3$ matrix) 4-point perspective calibration & physical grid mapping (`localization/find_chess_new.py`);</li><li>Developed the multithreaded PyQt5 interactive operator dashboard (`ui_main.py`);</li><li><b>Assisted in robotic motion control parameter tuning</b> (including cantilever gravity droop elevation feedforward compensation).</li></ul> |
| **Zhang Zichen (张子琛)** | **Game Decision & Rule Engine Lead** | <ul><li>Engineered complete 10x9 Chinese chess rule engine, move validation, and history rollback (`game/board.py`);</li><li>Developed PyTorch CNN evaluation network (`ChessNet`, 14-channel input / 8100-dim action space) (`game/ai.py`);</li><li>Standardized chessboard coordinate representations and initial game setup constants (`game/constants.py`).</li></ul> |
| **Xing Yilong (邢艺龙)** | **System Integration & Fixtures Lead** | <ul><li>Managed hardware wiring, power distribution, and electrical safety standards;</li><li>Fabricated custom magnetic chessboard, assembled handmade magnetic pieces, and built electromagnet toolhead fixtures;</li><li>Constructed multi-axis physical test bench, assisting in multi-axis motion integration (`tools/`).</li></ul> |

---

## 3. Hardware Demonstrations & Placement Accuracy

The complete 2K 60fps autonomous closed-loop gameplay demonstration video is published on Bilibili:

👉 **[Watch Live Demo on Bilibili: Full-Stack Autonomous 4-DOF Chinese Chess Robotic Arm Match](https://www.bilibili.com/video/BV1aXLn6HEQd)**

### 3.1 Live Gameplay Demonstrations (Authentic GIFs)

<div align="center">

| Demo GIF 1: Arm High-Level Cruising & Vertical Pickup | Demo GIF 2: Collision-Free Transfer & Precision Placement |
| :---: | :---: |
| <img src="docs/images/demo_robot_arm_1.gif" width="420px" alt="Arm Movement and Pickup Demo"/> | <img src="docs/images/demo_robot_arm_2.gif" width="420px" alt="Arm Placement and Standby Demo"/> |
| Responding to move dispatch, arm departs standby, vertically aligns with piece, and energizes electromagnet | Cruising along safety elevation, descending vertically at destination to release piece with 90% center coverage |

</div>

### 3.2 Authentic Live Match Capture Snapshots

<div align="center">

| Authentic HD Match Capture: Electromagnetic Pickup (`demo_live_arm_pickup.jpg`) | Authentic HD Match Capture: Precision Release (`demo_live_arm_place.jpg`) |
| :---: | :---: |
| <img src="docs/images/demo_live_arm_pickup.jpg" width="420px" alt="Electromagnetic Pickup"/> | <img src="docs/images/demo_live_arm_place.jpg" width="420px" alt="Precision Release"/> |

</div>

---

## 4. Modular Architecture & Algorithmic Evolution

The codebase is organized into three decoupled technical modules: **Mechanical & Motion Control**, **Vision Perception**, and **Game Decision & Rule Engine**.

---

### 4.1 Mechanical & Motion Control Module (Lead: Tian Junshuo, Co-tuning: Liu Chaoran)

The manipulator features a 4-DOF articulated design: Base Yaw (J1), Shoulder Pitch (J2), Elbow Pitch (J3), and Wrist Pitch (J4), with an electromagnetic toolhead constrained strictly vertical ( $\phi = -90^\circ$ ).

#### 1. Physical Linkage Parameters (`config.py`)
- Base rotation height: $L_1 = 73.0\,\text{mm}$
- Shoulder link length: $L_2 = 231.0\,\text{mm}$
- Elbow link length: $L_3 = 231.0\,\text{mm}$
- Wrist-to-effector length: $L_4 = 66.0\,\text{mm}$

#### 2. Non-Linear Kinematic Feedforward Compensation (`kinematics.py`)
To counteract cantilever gravitational deflection and gear backlash from open-loop digital servos:

**1. X-Axis Lateral Scaling Feedforward**:

$$
X_{\text{comp}} = X \times K_x \quad (K_x = 1.125)
$$

**2. Synthesized Working Radius Adjustment**:

$$
R_{\text{target}} = \sqrt{X_{\text{comp}}^2 + Y^2}, \quad R_{\text{comp}} = R_{\text{target}} \times (1 - K_r) \quad (K_r = 0.06)
$$

**3. Cantilever Sag Elevation Feedforward**:

$$
\text{comp}_z = Z + (R_{\text{target}} \times K_z) \quad (K_z = 0.12)
$$

#### 3. Analytical Geometric Inverse Kinematics
Enforcing end-effector vertical constraint decodes 3D coordinates into planar pitch equations:

**Step 1: Base Azimuth** (Solve base heading angle $\theta_1$ and map to servo angle $t_1^{\text{deg}}$)

$$
\theta_1 = \text{atan2}(X_{\text{comp}}, Y), \quad t_1^{\text{deg}} = \text{degrees}(\theta_1) + 90.0^\circ
$$

**Step 2: Elevation Delta & Link Chord**

$$
\Delta Z = (\text{comp}_z + L_4) - L_1, \quad D = \sqrt{R_{\text{comp}}^2 + \Delta Z^2}
$$

**Step 3: Law of Cosines for Shoulder Angle** $\alpha$ **& Elbow Angle** $\gamma$

$$
\cos(\alpha) = \frac{L_2^2 + D^2 - L_3^2}{2 L_2 D}, \quad \cos(\gamma) = \frac{L_2^2 + L_3^2 - D^2}{2 L_2 L_3}
$$

**Step 4: Joint Angles**

$$
\beta = \text{atan2}(\Delta Z, R_{\text{comp}})
$$

$$
t_2^{\text{deg}} = \text{degrees}(\beta + \alpha), \quad t_3^{\text{deg}} = t_2^{\text{deg}} - (180.0^\circ - \text{degrees}(\gamma)), \quad t_4^{\text{relative}} = -90.0^\circ - t_3^{\text{deg}}
$$

#### 4. Micro-Space 5-Phase Collision-Free Trajectory Planning (`control.py`)
To prevent the suction head from colliding with tightly spaced neighbor pieces:
- **Cruise Transfer**: Fast transfer to safety cruising altitude $Z_{\text{safe}} = 70.0\,\text{mm}$ ;
- **Micro-Hovering**: Descend to 8mm micro-buffer layer above piece ( $Z_{\text{micro}} = -25.0\,\text{mm}$ ) to stabilize wrist posture;
- **Slow Plunge & Suction**: Gentle descent to board plane $Z_{\text{board}} = -33.0\,\text{mm}$ , relay energizes electromagnet, holding for 0.5s;
- **Forced Vertical Lift**: Pure vertical elevation of 8mm back to $Z_{\text{micro}}$ , isolating lateral tipping;
- **Cruise Elevation**: Lift to $Z_{\text{safe}}$ and transfer to destination, executing reciprocal micro-approach placement.

#### 5. Motion Control Optimization Roadmap
Future work involves upgrading multi-axis interpolation profiles with S-curve acceleration filtering to further dampen mechanical vibrations during long-reach trajectories.

---

### 4.2 Vision Perception Module & 4-Gen Evolution (Lead: Liu Chaoran)

Liu Chaoran independently spearheaded 100% of dataset collection, annotation, and computer vision algorithm evolution across four generations:

#### Generation 1: Hough Circles & Synthetic Font OCR
- **Concept**: OpenCV Hough Circles localized circular piece boundaries; a lightweight CNN trained on 1,000+ synthetic bold Microsoft YaHei character crops handled OCR.
- **Pain Points**:
  - **Severe Sensitivity to Lighting**: Ambient reflections caused Hough parameter failure. Lower thresholds missed dark pieces; higher thresholds triggered false circles everywhere;
  - **Position Drift**: Surface glares induced $> 5\,\text{mm}$ center shifts, causing arm grasping failure.

#### Generation 2: Fine-Tuned YOLOv11 14-Class Detection
- **Concept**: Switched from classical CV to deep learning. Photographed 620 authentic chessboard images under diverse lighting, angles, and occlusions; labeled 14 piece classes on **Roboflow** (32 pieces/board); fine-tuned YOLOv11n on local **RTX 4060** (70/20/10 split).
- **Pain Points**:
  - Localization jumped dramatically, eliminating false circles;
  - **Perspective Compression Confusion**: Wide-angle lens distortion degraded small characters at board edges, causing severe confusion between similar characters (e.g., **Chariot 車 vs. Pawn 卒**, **Horse 馬 vs. Soldier 兵**). With 0% error tolerance, a single misclassification collapsed gameplay.

<div align="center">
  <img src="docs/images/vision_pipeline_evolution.png" width="820px" alt="Roboflow Chessboard Annotation Workspace"/>
  <p><b>Gen 2 Vision Development: Annotating authentic Chinese chess board images on Roboflow (620 images, manually labeled 14 piece classes)</b></p>
</div>

#### Generation 3: Strategic Engineering Downscaling (YOLO Detection + Red/Black Binary Classification)
- **Downscaling Strategy**:
  - Faced with deadlines and physical font recognition bottlenecks, made a decisive engineering trade-off: abandon 14-class character recognition, downscaling YOLO to **piece localization + red/black team binary classification**;
  - YOLO stably outputted piece center coordinates and team classes. Combined with homography perspective rectification, vision passed move coordinate shifts (e.g., `e6 -> e4`) to the decision brain.
- **Deadlock Resolution & Semi-Autonomous Interaction**:
  - **Piece-Capture Deadlock**: The pipeline relied on a 32-piece count for sliding-window stability. When a piece was captured, the count dropped to 31, causing the system to deadlock between "piece captured" and "hand occlusion";
  - **Human-in-the-Loop State Machine**: Implemented an operator **"Move Confirm Button / Space Bar"**. After moving (or capturing) and retracting hands, the human taps Space (`force_detect_move = True`). The system verifies the snapshot with zero hand shadow interference. This achieved 100% operational reliability, but left two compromises: **no endgame support** (only standard full opening) and **semi-autonomous operator reliance**.

#### Generation 4: Full Engineering Exhibition Reboot (Global Grid Tensor Inversion)
- **Upcoming Exhibition Architecture**:
  Rebooting for the Engineering Project Exhibition based on SOTA paradigms ([Chinese-Chess-Recognition](https://github.com/TheOne1006/chinese-chess-recognition)):
  - **MMPose 4-Corner Rectification**: OpenMMLab MMPose keypoint model detects 4 board corners automatically, rectifying the perspective without manual clicking;
  - **Swin-Transformer 2 Spatial Tensor Output**: Bypasses local bounding boxes; Swin-Transformer 2 outputs a $10 \times 9 \times 16$ spatial tensor, simultaneously classifying all 90 grid intersections into 16 categories (empty / other / 14 red-black pieces);
  - **Breakthrough**: **Fully supports arbitrary mid-game and endgame setups**; naturally immune to capture count drops; removes the confirmation button for 100% autonomous gameplay!

<div align="center">
  <img src="docs/images/vision_real_experiments.png" width="820px" alt="Homography Rectification & 10x9 Grid Inference Interface"/>
  <p><b>Gen 4 Vision Experimental Interface: Homography perspective rectification (Left: Step 2 Rectified Board) and 10×9 global grid recognition inference (Right: 0.54s latency, fully autonomous board digitizing)</b></p>
</div>

---

### 4.3 Game Decision & Rule Engine Module (Lead: Zhang Zichen)

#### 1. Rule Engine & Move Generation (`game/board.py`)
Complete 10x9 Chinese chess rule engine enforcing legal path constraints (Chariot/Cannon straight lines, Horse leg blocking, Elephant eye blocking, Advisor/General palace boundary constraints) with history rollback.

#### 2. Deep Learning Evaluation Network (`game/ai.py`)
- PyTorch `ChessNet` consisting of 3 convolutional and 2 linear layers;
- Consumes a $14 \times 10 \times 9$ board spatial tensor, outputting 8100-dimensional action logits;
- Employs legal-move masking; falls back gracefully to random selection from valid candidate moves if `aaa.pth` weights are absent.

#### 3. Game Decision Optimization Roadmap
For upcoming exhibitions, decision modeling will transition from basic convolutional evaluation to **Reinforcement Learning (RL, self-play training)**.

---

## 5. Hardware Electrical Topology & Bus Definitions

The physical hardware electrical topology is organized below:

| Module | Physical Interface | Protocol | Host Pins | Parameters & Electrical Specs |
| :--- | :--- | :--- | :--- | :--- |
| **PCA9685 Servo Driver** | SDA / SCL | Hardware I2C (100kHz Standard) | CH347 D0(SCL) / D1(SDA) | 16-ch 12-bit PWM, 50Hz ( $T = 20\,\text{ms}$ ) |
| **Base Yaw (J1)** | PWM Channel 0 | 50Hz PWM | PCA9685 Output 0 | RDS3235 Metal Servo, Azimuth |
| **Shoulder Pitch (J2)** | PWM Channel 4 | 50Hz PWM | PCA9685 Output 4 | RDS3235 35kg·cm High-Torque Servo, Lift |
| **Elbow Pitch (J3)** | PWM Channel 8 | 50Hz PWM | PCA9685 Output 8 | 20kg·cm Digital Metal Servo, Radius & Elevation |
| **Wrist Pitch (J4)** | PWM Channel 12 | 50Hz PWM | PCA9685 Output 12 | 180° Metal Servo, Keeps Suction Vertical |
| **Electromagnet Relay** | Control Signal | CH347 Native GPIO | CH347 GPIO2 (Pin 10 / DTR0) | 5V DC relay driving electromagnet, mask 0x04 |
| **Overhead Camera** | USB 2.0 / UVC | DirectShow | PC USB 3.0 Port | 1280×720 @ 30 FPS, low-distortion lens |
| **WCH CH347 Bridge** | USB Type-C | 480Mbps High-Speed USB | PC USB Port | Hardware USB to I2C/GPIO bridge |

### PCA9685 12-Bit PWM Mapping Model
Operating at $T = 20\,\text{ms}$ (50Hz), the pulse width converts to register ticks via:

$$
\text{Tick} = \left\lfloor \frac{T_{\text{pulse}}(\text{ms})}{20\,\text{ms}} \times 4096 \right\rfloor
$$

For RDS3235 servos ( $\theta \in [0^\circ, 270^\circ]$ , pulse width $0.5 \sim 2.5\,\text{ms}$ ):

$$
T_{\text{pulse}}(\theta) = 0.5\,\text{ms} + \frac{\theta}{270^\circ} \times 2.0\,\text{ms}
$$

$$
\text{Tick}(\theta) = \left\lfloor \frac{0.5 + \frac{\theta}{270} \times 2.0}{20} \times 4096 \right\rfloor = \left\lfloor 102.4 + \theta \times 1.517 \right\rfloor
$$

---

## 6. Repository Directory Structure

```bash
ChessRobot/
├── localization/                   # Spatial vision geometry & calibration
│   ├── find_chess_new.py           # Production vision pipeline (ChessVisionV15.4: YOLO localization + team classification)
│   ├── find_chess_simple_plus.py   # Asynchronous stream & high-sensitivity prototype
│   ├── find_chess_simple.py        # Early lightweight prototype (ChessVisionV7)
│   └── find_chess.py               # Classical 14-class engine (ChessVisionEngineV6)
├── recognition/                    # Training, datasets & persistence
│   ├── train.py                    # YOLO fine-tuning training script
│   ├── collect_real_data.py        # Multi-angle physical dataset collection script
│   ├── test_gpu.py                 # CUDA / GPU acceleration validation script
│   ├── yolo26n.pt                  # Iterative fine-tuned model weights
│   └── models/                     # Calibration matrices & model weights (best.pt, H_matrix.npy, ROI_corners.npy)
├── game/                           # Chinese Chess rule engine & AI decision
│   ├── board.py                    # 10x9 complete chess rule base (legal move generator / state rollback)
│   ├── ai.py                       # PyTorch CNN move decision engine (ChessNet, 14-ch in / 8100-dim out; falls back to random if aaa.pth missing)
│   ├── constants.py                # Initial board setups, piece weights & coordinate mappings
│   └── play.py                     # Standalone terminal game loop test runner
├── tools/                          # Low-level hardware diagnostic & calibration suite
│   ├── calibrate_servo.py          # 4-DOF joint microsecond reference pulse calibration tool (Channels 0, 4, 8, 12)
│   ├── test_electromagnet.py       # WCH CH347 native GPIO2 (DTR0) electromagnet actuation test script
│   └── calc_reach.py               # Arm maximum reach envelope & theoretical workspace boundary verification script
├── control.py                      # Manipulator motion controller (5-phase collision-free trajectory planner)
├── kinematics.py                   # 4-DOF analytical IK solver (with X-axis scaling & cantilever droop feedforward)
├── hardware.py                     # WCH CH347 high-speed USB bridge (100kHz I2C + GPIO2) + PCA9685 12-bit PWM driver
├── config.py                       # Global physical link lengths (L1-L4), servo channels & board benchmarks
├── main.py                         # Terminal CLI entry point (asynchronous move_queue closed loop)
├── ui_main.py                      # PyQt5 asynchronous multithreaded GUI dashboard (telemetry / virtual board / calibration)
├── yolo11n.pt                      # Fine-tuned lightweight Chinese chess detection weights
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT Official Open Source License
├── README.md                       # Simplified Chinese Specification
├── README_EN.md                    # English Technical Specification
└── README_JA.md                    # Japanese Technical Specification
```

---

## 7. Quickstart & Deployment Guide

### 7.1 System Requirements
- **OS**: Windows 10 / 11 (64-bit)
- **Python**: Python 3.10+ (Anaconda / Miniconda recommended)
- **CUDA**: NVIDIA GPU (GTX 1650 / RTX 4060 or higher)
- **Hardware Drivers**: WCH CH347 driver (`CH347DLLA64.DLL`) bundled in project root

### 7.2 Environment Setup & Installation
```bash
# 1. Clone repository
git clone https://github.com/TJS-Git-Hub/ChessRobot.git
cd ChessRobot

# 2. Create and activate Python 3.10 environment
conda create -n chess_robot python=3.10 -y
conda activate chess_robot

# 3. Install dependencies and CUDA PyTorch
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 7.3 Hardware Self-Test & Tuning
```bash
# 1. Calibrate servo zero-pulse offsets (90° vertical reference posture)
python tools/calibrate_servo.py

# 2. Test WCH CH347 native GPIO2 relay & electromagnet actuation
python tools/test_electromagnet.py

# 3. Verify robotic arm reach envelope & theoretical boundaries
python tools/calc_reach.py
```

### 7.4 Launching the System
- **Launch PyQt5 Operator Dashboard**:
  ```bash
  python ui_main.py
  ```
- **Launch CLI Terminal Mode**:
  ```bash
  python main.py
  ```

---

## 8. Operating Guide (Interactive Calibration & Play)

1. **4-Point Homography Interactive Calibration**:
   - On initial launch, click the four chessboard corners clockwise (1.Top-Left $\to$ 2.Top-Right $\to$ 3.Bottom-Right $\to$ 4.Bottom-Left) in the camera feed;
   - SVD solves matrix $\mathbf{H}$, establishing homography mapping from pixels to physical millimeters.
2. **Setup Initial Board**:
   - Place all 32 handmade magnetic pieces on the beige board at standard opening coordinates;
   - Tap `Start Match` on the GUI to lock the baseline.
3. **Human Move & Confirmation Trigger**:
   - The human player moves Red; upon retracting hands, tap the **Space Bar** (or click **Move Done**);
   - Vision verifies the displacement, pushing coordinates into `move_queue`, prompting AI to compute Black's move.
4. **Autonomous Arm Execution**:
   - The robotic arm departs standby, executing a 5-phase micro-approach trajectory: cruise arrival $\to$ 8mm hovering stabilization $\to$ vertical descent and magnetic suction $\to$ vertical 8mm lift $\to$ cruise transfer $\to$ precision placement and return to standby.

---

## 9. Open Source License & Acknowledgments

This project is open-sourced under the **[MIT License](LICENSE)**.

- **Core Engineering Team**:
  - **Tian Junshuo (田峻硕)**: Team Lead, Mechanical Assembly, Low-Level Drivers & IK Trajectory Planning
  - **Liu Chaoran (刘超然)**: Vision Lead, Independent Dataset Collection & Annotation, 4-Gen Vision Evolution, Motion Co-Tuning & PyQt5 GUI
  - **Zhang Zichen (张子琛)**: Game Decision Lead, Chinese Chess Rule Engine & PyTorch Decision Model
  - **Xing Yilong (邢艺龙)**: System Integration, Electrical Wiring, Magnetic Board & Hardware Fixture Fabrication
- **Project Presentation & Reports**: Technical reports, presentation slides, and oral defenses were prepared and delivered collaboratively by all four team members.
- **Acknowledgments**: Thanks to Ultralytics YOLO, OpenCV, OpenMMLab, and WCH open-source technical communities.

<div align="center">
  <b>ChessRobot — Zhiyi Lingshou</b>, an intelligent Chinese chess robotic system uniting computer vision and multi-axis kinematics.
</div>
