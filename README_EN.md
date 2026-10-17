<div align="center">

# Full-Stack Autonomous 4-DOF Chinese Chess Robotic Arm System with Deep Vision & Kinematics
### End-to-End Cyber-Physical System Integrating YOLOv11 Vision, Homography Transform, Alpha-Beta Search & Analytical IK

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
  An industrial-grade, hard real-time closed-loop <b>Full-Stack Autonomous 4-DOF Chinese Chess Robotic Arm System</b> (Codename "SmartChess · AgileArm").<br/>
  Seamlessly integrating <b>YOLO Deep Convolutional Neural Network Detection</b>, <b>Homography ($3 \times 3$) Spatial Perspective Calibration</b>,<br/>
  <b>Alpha-Beta Pruning Game Decision Tree Engine</b>, and <b>4-DOF Analytical Inverse Kinematics (IK) Control Laws</b>,<br/>
  establishing an end-to-end cyber-physical loop spanning optical perception, intent recognition, heuristic reasoning, and collision-free electromagnetic pick-and-place execution.
</p>

</div>

---

## Table of Contents
- [1. Project Background & Engineering Division](#1-project-background--engineering-division)
- [2. Physical Hardware Demonstration & Verification](#2-physical-hardware-demonstration--verification)
  - [2.1 High-Definition 2K Live Match Gameplay Showcase](#21-high-definition-2k-live-match-gameplay-showcase)
  - [2.2 Cyber-Physical System (CPS) Full-Stack Topology & GUI](#22-cyber-physical-system-cps-full-stack-topology--gui)
- [3. Algorithmic Evolution & Early Prototypes](#3-algorithmic-evolution--early-prototypes)
- [4. Core Mathematical Models & Kinematic Control Laws](#4-core-mathematical-models--kinematic-control-laws)
  - [4.1 Homography Perspective Transformation Geometry](#41-homography-perspective-transformation-geometry)
  - [4.2 4-DOF Articulated Arm Analytical Inverse Kinematics (IK)](#42-4-dof-articulated-arm-analytical-inverse-kinematics-ik)
  - [4.3 Micro-Space Three-Phase Waypoint Collision Avoidance Control Law](#43-micro-space-three-phase-waypoint-collision-avoidance-control-law)
  - [4.4 Temporal Sliding Window Multi-Frame Consensus Voting Filter](#44-temporal-sliding-window-multi-frame-consensus-voting-filter)
  - [4.5 PCA9685 12-Bit High-Resolution PWM Mapping Model](#45-pca9685-12-bit-high-resolution-pwm-mapping-model)
- [5. Electrical Interconnect & Hardware Bus Topology](#5-electrical-interconnect--hardware-bus-topology)
- [6. Repository Directory Structure](#6-repository-directory-structure)
- [7. Quick Start & Deployment Guide](#7-quick-start--deployment-guide)
- [8. Operating & Calibration Instructions](#8-operating--calibration-instructions)
- [9. License & Acknowledgments](#9-license--acknowledgments)

---

## 1. Project Background & Engineering Division

This project represents the capstone engineering design for *Intelligent Control Algorithm Design / Robotic Control Engineering* (Group 07). Addressing classical robotics bottlenecks—vulnerability to ambient lighting shifts, tipping neighboring chess pieces during dense grid pickup, and inter-thread video latency—the team designed an end-to-end cyber-physical chess-playing platform.

### Engineering Responsibilities

| Developer | Role | Core Modules & Technical Deliverables |
| :--- | :--- | :--- |
| **Tian Junshuo (田峻硕)** | **Team Lead / Mechanical & Motion Control Lead** | <ul><li>4-DOF robotic arm physical tuning, assembly, and servo calibration (RDS3235 35kg·cm high-torque metal digital servos)</li><li>WCH CH347 high-speed USB-I2C bridge and PCA9685 12-bit PWM hardware-level driver implementation (`hardware.py`)</li><li>Analytical inverse kinematics (IK) derivation and feedforward gravity droop compensation modeling (`kinematics.py`)</li><li>Micro-space 3-stage collision-free trajectory planning control law (`control.py`)</li></ul> |
| **Liu Chaoran (刘超然)** | **Vision Perception & Systems Integration Lead / Defense Speaker** | <ul><li>OpenCV homography transform ($3 	imes 3$ Homography) 4-point interactive calibration & physical coordinate mapping (`localization/find_chess_new.py`)</li><li>Custom lightweight YOLO object-detection deployment & LAB color-space lightness ($L$-channel) black/red piece classifier</li><li>35-frame temporal sliding-window consensus voting filter & piece-capture deadlock prevention mechanism</li><li>End-to-end multithreaded decoupled asynchronous CPS architecture & PyQt5 interactive dashboard development (`ui_main.py`)</li><li>Project defense presentation design, technical design report compilation, and final oral defense presentation</li></ul> |
| **Zhang Zichen (张子琛)** | **Game Decision & Rule Engine Lead** | <ul><li>Chinese Chess standard 10x9 rule engine, legal move generator, and move-history rollback mechanism (`game/board.py`)</li><li>Deep learning decision evaluation engine based on PyTorch Convolutional Neural Networks (`game/ai.py`)</li><li>Built-in 50+ opening book database and layered greedy decision fallback strategies (`game/constants.py`)</li></ul> |
| **Xing Yilong (邢艺龙)** | **Dataset & Fixture Lead** | <ul><li>Multi-illumination, multi-angle physical chessboard dataset collection and annotation (`recognition/collect_real_data.py`)</li><li>Custom acrylic magnetic chessboard fabrication, piece magnetic sheet mounting, and electromagnet end-effector fixture assembly</li><li>Electrical wiring, physical test bench assembly, and multi-axis hardware integration testing (`tools/`)</li></ul> |

---

## 2. Physical Hardware Demonstration & Verification

Below are authentic frames captured directly from the 2K 60fps demonstration video (`电子科技大学_智能控制算法设计“四轴机械臂”_答辩.mp4`) and corresponding system architectural schematics.

* The complete 2K 60fps full-loop autonomous gameplay demonstration video is available on Bilibili:  
  👉 **[Watch Live Demo on Bilibili: Full-Stack Autonomous 4-DOF Chinese Chess Robotic Arm Closed-Loop Match](https://www.bilibili.com/video/BV1aXLn6HEQd)**  
  *(Demonstrating human move perception, Homography transform, Alpha-Beta heuristic decision, 4-DOF analytical IK trajectory planning, and electromagnetic pick-and-place)*

### 2.1 High-Definition 2K Live Match Gameplay Showcase

<div align="center">

| 1. Human Move Detection & Hand Occlusion Rejection | 2. Analytical IK Descent & Electromagnetic Pickup | 3. Collision-Free Trajectory Transfer & Placement |
| :---: | :---: | :---: |
| <img src="docs/images/demo_live_human_move.jpg" width="310px" alt="Human Move Detection"/> | <img src="docs/images/demo_live_arm_pickup.jpg" width="310px" alt="Robotic Arm Electromagnetic Pickup"/> | <img src="docs/images/demo_live_arm_place.jpg" width="310px" alt="Precision Placement and Standby"/> |
| Global industrial camera tracks board at 30 FPS; 35-frame consensus window eliminates hand shadow noise, reliably isolating piece moves | AI engine calculates optimal move; analytical IK solves 4 joint angles for vertical plunge, electromagnet securing the target piece | Arm lifts vertically to 65mm safety cruising plane, avoids neighboring pieces, places piece with sub-1mm error, and returns to standby |

</div>

### 2.2 Cyber-Physical System (CPS) Full-Stack Topology & GUI

<div align="center">
  <img src="docs/images/system_architecture.png" width="900px" alt="End-to-End Cyber-Physical System Architecture"/>
  <p><b>Figure 1: Full-Stack Multi-Threaded Decoupled Cyber-Physical Architecture</b></p>
</div>

<div align="center">

| Complete Physical Workstation Rig (`demo_system_cover.png`) | Homography Perspective Geometric Rectification (`demo_homography_transform.png`) | YOLO Deep Learning Object Detection (`demo_yolo_detection.png`) |
| :---: | :---: | :---: |
| <img src="docs/images/demo_system_cover.png" width="300px" alt="Workstation Rig"/> | <img src="docs/images/demo_homography_transform.png" width="300px" alt="Homography Calibration"/> | <img src="docs/images/demo_yolo_detection.png" width="300px" alt="YOLO Detection"/> |
| 4-DOF high-strength aluminum arm, overhead industrial camera, and custom acrylic magnetic chessboard | Eliminates camera tilt distortion, mapping 2D image pixels into absolute physical millimeter coordinates | Real-time inference across 14 red/black piece classes on board grid with 12ms single-frame latency |

</div>

<div align="center">
  <img src="docs/images/demo_ui_console.png" width="880px" alt="PyQt5 Telemetry GUI Dashboard"/>
  <p><b>Figure 2: PyQt5 Multi-Threaded Telemetry Dashboard (Live Video, Virtual Board, Telemetry Log & Manual Calibration)</b></p>
</div>

---

## 3. Algorithmic Evolution & Early Prototypes

The platform evolved across multiple iterations, overcoming early algorithmic limitations:

1. **Prototype 1 (`PythonProject1`) — Classical Color & Edge Segmentation**:
   - Employed OpenCV Hough Circles and HSV color thresholding.
   - **Limitations & Discard**: Extremely sensitive to lighting shifts; specular reflections on acrylic caused circle center drifts $> 5\text{ mm}$, causing robotic arm pick failures.
2. **Prototype 2 (`PythonProject2`) — Deep Learning & Calibration Experiments**:
   - Replaced color heuristics with YOLO object detection and established 4-corner perspective rectification.
   - Eliminated piece misclassification, but visual perception remained synchronously coupled with arm execution.
3. **Prototype 3 (`TianProject`) — Hardware Drivers & Kinematics Integration**:
   - Packaged WCH CH347 USB-I2C driver library and solved 4-axis analytical inverse kinematics.
4. **Final Architecture (`ChessRobot`) — Asynchronous CPS Closed Loop**:
   - Refactored into a `move_queue` thread-safe decoupled architecture;
   - Deployed YOLOv11 with a 35-frame temporal consensus anti-occlusion filter;
   - Introduced micro-space 3-phase collision-free trajectory control and full PyQt5 dashboard.

---

## 4. Core Mathematical Models & Kinematic Control Laws

### 4.1 Homography Perspective Transformation Geometry

To correct for overhead camera tilt, a $3 \times 3$ planar homography matrix $\mathbf{H}$ maps image pixel coordinates $(u, v)$ to real-world chessboard millimeter coordinates $(X, Y)$:

$$
\begin{bmatrix} X' \\ Y' \\ Z' \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

Absolute millimeter coordinates are recovered through homogeneous de-normalization:

$$
X = \frac{X'}{Z'} = \frac{h_{11} u + h_{12} v + h_{13}}{h_{31} u + h_{32} v + h_{33}}, \quad Y = \frac{Y'}{Z'} = \frac{h_{21} u + h_{22} v + h_{23}}{h_{31} u + h_{32} v + h_{33}}
$$

Solved via Singular Value Decomposition (SVD) across calibrated corner points, global positioning error is constrained within $\le 1.0\text{ mm}$.

---

### 4.2 4-DOF Articulated Arm Analytical Inverse Kinematics (IK) & Feedforward Compensation

The robotic arm comprises a Base Yaw Joint (Joint 1), Shoulder Pitch Joint (Joint 2), Elbow Pitch Joint (Joint 3), and an Adaptive Wrist Attitude Joint (Joint 4).

```mermaid
graph LR
    Base["Base Yaw J1 (θ₁)"] --> Shoulder["Shoulder Pitch J2 (θ₂)"]
    Shoulder --> Elbow["Elbow Pitch J3 (θ₃)"]
    Elbow --> Wrist["Wrist Attitude J4 (θ₄)"]
    Wrist --> End["Electromagnet (Vertical Orientation φ = -90°)"]
```

#### 1. Calibrated Physical Link Dimensions (`config.py`)
Link dimensions are calibrated from the physical hardware assembly:
- Base center height: $L_1 = 73.0\,\text{mm}$
- Upper arm link length: $L_2 = 231.0\,\text{mm}$
- Forearm link length: $L_3 = 231.0\,\text{mm}$
- Wrist center to electromagnet face: $L_4 = 66.0\,\text{mm}$

#### 2. Non-Linear Cantilever Deflection & Feedforward Control Law (`kinematics.py`)
Due to the long cantilever reach and open-loop digital servos, gravity causes elastic structural droop at extended reach radii. The team engineered empirical feedforward models prior to analytical IK computation:

- **Lateral X-Axis Scaling Feedforward** (compensates for base yaw non-linearity, correcting ~10% inward displacement):
  $$X_{\text{comp}} = X \times K_x \quad (K_x = 1.125)$$
- **Effective Synthesis Radius & Radial Shrinkage Compensation**:
  $$R_{\text{target}} = \sqrt{X_{\text{comp}}^2 + Y^2}$$
  $$R_{\text{comp}} = R_{\text{target}} \times (1 - K_r) \quad (K_r = 0.06)$$
- **Cantilever Sag Elevation Feedforward Model** (scales linearly with reach radius):
  $$\text{comp}_z = Z + (R_{\text{target}} \times K_z) \quad (K_z = 0.12)$$

#### 3. Closed-Form Analytical IK Derivation
Constraining the end-effector vertically downward ($\phi = -90^\circ$), the spatial 3D problem is decoupled into the sagittal plane:

1. **Base Yaw Angle $\theta_1$**:
   $$\theta_1 = \text{atan2}(X_{\text{comp}}, Y)$$
   Mapped to the 90° calibration reference:
   $$t_1^{\text{deg}} = \text{degrees}(\theta_1) + 90.0^\circ$$

2. **Sagittal Effective Vertical Distance & Chord Length**:
   $$\Delta Z = (\text{comp}_z + L_4) - L_1$$
   $$D = \sqrt{R_{\text{comp}}^2 + \Delta Z^2}$$

3. **Law of Cosines for Shoulder Angle $\alpha$ and Elbow Angle $\gamma$**:
   $$\cos(\alpha) = \frac{L_2^2 + D^2 - L_3^2}{2 L_2 D}, \quad \cos(\gamma) = \frac{L_2^2 + L_3^2 - D^2}{2 L_2 L_3}$$
   $$\alpha = \arccos\left(\text{clip}(\cos(\alpha), -1, 1)\right), \quad \gamma = \arccos\left(\text{clip}(\cos(\gamma), -1, 1)\right)$$
   $$\beta = \text{atan2}(\Delta Z, R_{\text{comp}})$$

4. **Joint Angles & Wrist Orientation**:
   $$t_2^{\text{deg}} = \text{degrees}(\beta + \alpha)$$
   $$t_3^{\text{deg}} = t_2^{\text{deg}} - (180.0^\circ - \text{degrees}(\gamma))$$
   $$t_4^{\text{relative}} = -90.0^\circ - t_3^{\text{deg}}$$

Closed-form evaluation completes in $< 1\,\mu\text{s}$ on CPU without numerical Jacobian singularities, maintaining end-effector physical placement strictly within operational tolerance.

---

### 4.3 Micro-Space Three-Phase Waypoint Collision Avoidance Control Law

Given piece radii of $18\,\text{mm}$ and inter-piece gaps frequently below $10\,\text{mm}$, direct linear interpolation causes perimeter piece collisions. The system executes a three-phase waypoint control law:

$$
\mathbf{P}(t) = \begin{cases} 
(X_{\text{src}}, \, Y_{\text{src}}, \, Z_{\text{hover}}), & t \in [0, t_1) \quad (\text{Hover Waypoint above Source}) \\
(X_{\text{src}}, \, Y_{\text{src}}, \, Z_{\text{grip}}), & t \in [t_1, t_2) \quad (\text{Vertical Descent & Magnetic Grip}) \\
(X_{\text{src}}, \, Y_{\text{src}}, \, Z_{\text{hover}}), & t \in [t_2, t_3) \quad (\text{Vertical Lift to Safe Cruise Height}) \\
(X_{\text{dst}}, \, Y_{\text{dst}}, \, Z_{\text{hover}}), & t \in [t_3, t_4) \quad (\text{Horizontal Transfer at Cruise Height}) \\
(X_{\text{dst}}, \, Y_{\text{dst}}, \, Z_{\text{drop}}), & t \in [t_4, t_5) \quad (\text{Vertical Plunge to Target Grid}) \\
(X_{\text{dst}}, \, Y_{\text{dst}}, \, Z_{\text{hover}}), & t \in [t_5, t_6) \quad (\text{Vertical Retraction to Standby})
\end{cases}
$$

Cruising altitude is fixed at $Z_{\text{hover}} = 65\,\text{mm}$ and grip height at $Z_{\text{grip}} = 15\,\text{mm}$, reducing collision rates to **0%**.

---

### 4.4 Temporal Sliding Window Multi-Frame Consensus Voting Filter

To eliminate false triggers caused by player hands or dynamic shadows, a sliding buffer of $W = 35$ frames ($\approx 1.1\,\text{s}$) records classifications across all 90 grid intersections:

$$
S_{\text{consensus}}(i, j) = \arg\max_{c \in \mathcal{C}} \sum_{t=1}^{W} \mathbb{I}(\hat{C}_t(i, j) == c)
$$

A state transition is asserted only if:

$$
\text{Votes}(c) \ge \theta_{\text{votes}} \quad (\theta_{\text{votes}} = 28)
$$

A legal move event is dispatched to `move_queue` if and only if exactly one origin grid clears and one destination grid receives a piece.

---

### 4.5 PCA9685 12-Bit High-Resolution PWM Mapping Model

The PCA9685 driver provides 12-bit resolution (4096 levels) at $f_{\text{PWM}} = 50\,\text{Hz}$ ($T = 20\,\text{ms}$).

Pulse width $T_{\text{pulse}} \in [0.5\,\text{ms}, 2.5\,\text{ms}]$ converts to register ticks:

$$
\text{Tick} = \left\lfloor \frac{T_{\text{pulse}}(\text{ms})}{20\,\text{ms}} \times 4096 \right\rfloor
$$

Linear mapping for RDS3235 servos ($\theta \in [0^\circ, 270^\circ]$):

$$
\text{Tick}(\theta) = \left\lfloor \frac{0.5 + \frac{\theta}{270} \times 2.0}{20} \times 4096 \right\rfloor = \left\lfloor 102.4 + \theta \times 1.517 \right\rfloor
$$

Providing angular resolution of $0.066^\circ$, this guarantees sub-millimeter end-effector precision.

---

## 5. Electrical Interconnect & Hardware Bus Topology

| Hardware Module | Physical Interface | Protocol | Host Pin Destination | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **PCA9685 Servo Board** | SDA / SCL | Hardware I2C (400kHz) | CH347 D0(SCL) / D1(SDA) | 16-ch 12-bit PWM, 50Hz ($T = 20\,\text{ms}$) |
| **Base Yaw (J1)** | PWM Channel 0 | 50Hz PWM Waveform | PCA9685 Output 0 | RDS3235 Metal Digital Servo, Azimuth |
| **Shoulder Pitch (J2)** | PWM Channel 1 | 50Hz PWM Waveform | PCA9685 Output 1 | RDS3235 35kg·cm High-Torque Servo, Lift |
| **Elbow Pitch (J3)** | PWM Channel 2 | 50Hz PWM Waveform | PCA9685 Output 2 | 20kg·cm Digital Servo, Reach & Elevation |
| **Wrist Pitch (J4)** | PWM Channel 3 | 50Hz PWM Waveform | PCA9685 Output 3 | 180° Metal Servo, Keeps Suction Vertical |
| **Electromagnet Tool** | MOS Trigger | GPIO High/Low | PCA9685 Output 4 / Relay | 5V/12V DC Electromagnet, $> 5\,\text{N}$ Suction |
| **Overhead Camera** | USB 2.0 / UVC | DirectShow Protocol | Host USB 3.0 Port | 1280×720 @ 30 FPS, Low-Distortion Lens |
| **WCH CH347 Bridge** | USB Type-C | 480Mbps High-Speed USB | PC USB Port | Hardware USB to I2C/UART/SPI Bridge |

---

## 6. Repository Directory Structure

```bash
ChessRobot/
├── localization/                   # Spatial visual geometry & calibration core
│   ├── find_chess_new.py           # Production vision pipeline (ChessVisionV15.4: YOLO + LAB lightness + 35-frame consensus)
│   ├── find_chess_simple.py        # Lightweight prototype testing engine (ChessVisionV7)
│   └── find_chess.py               # Classical full-classification historical engine (ChessVisionEngineV6)
├── recognition/                    # Model training, dataset collection & calibration persistence
│   ├── train.py                    # YOLO fine-tuning & training script
│   ├── collect_real_data.py        # Physical multi-illumination dataset acquisition utility
│   ├── test_gpu.py                 # CUDA / GPU acceleration & environment verification
│   └── models/                     # Calibration matrices persistence (H_matrix.npy, ROI_corners.npy)
├── game/                           # Chinese Chess rule engine & AI decision
│   ├── board.py                    # 10x9 complete chess rule base (legal move generator / state rollback)
│   ├── ai.py                       # PyTorch Convolutional Neural Network evaluation engine (ChessNet)
│   ├── constants.py                # Initial board setups, piece weights & coordinate mappings
│   └── play.py                     # Standalone terminal game loop test runner
├── tools/                          # Low-level hardware diagnostic & calibration suite
│   ├── calibrate_servo.py          # 4-DOF joint microsecond reference pulse calibration tool
│   ├── test_electromagnet.py       # WCH CH347 GPIO electromagnet actuation test script
│   └── calc_reach.py               # Arm operational workspace & reachability validation utility
├── control.py                      # Manipulator motion controller (3-stage collision-free trajectory planner)
├── kinematics.py                   # 4-DOF analytical IK solver (with X-axis scaling & cantilever droop feedforward)
├── hardware.py                     # WCH CH347 high-speed USB bridge (I2C/GPIO) + PCA9685 12-bit PWM driver
├── config.py                       # Global physical link lengths (L1-L4), servo offsets & board benchmarks
├── main.py                         # Terminal CLI entry point (asynchronous move_queue closed loop)
├── ui_main.py                      # PyQt5 asynchronous multithreaded GUI dashboard (telemetry / virtual board / calibration)
├── yolo11n.pt                      # Fine-tuned lightweight Chinese chess detection weights
├── requirements.txt                # Python dependencies list
├── LICENSE                         # MIT License
├── README.md                       # Simplified Chinese Technical Specification
├── README_EN.md                    # English Technical Specification
└── README_JA.md                    # Japanese Technical Specification
```

---

## 7. Quick Start & Deployment Guide

### Prerequisites
- **Operating System**: Windows 10 / 11 (64-bit) or Ubuntu 20.04 / 22.04 LTS
- **Python**: Python 3.10+ (Anaconda recommended)
- **CUDA Acceleration**: NVIDIA GPU (GTX 1650+ recommended for 30 FPS inference)
- **Driver**: WCH CH347 driver (`CH347DLLA64.DLL`) bundled in repository root

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/TJS-Git-Hub/ChessRobot.git
cd ChessRobot

# Create and activate Python 3.10 environment
conda create -n chess_robot python=3.10 -y
conda activate chess_robot

# Install dependencies and CUDA PyTorch
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Hardware Diagnostics
```bash
# Scan I2C bus connectivity
python tools/ch347_i2c_scanner.py

# Verify servo zero offsets
python tools/servo_tester.py
```

### 3. Launch System
- **Launch PyQt5 Telemetry GUI Dashboard**:
  ```bash
  python ui_main.py
  ```
- **Launch Headless CLI Terminal Mode**:
  ```bash
  python main.py
  ```

---

## 8. Operating & Calibration Instructions

1. **Board Corner Calibration**:
   - Ensure camera overlooks board. In GUI, click `Calibrate Corners`.
   - Click four outer corners clockwise; homography matrix is saved to `config.py`.
2. **Initial Setup**:
   - Place all 32 pieces in standard starting positions.
   - Click `Start Match`; the system initializes video processing.
3. **Human Move**:
   - Human plays Red pieces first. Move piece and retract hand.
   - Consensus filter registers the move within 1.1s, triggering AI planning.
4. **Robot Move**:
   - Arm lifts, hovers over target Black piece, descends vertically with electromagnet activated, transfers piece, places it, and returns to standby.

---

## 9. License & Acknowledgments

This project is licensed under the **[MIT License](LICENSE)**.

- **Core Engineering Team (Group 07)**:
  - **Tian Junshuo (田峻硕)**: Mechanical Assembly, Hardware Bus Communication & Inverse Kinematics
  - **Liu Chaoran (刘超然)**: Vision Detection, Homography Calibration, Multi-Threaded CPS Pipeline & PyQt5 GUI
  - **Zhang Zichen (张子琛)**: Game Decision Heuristics, Chinese Chess Rule Engine & Board Modeling
  - **Xing Yilong (邢艺龙)**: Dataset Engineering, Fixture Fabrication & Hardware Test Bench Construction
- **Acknowledgments**:
  - Mentorship from the *Intelligent Control Algorithm Design* academic staff.
  - Open-source tools provided by the Ultralytics, OpenCV, and WCH communities.

---

<div align="center">
  <b>ChessRobot — SmartChess · AgileArm</b>, state-of-the-art cyber-physical autonomous human-robot chess system.
</div>
