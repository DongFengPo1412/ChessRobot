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
- **Physical Chessboard Selection & Mechanical Evolution**:
  - **Eliminated Initial Version ( $40 \times 40\,\text{cm}$ Dark Green Flexible Mat )**: Initially procured a $40 \times 40\,\text{cm}$ dark green flexible soft chessboard. However, during kinematic integration with the 4-DOF manipulator, mechanical workspace limits (upper arm $L_2 = 231\,\text{mm}$, forearm $L_3 = 231\,\text{mm}$) prevented the electromagnetic end-effector from reaching outer edge intersections on the furthest 9th rank (Red baseline), exceeding the arm's reach limit and forcing its prompt elimination;
  - **Standardized Production Version ( $35 \times 35\,\text{cm}$ Beige Flexible Mat )**: Replaced with a custom $35 \times 35\,\text{cm}$ beige portable soft Chinese chessboard (flexible mat). The grid center-to-center spacing is compressed to approx. $34.2\,\text{mm}$ (strictly matching `SQUARE_WIDTH = 34.2` and `BOARD_ORIGIN_Y = 200.0` in `config.py`), enabling the robotic arm to reliably and smoothly cover all 90 grid intersections across the entire board.
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
| **Liu Chaoran (刘超然)** | **Vision Perception Lead** | <ul><li><b>Independently conducted 100% of physical dataset collection, cleaning, and annotation; engineered an autonomous single-piece micro-crop and empty-board synthesis training pipeline</b>;</li><li><b>Custom-designed and fabricated specialized chessboard fixtures and 32 magnetic pieces</b>;</li><li>Spearheaded all 4 generations of vision architecture, developing a core algorithm matrix encompassing MOG2 background subtraction occlusion detection, CLAHE adaptive equalization, 60px ROI graveyard isolation mask, Chu River deadzone filtering, and energy-bucket majority-voting anti-jitter;</li><li>Developed Gen 4 fully autonomous spatial perception network: implemented MMPose 4-corner automatic perspective rectification and unwarping, trained Swin-Transformer global feature network to directly output a $10 \times 9 \times 16$ spatial state tensor for high-precision 16-class classification across all 90 grid intersections;</li><li>Developed multithreaded PyQt5 interactive operator dashboard (featuring lossless $2\times$ calibration coordinate recovery, `ui_main.py`);</li><li><b>Assisted in robotic motion control parameter tuning</b> (including cantilever gravity droop elevation feedforward compensation).</li></ul> |
| **Zhang Zichen (张子琛)** | **Game Decision & Rule Engine Lead** | <ul><li>Engineered complete 10x9 Chinese chess rule engine, move validation, and history rollback (`game/board.py`);</li><li>Developed PyTorch CNN evaluation network (`ChessNet`, 14-channel input / 8100-dim action space) (`game/ai.py`);</li><li>Standardized chessboard coordinate representations and initial game setup constants (`game/constants.py`).</li></ul> |
| **Xing Yilong (邢艺龙)** | **System Integration & Fixtures Lead** | <ul><li>Managed full-machine electrical wiring, power regulation distribution unit construction, and wiring safety standards;</li><li>Fabricated and mounted the robotic arm end-effector electromagnet mechanical fixture;</li><li>Constructed multi-axis physical test bench, assisting in multi-axis motion integration (`tools/`).</li></ul> |

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

#### Generation 1 Prototype (Gen 1: Hough Circles + Synthetic Font OCR Exploration)
- **Concept**: Utilized OpenCV Hough Circles to localize circular piece contours and center coordinates; automatically generated 1,000+ synthetic bold Microsoft YaHei character crops on white backgrounds via computer scripts, attempting to train a lightweight CNN for two-stage character OCR.
- **Internal Iterations & Limitations**:
  - **Sub-iteration 1 (Simultaneous CNN Color & Character Classification)**: Initially attempted to train a single CNN model to simultaneously output red/black team and Chinese character classes. Under complex ambient illumination and optical distortion, red vs. black color chromatic features had minimal distinction, preventing stable model convergence;
  - **Sub-iteration 2 (Decoupled HSV Color Space + CNN Character OCR)**: Decoupled the tasks by delegating red/black team classification to the HSV color space (which performed reliably in real tests), while CNN focused strictly on character recognition.
- **Root Physical Causes of Abandonment**:
  - **Complete OCR Generalization Failure**: The training set relied 100% on ideal computer-generated Microsoft YaHei glyphs with **zero real photographic samples**, resulting in extremely fragile real-world generalization;
  - **Severe Small-Scale Distortion**: To cover the entire board, the camera was mounted at a relatively high altitude. Cropped Hough-circle piece patches suffered from extremely low resolution where character stroke details were blurred and unidentifiable;
  - **Hyper-Sensitivity of Geometric Tuning**: Hough circle detection was hypersensitive to ambient lighting and glares. Lowering sensitivity caused severe false negatives under shadows, while raising sensitivity produced swarming false-positive circles from reflections. During pure visual bench validation—prior to any robotic arm integration—the approach was abandoned due to unacceptable error rates.

#### Generation 2 Evolution (Gen 2: Fine-Tuned YOLOv11 14-Class Detection)
- **Concept & Overhead Dataset Acquisition**:
  Abandoned classical geometric operators in favor of deep learning. An overhead industrial camera was rigidly mounted to a dedicated metallic bracket on the system base frame at a working altitude of approx. $40\,\text{cm}$ (`collect_real_data.py`), photographing 620 authentic chessboard images under diverse lighting, angles, empty boards, mid-game positions, and partial occlusions; manually annotated 14 piece classes on **Roboflow** (32 pieces/board); fine-tuned a YOLOv11n model on a local **RTX 4060** GPU (70% train, 20% val, 10% test).
- **Extreme Data Augmentation Pipeline**:
  To combat harsh specular reflections and boost model robustness, a comprehensive data augmentation pipeline was deployed in `train.py`:
  - **Geometric & Affine Transformations**: $\pm 45^\circ$ board rotation (`degrees=45.0`), $\pm 15\%$ spatial translation, $\pm 60\%$ multi-scale scaling, $\pm 15^\circ$ shear, and random horizontal/vertical flips (`fliplr/flipud=0.5`);
  - **Occlusion & Composition Augmentations**: Mosaic tiling (1.0), MixUp (0.2), random erasing to simulate hand shadows and reflections (`erasing=0.4`), and Copy-Paste (0.2);
  - **Photometric Variations**: Random HSV saturation (0.5) and value (0.4) perturbations to adapt to non-uniform ambient illumination.
- **Practical Performance & Cross-Board Domain Shift**:
  - **Superb Target Localization**: Piece detection recall and center localization accuracy jumped dramatically, completely eliminating the false-positive circle artifacts of Hough detection;
  - **Severe Cross-Board Domain Shift**: Of the 620 annotated images, **approx. 500 were photographed on the early eliminated dark green board**. Following the switch to the beige soft mat, due to tight delivery deadlines, only 120 additional images could be collected. The radical shift in background color (dark green vs. light beige) introduced severe domain shift, significantly limiting generalization on the finalized board;
  - **Physical Root Cause of High Character Error**: **Planar orthographic projection unwarping was not yet introduced (lack of homography perspective rectification)**. Distant piece strokes from the oblique overhead angle suffered from perspective compression, causing severe confusion between visually similar characters (e.g., **Chariot 車 vs. Pawn 卒**, **Horse 馬 vs. Soldier 兵**), with only a few salient characters recognized accurately. With 0% error tolerance in gameplay, this model could not support autonomous game decisions.

<div align="center">
  <img src="docs/images/vision_pipeline_evolution.png" width="820px" alt="Roboflow Chessboard Annotation Workspace"/>
  <p><b>Gen 2 Vision Development: Annotating authentic Chinese chess board images on Roboflow (620 images, manually labeled 14 piece classes)</b></p>
</div>

#### Generation 3 Production (Gen 3: Strategic Engineering Downscaling — YOLO Localization + Red/Black Binary Classification)
- **Engineering Reduction & Positive Results**:
  - Faced with tight deadlines and the physical bottleneck of unwarped character classification, made a decisive engineering trade-off: abandoned complex 14-class character recognition, downscaling YOLO to **piece localization + red/black team binary classification**;
  - The dataset required only two target classes ("Red" and "Black"), achieving rapid training convergence. **Target localization and red/black classification achieved near-zero error and exceptional stability during live matches**;
  - Combined with homography perspective transformation (mapping center coordinates to the physical 10×9 grid), vision passed only the coordinate difference before and after a move (e.g., `e6 -> e4`) to the decision brain.
- **Core Vision Algorithm Matrix (Nine Foundational Engineering Mechanisms)**:
  - **1. Clockwise 4-Point Homography & 60px ROI Masking (Graveyard Dead-Piece Isolation)**:
    Clockwise calibration clicks (Top-Left $\to$ Top-Right $\to$ Bottom-Right $\to$ Bottom-Left) compute the homography matrix $\mathbf{H}$ via SVD. An expanded convex polygon (`roi_buffer_px = 60`, approx. 2-3cm) generates a binary mask via `cv2.fillPoly` and `cv2.bitwise_and`, zeroing out everything outside the board. When the robot arm captures pieces and deposits them in the external right-hand Graveyard (`GRAVEYARD_XYZ`), **all captured dead pieces outside the board are blocked by the ROI mask, completely preventing false positive ghost detections!**
  - **2. MOG2 Gaussian Mixture Background Subtraction — Physical-Layer Dynamic Hand Occlusion Detection**:
    Integrated `cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=120, detectShadows=True)` in `find_chess.py`. Downsamples frames to $640 \times 360$ for high-speed processing, computing moving foreground masks intersected with the board ROI. Using `cv2.findContours`, if the maximum foreground contour area exceeds 2,000 pixels, the system detects human hand penetration, immediately setting `is_occluded = True` to halt board state updates;
  - **3. Asynchronous Single-Buffer Zero-Latency Video Stream Engine (`VideoStream`)**:
    Encapsulates OpenCV video capture in a dedicated daemon thread, forcing the driver buffer size to 1 (`CAP_PROP_BUFFERSIZE = 1`). This eliminates multi-threaded frame queue buildup, guaranteeing real-time 0-delay 720p frames for inference;
  - **4. LAB Color Space CLAHE Adaptive Histogram Equalization Preprocessing**:
    Transforms frames to LAB color space, applying Contrast Limited Adaptive Histogram Equalization (`cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))`) to the L channel before returning to BGR, boosting stroke contrast and overcoming ambient shadows;
  - **5. L-Channel Luminance-Ratio High-Purity Black Team Classification (`is_black_piece`)**:
    To eliminate susceptibility to ambient color temperature shifts, the algorithm crops piece interiors and analyzes the LAB L channel. If pixels with $L < 45$ (characteristic dark ink) exceed 50% of the piece area, it is conclusively classified as Black, achieving complete immunity to chromatic ambient light variations;
  - **6. Chu River Physical Deadzone Filtering (Ghost Detection Filter)**:
    Establishes a physical deadzone across the open Chu River boundary ( $4.35 < y < 4.65$ between ranks 4 and 5), immediately rejecting false-positive boxes caused by decorative calligraphy printed on the board;
  - **7. Dual-Threshold Confidence Gate & EMA Centroid Smoothing Anti-Jitter Filtering**:
    Employs a detection entry threshold of `detect_conf = 0.50` and tracking survival threshold of `keep_conf = 0.20`, coupled with Exponential Moving Average ( $\alpha = 0.28$ ) and a 1.5-pixel deadzone (`dist_sq > 2.25`) to eliminate pixel jitter and protect joint servos;
  - **8. Piece Energy Tracking & 12-Frame Historical Majority Voting**:
    Maintains an energy state machine for each track ID (+40 on detection, -15 on single-frame loss, activation threshold 50) paired with a 12-frame double-ended queue (`deque(maxlen=12)`). Statistical majority voting (`Counter.most_common(1)`) resolves physical grid coordinates, eliminating spurious single-frame coordinate jumps;
  - **9. Active Visual Suppression & Temporal Decoupling During Arm Manipulation (`suppress_detection`)**:
    While the robot arm moves across the board to pick or place pieces, the main controller sets `suppress_detection = True` to suspend vision reporting, preventing the metallic arm and its shadow from being misinterpreted as chess moves. Upon returning to the standby position, `reset_baseline()` refreshes the board state, achieving seamless temporal decoupling between perception and actuation.
- **Unidentified Characters & Piece-Capture Deadlock Resolution**:
  - **Deadlock Caused by Unrecognized Characters**: Because the downscaled model did not classify Chinese character roles and only tracked occupancy coordinates, the system had **no semantic knowledge of which specific piece disappeared**. When a capture reduced the piece count from 32 to 31, the algorithm could not determine whether a piece was legitimately captured or temporarily occluded by hand shadows, breaking the sliding-window conservation baseline;
  - **Human-in-the-Loop State Machine**: Implemented an operator UI button / Spacebar confirmation mechanism. After completing a move and retracting hands, the human taps Space (`force_detect_move = True`). The system triggers difference extraction on a clean, stationary snapshot without hand occlusion. This achieved 100% operational reliability, with two compromises: **no endgame support** (required standard 32-piece opening) and **reliance on manual operator confirmation (semi-autonomous)**.

#### Generation 4 Upgrade (Gen 4: Autonomous Research Breakthrough — Global Grid Tensor Classification & Synthesis Pipeline)
- **Autonomous Research & Complete Self-Trained Synthesis Pipeline**:
  To overcome endgame limitations and eliminate operator key confirmation, following comprehensive analysis of modern computer vision literature and spatial pose estimation paradigms, the vision system was entirely reconstructed with an independently developed Gen 4 spatial perception network, creating an end-to-end self-trained pipeline from physical data acquisition to tensor inference:
  - **1. Physical Single-Piece Micro-Crop & Empty-Board Sampling**:
    Photographed each of the 32 handmade magnetic pieces individually at close range across all 14 piece classes, performing precise background alpha matting to create a high-fidelity single-piece library; simultaneously captured a clean, standardized orthographic background image of the empty $35 \times 35\,\text{cm}$ beige soft chessboard;
  - **2. Automated Board Image Synthesis Engine (`CChessMixSinglePngCls`)**:
    Engineered an automated data rendering and labeling pipeline. The engine applies random affine transformations, small-angle rotations (simulating human placement irregularities), and HSV illumination variations to single-piece assets, dynamically compositing them onto the 90 intersections (A0 to J8) according to valid chess configurations or randomized layouts. During rendering, **the system automatically generates matching $10 \times 9 \times 16$ Ground Truth spatial state tensors**, synthesizing tens of thousands of diverse mid-game and endgame training images at zero manual annotation cost;
  - **3. Two-Stage Joint Deep Learning Network & Strict Metric Training**:
    - **Stage 1 (Keypoint Pose Rectification Network MMPose / RTMPose)**: Predicts the 4 physical board corner keypoints (A0, A8, J0, J8) from oblique camera angles, automatically rectifying perspective distortion into a standardized orthographic top-down view;
    - **Stage 2 (Swin-Transformer V2 + CChessTableHead)**: Consumes the rectified orthographic board, using a Swin-Transformer V2 backbone with a custom `Chess10_9Neck` and `CChessTableHead` multi-region classification head to output a $10 \times 9 \times 16$ spatial matrix tensor, performing 16-class classification (Red 7 classes, Black 7 classes, empty `.`, other `x`) across all 90 grid intersections simultaneously;
    - **Loss Functions & Rigorous Metrics**: Trained with `KLDiscretLoss` and `SmoothL1Loss`, evaluated under the strict zero-error benchmark across all 90 intersections (Full-err-zero);
  - **Ultimate Breakthrough & Closed-Loop Operation**: Achieved exceptional accuracy with ultra-low error rate; **fully supports arbitrary mid-game and endgame setups**; immediately detects cell state transitions and is **naturally immune to piece-count capture deadlocks**, eliminating manual confirmation keys and achieving a 100% fully autonomous closed-loop perception!

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

1. **Four-Point Homography Interactive Calibration & ROI Locking (with UI $2\times$ Lossless Recovery)**:
   - Upon system launch, click the 4 physical board boundary corners clockwise in the PyQt5 dashboard $640 \times 360$ video display (1.Top-Left $\to$ 2.Top-Right $\to$ 3.Bottom-Right $\to$ 4.Bottom-Left);
   - The UI automatically doubles click coordinates to restore the native $1280 \times 720$ resolution without scaling drift, computing homography matrix $\mathbf{H}$ via SVD and generating an expanded 60px ROI mask to isolate captured pieces in the external graveyard from vision inference.
2. **Piece Setup & Game Initialization**:
   - Place all 32 handmade magnetic pieces on the beige flexible mat in their standard opening positions;
   - Click `Start Game` on the dashboard to lock the initial baseline board.
3. **Human Move & Confirmation Trigger**:
   - Human player moves Red first; after completing the move and fully retracting hands, tap the **Keyboard Spacebar** (or click **Confirm Move** on the UI);
   - Vision extracts displacement coordinates in a clean, hand-free frame and pushes them to `move_queue`, prompting the AI engine to compute Black's optimal response.
4. **Autonomous Manipulator Execution & Temporal Decoupling**:
   - While the arm executes, the system sets `suppress_detection = True` to prevent visual false positives; the arm departs standby to execute a 5-phase collision-free trajectory: cruise approach $\to$ 8mm micro-leveling $\to$ vertical pickup $\to$ vertical lift $\to$ cruise transfer $\to$ vertical placement at destination (transferring captured enemy piece to external graveyard if capturing) $\to$ smooth return to standby, releasing visual suppression to resume perception.

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
