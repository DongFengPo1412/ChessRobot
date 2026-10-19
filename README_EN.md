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
  - **Eliminated Initial Version ( 40 × 40 cm Dark Green Flexible Mat )**: Initially procured a 40 × 40 cm dark green flexible soft chessboard. However, during kinematic integration with the 4-DOF manipulator, mechanical workspace limits (upper arm $L_2 = 231$ mm, forearm $L_3 = 231$ mm) prevented the electromagnetic end-effector from reaching outer edge intersections on the furthest 9th rank (Red baseline), exceeding the arm's reach limit and forcing its prompt elimination;
  - **Standardized Production Version ( 35 × 35 cm Beige Flexible Mat )**: Replaced with a custom 35 × 35 cm beige portable soft Chinese chessboard (flexible mat). The grid center-to-center spacing is compressed to approx. 34.2 mm (strictly matching `SQUARE_WIDTH = 34.2` and `BOARD_ORIGIN_Y = 200.0` in `config.py`), enabling the robotic arm to reliably and smoothly cover all 90 grid intersections across the entire board.
- **Handmade Magnetic Pieces**:
  - Printed high-contrast red and black Chinese chess character templates (Microsoft YaHei font, bold) sized at exactly 3.0 cm diameter;
  - Manually precision-cut all 32 character circles;
  - Procured 3.0 cm diameter circular magnetic iron discs, manually adhering the printed labels onto the discs to create custom magnetic pieces.
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

The robotic system is architected into three decoupled technical modules: **Mechanical & Motion Control**, **Vision Perception**, and **Game Decision & Rule Engine**. This section presents a formal breakdown of the mathematical modeling, algorithmic implementations, and iterative engineering evolution across each module.

---

### 4.1 Mechanical & Motion Control Module (Lead: Tian Junshuo, Co-tuning: Liu Chaoran)

The manipulator features a 4-DOF articulated serial configuration: Base Yaw (J1), Shoulder Pitch (J2), Elbow Pitch (J3), and Wrist Pitch (J4), terminating in a miniaturized electromagnetic suction toolhead constrained strictly perpendicular to the board ( $\phi = -90^\circ$ ).

#### 1. Physical Linkage Parameters (`config.py`)
- Base rotation axis height: $L_1 = 73.0$ mm
- Upper-arm effective length: $L_2 = 231.0$ mm
- Forearm effective length: $L_3 = 231.0$ mm
- Wrist joint to suction center: $L_4 = 66.0$ mm

#### 2. Motion Control Algorithmic Catalog

- **Algorithm 1: Analytical Closed-Form Inverse Kinematics (IK) Formulation**:
  - **Core Principle**: To prevent numerical divergence, singularities, and millisecond-level iterative latencies inherent in numerical Jacobian methods, the spatial 3D Cartesian coordinates are decoupled into the **horizontal yaw plane** and the **vertical pitch plane**. Subject to the vertical wrist constraint ( $\phi = -90^\circ$ ), the Law of Cosines directly yields an analytical closed-form solution with < 0.1 ms computational overhead;
  - **Mathematical Modeling**:
    - **Base Yaw Angle**:
      $$\theta_1 = \text{atan2}(X_{\text{comp}}, Y), \quad t_1^{\text{deg}} = \text{degrees}(\theta_1) + 90.0^\circ$$
    - **Equivalent Pitch Chord Length**:
      $$\Delta Z = (\text{comp}_z + L_4) - L_1, \quad D = \sqrt{R_{\text{comp}}^2 + \Delta Z^2}$$
    - **Law of Cosines for Shoulder Angle $\alpha$ and Elbow Angle $\gamma$**:
      $$\cos(\alpha) = \frac{L_2^2 + D^2 - L_3^2}{2 L_2 D}, \quad \cos(\gamma) = \frac{L_2^2 + L_3^2 - D^2}{2 L_2 L_3}$$
    - **Joint Absolute & Relative Angular Mapping**:
      $$\beta = \text{atan2}(\Delta Z, R_{\text{comp}})$$
      $$t_2^{\text{deg}} = \text{degrees}(\beta + \alpha), \quad t_3^{\text{deg}} = t_2^{\text{deg}} - (180.0^\circ - \text{degrees}(\gamma)), \quad t_4^{\text{relative}} = -90.0^\circ - t_3^{\text{deg}}$$

- **Algorithm 2: Multi-Source Nonlinear Feedforward Compensation Law (`kinematics.py`)**:
  - **Core Principle**: To counter open-loop digital servo gear backlash, structural play, and cantilever elastic deflection under gravity and payload, a 3D decoupled feedforward compensation law is injected prior to inverse kinematics;
  - **Governing Equations**:
    - **Lateral $X$-Axis Geometric Feedforward Scaling**: Expands lateral yaw displacement to overcome mechanical travel attenuation at outermost board columns,
      $$X_{\text{comp}} = X \times K_x \quad (K_x = 1.125)$$
    - **Radial Moment Arm Contraction Correction**:
      $$R_{\text{target}} = \sqrt{X_{\text{comp}}^2 + Y^2}, \quad R_{\text{comp}} = R_{\text{target}} \times (1 - K_r) \quad (K_r = 0.06)$$
    - **Cantilever Gravitational Elastic Sag Compensation**: Compensates for bending sag increasing with moment arm length by elevating the target $Z$,
      $$\text{comp}_z = Z + (R_{\text{target}} \times K_z) \quad (K_z = 0.12)$$
    - **Base Servo Travel Gain Scaling (`base_gain`)**:
      $$u_0 = u_{0,\text{ref}} - (t_1 - 90.0^\circ) \times K_{\text{deg}} \times K_{\text{base}} \quad (K_{\text{base}} = 1.0)$$

- **Algorithm 3: 5-Phase Micro-Approach Collision-Free Trajectory Planning (`control.py`)**:
  - **Core Principle**: Given the tight 35 mm grid spacing versus ≈ 30 mm piece diameters (leaving < 5 mm clearance), diagonal suction descent inevitably knocks over adjacent pieces. Trajectories are discretized into 5 sequential 3D waypoints:
    1. **Cruise Transit**: Rapid horizontal transit at safe cruise elevation $Z_{\text{safe}} = 70.0$ mm;
    2. **Micro-Approach Hover**: Vertical descent to hover 8mm above piece surface ( $Z_{\text{micro}} = -25.0$ mm ), holding 0.3s for wrist plumb stabilization;
    3. **Slow Vertical Plunge & Magnetization**: Slow vertical insertion to board level $Z_{\text{board}} = -33.0$ mm, energizing the electromagnet with a 0.5s dwell for magnetic flux coupling;
    4. **Elbow-Driven Pure Vertical Lift**: Slow, strictly vertical 8mm lift back to $Z_{\text{micro}}$, eliminating diagonal shear that trips neighboring pieces;
    5. **Cruise Transfer & Symmetrical Release**: Vertical ascent to $Z_{\text{safe}}$, transit to target grid point, followed by identical vertical plunge, demagnetization, and vertical retraction;
  - **Capture Sequencing**: When capturing an opponent piece, the planner first commands acquisition and disposal of the victim piece into the external Graveyard ((220, 120, BOARD_Z + 20)) before executing the attacker's trajectory.

- **Algorithm 4: 50Hz Trajectory Interpolation with 0.1mm Float Deadzone Control (`move_to_smooth`)**:
  - **Core Principle**: The interpolator evaluates intermediate steps at 50Hz (`steps = int(duration * 50)`). During pure vertical travel, floating-point roundoff in inverse trigonometry produces spurious $10^{-4}$ mm lateral deltas, causing high-frequency chatter in the base servo;
  - **Deadzone Formulation**:

    $$
    \Delta X_{\text{step}} = \frac{X_{\text{target}} - X_{\text{cur}}}{\text{steps}} \quad (\text{if } |X_{\text{target}} - X_{\text{cur}}| > 0.1\text{ mm, else } 0.0)
    $$

    Clamping sub-0.1mm increments strictly to zero cuts off horizontal servo pulses, ensuring pristine vertical descent.

- **Algorithm 5: 12-Bit PCA9685 PWM Quantization & Hardware Driver Mapping (`hardware.py`)**:
  - **Core Principle**: Commanded via CH347 High-Speed USB bridge over 100KHz I2C bus. The PCA9685 prescaler is set to 121 for a 50 Hz carrier ( $T = 20$ ms );
  - **Pulse-to-Tick Mapping**:
    $$\text{tick} = \left\lfloor \frac{\text{clamp}(u, u_{\min}, u_{\max})}{20000} \times 4096 \right\rfloor$$
    Maps microsecond pulse widths into 12-bit registers written in multi-byte blocks, while native CH347 GPIO2 drives the relay circuit with nanosecond switching response.

#### 3. Motion Control Optimization Roadmap
Future iterations will introduce continuous multi-axis S-curve jerk-limited acceleration profiles to dampen structural vibration during rapid long-distance transits.

---

### 4.2 Vision Perception Module & 4-Gen Evolution (Lead: Liu Chaoran)

The vision perception pipeline was independently developed by Liu Chaoran across all data collection, annotation, and model iteration phases. The system evolved through four distinct architectural generations:

#### Generation 1 Prototype (Gen 1: Hough Circles + Synthetic Font OCR Exploration)
- **Concept & Internal Trials**:
  Explored classical geometric feature extraction for circular piece detection. Two internal variants were tested:
  - **Trial 1 (Joint CNN Color & Font Classifier)**: A single CNN predicted piece color and Chinese character simultaneously. Severe lens distortion and reflection caused gradient conflict and non-convergence;
  - **Trial 2 (Decoupled HSV Color Gating + Independent Character CNN)**: Separated team classification into HSV color space thresholding, using the CNN solely for character recognition.
- **Gen 1 Algorithmic Catalog**:
  - **1. Hough Gradient Circles Transform (`cv2.HoughCircles`)**:
    Leveraged Sobel gradient vectors and accumulator space voting to detect circle boundaries, centers $(x_c, y_c)$, and radius $r$;
  - **2. HSV Color Space Bilateral Thresholding**:
    Decoupled luminance (V) from chrominance (H), applying bilateral hue threshold masks for red dual-peaks and black low-saturation clusters;
  - **3. Automated Font Template Synthesis & Cascaded 2-Stage CNN OCR**:
    Synthesized over 1,000 synthetic white-background red/black character images using Microsoft YaHei Bold, cascading circular ROI crops into the downstream CNN.
- **Objective Root Causes of Retirement**:
  - **Synthetic-to-Real Domain Gap**: Training on synthetic font tiles collapsed in the real physical world (< 20% real-world accuracy);
  - **High-Altitude Scale Distortion**: The overhead camera rendered individual piece characters at very low pixel dimensions, severely blurring calligraphic strokes;
  - **Extreme Lighting Sensitivity**: Hough accumulators generated false positives on woodgrain specular highlights while missing shadow-obscured pieces.

#### Generation 2 Evolution (Gen 2: YOLOv11 Fine-Grained 14-Class Detection & Data Science)
- **Concept & Overhead Dataset Engineering**:
  Abandoned classical heuristics in favor of deep object detection architectures. Designed a complete physical acquisition protocol:
- **Gen 2 Algorithmic Catalog**:
  - **1. 40cm Altitude Overhead Image Acquisition Protocol (`collect_real_data.py`)**:
    Rigidly mounted the camera 40 cm above the board, collecting 620 high-altitude images (1280×720) across empty layouts, opening phases, mid-game, and hand-occluded states. Annotated 14 piece classes on **Roboflow** (32 pieces per image);
  - **2. Extreme Hybrid Augmentation Pipeline (`train.py`)**:
    To maximize robustness under harsh ambient lab lighting, fine-tuned YOLOv11n on local **RTX 4060** using extreme data augmentations:
    - **Spatial Affine Transforms**: Arbitrary rotation $\pm 45^\circ$ (`degrees=45.0`), translation ±15%, scale ±60%, shear $\pm 15^\circ$, and random horizontal/vertical flips (`fliplr/flipud=0.5`);
    - **Mix & Occlusion Simulation**: Full Mosaic 1.0, MixUp 0.2, Random Erasing 0.4 (simulating manipulator arm shadows and player hands), and Copy-Paste 0.2;
    - **Photometric Perturbations**: HSV saturation (±50%) and value (±40%) perturbations to strip specular reflections;
  - **3. Class Loss Re-weighting & Label Smoothing Regularization**:
    Applied `cls=2.5` to prioritize classification gradients over stroke-similar characters, with `label_smoothing=0.1` to mitigate overconfident over-fitting.
- **Physical Bottlenecks & Domain Shift**:
  - **High Localization Quality**: Solved the Hough false-circle problem completely, achieving crisp bounding box recall;
  - **Board Fabric Domain Shift**: Initial 500 images were captured on an early dark green vinyl mat, while final delivery switched to a light beige soft mat (supplemented with only 120 images), inducing significant background domain shift;
  - **Perspective Compression Causing Stroke Confusion**: **The system had not yet introduced homography rectification**. Far-field pieces suffered severe oblique perspective compression. Stroke-similar characters (e.g., **"車" vs. "卒"**, **"馬" vs. "兵"**) experienced high confusion rates, rendering fine-grained 14-class recognition impractical for zero-tolerance chess rules.

<div align="center">
  <img src="docs/images/vision_pipeline_evolution.png" width="820px" alt="Roboflow Chinese Chess Dataset Workspace"/>
  <p><b>Generation 2 Vision Data Science: Annotating full 14-class Chinese chess pieces on Roboflow (620 real-world images, 32 bounding boxes per board)</b></p>
</div>

#### Generation 3 Production (Gen 3: Strategic Dimensionality Reduction —— YOLO Localization + Red/Black Binary + Heuristic Matrix)
- **Strategic Trade-off & Engineering Subtraction**:
  Facing delivery deadlines and unrectified oblique stroke ambiguity, the YOLO task was strategically reduced to **single-class piece localization (`nc: 1, names: ['piece']`) combined with color and topological tracking**; centers were mapped via planar homography to the 10×9 discrete grid, reporting displacement differentials (`e6 -> e4`) to the decision engine.
- **Proprietary Engineering Algorithm Matrix (16 Core Mechanisms)**:
  To eliminate noise from table vibration, ambient reflection, human hand occlusion, and capture state invalidation, Generation 3 deployed a robust matrix of 16 algorithms:
  - **1. Clockwise 4-Point SVD Homography Matrix Estimation ( $\mathbf{H}$ )**:
    Clockwise corner selection (TL $\to$ TR $\to$ BR $\to$ BL) computes the homography matrix $\mathbf{H}$ via Singular Value Decomposition (SVD), mapping oblique camera pixels directly to the normalized $10 \times 9$ discrete board matrix;
  - **2. Radial Vector Outward Extrusion ROI Polygon Mask (`expanded_roi`)**:
    Computes radial unit vectors from the board centroid $\mathbf{C}$: $\mathbf{u}_i = \frac{\mathbf{P}_i - \mathbf{C}}{\|\mathbf{P}_i - \mathbf{C}\|}$, extruding corners outward by a 60px safety margin. Binary polygon masking (`cv2.fillPoly` + `cv2.bitwise_and`) zeroes out everything outside the board, **completely blocking ghost detections from captured pieces in the right-side Graveyard**;
  - **3. MOG2 Dynamic Hand Occlusion Detector (`find_chess.py`)**:
    Maintains a 500-frame background model with variance threshold 120, operating on downsampled $640 \times 360$ frames. Intersecting foreground motion contours with the board ROI triggers `is_occluded = True` whenever contour area exceeds $2000 px²$, locking board updates while hands are in view;
  - **4. Asynchronous Single-Buffer Zero-Latency VideoStream Engine & Hardware Failsafe**:
    A dedicated daemon thread pulls camera frames with driver buffer size clamped to 1 (`CAP_PROP_BUFFERSIZE = 1`), eliminating inter-thread frame queue lag. An internal `if vs.stopped: break` sentinel instantly terminates child threads upon emergency stop, preventing zombie camera processes;
  - **5. LAB Color Space CLAHE Adaptive Histogram Equalization**:
    Transforms frames to LAB color space, applying Contrast Limited Adaptive Histogram Equalization (`clipLimit=3.5, tileGridSize=(8, 8)`) to the L-channel, enhancing dark strokes and evening out asymmetric desk lamp illumination;
  - **6. 15px Core Margin Erosion & L-Channel Luminance Ratio Black Piece Classifier (`is_black_piece`)**:
    Erodes detection bounding boxes inward by 15px (`y1+15:y2-15, x1+15:x2-15`) to isolate wooden rims and board gridlines. If deep black pixels ( $L < 45$ ) exceed 50% of the eroded core, the piece is classified as Black, fully immune to ambient colored illumination;
  - **7. Orientation Polarity Auto-Alignment (`set_black_view`)**:
    Calculates the mean Y coordinate of Black pieces at game setup: if Black is in the upper visual half (far side), Pt1 maps to A0; if Black is in the lower half (near side), Pt3 maps to A0, automatically handling 180° board rotation without manual configuration;
  - **8. Chu River Physical Deadzone Filtering**:
    Enforces a spatial deadzone across the river divide ( $4.35 < y < 4.65$ ), discarding spurious bounding boxes triggered by decorative calligraphy printed in the center river;
  - **9. Dual-Threshold Confidence Gate with Hysteresis**:
    Sets entrance threshold `detect_conf = 0.50` and retention threshold `keep_conf = 0.20`, creating a Schmitt-trigger hysteresis loop that rejects transient false positives while maintaining persistent track of pieces under partial shadow;
  - **10. Centroid EMA Smoothing with 1.5px Deadzone**:
    Updates tracked centroids via Exponential Moving Average ( $\alpha = 0.28$ ) with a 1.5px Euclidean deadzone ( $\text{dist}^2 > 2.25\text{ px}^2$ ), eliminating sensor thermal subpixel noise and arm vibration;
  - **11. Teleportation Fast-Move Bypass (`find2.txt`)**:
    When squared centroid displacement exceeds $\Delta d^2 > 2500\text{ px}^2$, the algorithm flags rapid human manipulation, bypassing EMA filtering to instantly register new physical coordinates without trajectory lag;
  - **12. Global Motion Gating**:
    When any piece displacement satisfies $\Delta d^2 > 25\text{ px}^2$, sets `global_motion = True` and flushes the sliding stabilization window, ensuring no partial moves are committed;
  - **13. Piece Energy Tracking & 12-Frame Historical Majority Voting**:
    Maintains an energy accumulator for each track ID (+40 on detection, -15 on miss, activation at 50). A 12-frame double-ended queue (`deque(maxlen=12)`) performs majority voting (`Counter.most_common(1)`) to lock discrete grid positions against single-frame flickers;
  - **14. Active Visual Suppression During Robotic Manipulation (`suppress_detection`)**:
    While the robot arm is executing trajectory waypoints, the orchestrator sets `suppress_detection = True`, masking camera feed to prevent the arm link and its shadow from being parsed as human moves. Calls `reset_baseline()` once parked;
  - **15. Dual Occupancy & Track-ID Displacement Differential (`main.py`)**:
    Resolves piece capture where an attacking piece replaces a victim piece on the same square by evaluating `pos not in current_board` alongside `last_stable_board[pos] != tid`, gracefully separating source and destination while trapping edge-case ID collisions;
  - **16. UI 2x Lossless Coordinate Mapping (`ui_main.py`)**:
    Locks the video rendering viewport to $640 \times 360$ (exact 16:9 ratio of $1280 \times 720$ ). Mouse click calibration events are multiplied by 2 to map losslessly back to raw sensor pixels, eliminating coordinate drift caused by window resizing.
- **Capture State Deadlock & Semi-Automatic Spacebar State Machine**:
  - **Deadlock Caused by Unidentified Characters**: Because the reduced-dimension model localized bounding boxes without recognizing character identities, the algorithm had **no way of identifying which piece disappeared**. When piece count dropped from 32 to 31 upon capture, the system could not programmatically distinguish between a genuine capture and a temporary hand-occlusion dropout, breaking piece-conservation invariants;
  - **Human-in-the-Loop State Machine Compromise**: Resolved by introducing a **"Player Move Confirmation Button / Keyboard Spacebar"** in the UI. After completing a move and retracting hands, the human player hits the spacebar (`force_detect_move = True`). The system samples the static board free of hand shadows. This guaranteed 100% operational reliability for class delivery, though at the expense of **inability to recognize arbitrary mid-game layouts** and **reliance on manual keystroke confirmation**.

#### Generation 4 Autonomous Perception (Gen 4: Full Grid Tensor Classification & Synthesis Pipeline)
- **Autonomous End-to-End Perception Paradigm**:
  To eliminate manual confirmation and support arbitrary mid-game / end-game setups, the vision architecture was re-engineered from the ground up into a deep spatial tensor classification pipeline:
- **Gen 4 Algorithmic & Architectural Innovations**:
  - **1. Single-Piece Micro-Crop & Alpha Mask Library**:
    Photographed all 32 custom magnetic pieces under high magnification, generating clean transparent cutouts with preserved alpha masks across all 14 Chinese chess piece classes, alongside clean empty-board reference images;
  - **2. Automated Board Synthesis & Ground Truth Tensor Generator (`CChessMixSinglePngCls`)**:
    Built an automated synthesis pipeline applying randomized affine perturbations, orientation angles ( $\pm 15^\circ$ ), and HSV photometric noise to composite individual piece cutouts across the 90 intersections (A0 to J8). Automatically produces synchronized $10 \times 9 \times 16$ Ground Truth label tensors, synthesizing tens of thousands of diverse mid-game scenarios with zero manual labeling overhead;
  - **3. RTMPose-Tiny 4-Corner Pose Regression & Automatic Rectification (`cchess_pose`)**:
    A lightweight coordinate-classification pose network (RTMCC) regresses the 4 board corners (A0, A8, J0, J8) directly from raw oblique camera feeds, automatically performing inverse perspective mapping to render standardized top-down orthographic images;
  - **4. Swin-Transformer V2 Hierarchical Global Feature Backbone (`cchess_reg`)**:
    Processes standardized top-down board images through Swin-Transformer V2 with shifted-window self-attention, capturing multi-scale board topology and long-range contextual dependencies between opposing pieces;
  - **5. Spatial Aggregation Neck `Chess10_9Neck` & Multi-Region Classification Head `CChessTableHead`**:
    Reshapes and pools global feature representations into the physical $10 \times 9$ spatial grid, concurrently predicting 16 class logits (7 Red, 7 Black, empty `.`, obstacle `x`) across all 90 grid intersections;
  - **6. Composite Loss Supervision (`KLDiscretLoss` + `SmoothL1Loss`) with Full-err-zero Metric**:
    Supervised via discrete KL divergence and Smooth L1 losses, benchmarked against a zero-error-per-board (Full-err-zero) evaluation threshold;
  - **Closed-Loop Breakthrough**: Delivers ultra-high accuracy and low error, **mastering arbitrary mid-game and endgame setups without prior game history**, completely immune to capture count deadlocks, and eliminating manual keystroke confirmation for true 100% autonomous perception.

<div align="center">
  <img src="docs/images/vision_real_experiments.png" width="820px" alt="Perspective Rectification and 10x9 Grid Tensor Inference"/>
  <p><b>Generation 4 Experimental Bench: Board perspective rectification (Left: Step 2 orthographic stretch) and 10×9 full-board grid tensor classification (Right: 0.54s per-frame inference, achieving automated digital twin mapping)</b></p>
</div>

---

### 4.3 Game Decision & Rule Engine Module (Lead: Zhang Zichen)

The decision engine evaluates full-board positions, enforces legal moves, and selects tactical trajectories via a rule-constraint solver and a deep convolutional evaluation network:

#### 1. 10×9 Chinese Chess Constraint Solver (`game/board.py`)
Implements the comprehensive rule set of Chinese Chess, providing legal move validation, check/checkmate detection, and state tree search:
- **Raycast Obstacle Collision (Chariot/Cannon)**:
  `is_path_clear` steps through collinear intersections:
  - **Chariot (Ju)**: Requires collinear source and target with intervening obstacle count $N_{\text{obs}} = 0$;
  - **Cannon (Pao)**: Move without capture requires $N_{\text{obs}} = 0$ onto an empty square; capture strictly mandates **exactly one cannon mount ( $N_{\text{obs}} = 1$ )** onto an opponent piece;
- **Horse Leg Obstacle Detection (`is_ma_valid`)**:
  Enforces $|dx| \times |dy| = 2$. If moving horizontally ( $|dx|=2$ ), verifies that midpoint $(y_1, \frac{x_1 + x_2}{2})$ is empty; if moving vertically ( $|dy|=2$ ), verifies that midpoint $(\frac{y_1 + y_2}{2}, x_1)$ is empty;
- **Elephant Eye & River Boundary Constraints (`is_xiang_valid`)**:
  Enforces $|dx|=2, |dy|=2$ with unblocked center $(\frac{y_1 + y_2}{2}, \frac{x_1 + x_2}{2})$. Red elephants are constrained to $y_2 \ge 5$, while Black elephants are constrained to $y_2 \le 4$;
- **Advisor Palace Diagonal Constraints (`is_shi_valid`)**:
  Restricted to $|dx|=1, |dy|=1$ within the Nine-Palace boundaries: Red Advisor confined to $3 \le x_2 \le 5, 7 \le y_2 \le 9$; Black Advisor confined to $3 \le x_2 \le 5, 0 \le y_2 \le 2$;
- **General Orthogonal Movement & Flying General Rule (`is_jiang_valid`)**:
  Restricted to $|dx| + |dy| = 1$ within the palace; enforces the "Flying General" prohibition against unobstructed collinear generals;
- **Soldier Forward / Lateral River Transition State Machine (`is_zu_valid`)**:
  - Red Soldier: Prior to crossing river ( $y_1 > 4$ ), only forward motion allowed ( $dx=0, dy=-1$ ); after crossing ( $y_1 \le 4$ ), allows forward or lateral moves ( $dy=-1$ or $dy=0, |dx|=1$ ), backward moves forbidden;
  - Black Soldier: Prior to crossing river ( $y_1 < 5$ ), only forward motion allowed ( $dx=0, dy=1$ ); after crossing ( $y_1 \ge 5$ ), allows forward or lateral moves ( $dy=1$ or $dy=0, |dx|=1$ ), backward moves forbidden;
- **Full Legal Move Enumeration (`get_all_legal_moves`) & Undo Stack**:
  Scans all active pieces to enumerate all valid actions across 90 intersections, driving AI decision trees and checkmate verification.

#### 2. 14-Channel Spatial Binary Feature Tensor Extractor (`get_state_tensor`)
Encodes the discrete board into a spatial binary tensor $\mathbf{S} \in \mathbb{R}^{14 \times 10 \times 9}$:
- Channels 0–6: One-Hot distribution for Black General, Chariot, Horse, Cannon, Elephant, Advisor, Soldier;
- Channels 7–13: One-Hot distribution for Red General, Chariot, Horse, Cannon, Elephant, Advisor, Soldier;
Squares occupied by the respective piece evaluate to $1.0$, while vacant squares evaluate to $0.0$, feeding structured topological inputs to the neural network.

#### 3. Deep Convolutional Decision Network `ChessNet` & Action Masking (`game/ai.py`)
- **Neural Network Architecture**:
  - **Convolutional Feature Extractor**: 3 sequential 2D convolutional layers with expanding channels:
    - $\text{Conv1}: 14 \to 64$ (Kernel $3 \times 3$, Padding 1, ReLU)
    - $\text{Conv2}: 64 \to 128$ (Kernel $3 \times 3$, Padding 1, ReLU)
    - $\text{Conv3}: 128 \to 128$ (Kernel $3 \times 3$, Padding 1, ReLU)
  - **Fully-Connected Head**: Flattened to $128 \times 10 \times 9 = 11,520$ features, projected through 2 linear layers:
    - $\text{Linear1}: 11,520 \to 1024$ (ReLU)
    - $\text{Linear2}: 1024 \to 8100$ (Unnormalized Logits)
- **Action Space Indexing & Legal-Move Masking**:
  - The 8100-dimensional action space spans all directional transfers between 90 board squares ( $90 \times 90 = 8100$ ), indexed as:
    $$\text{idx} = (y_1 \times 9 + x_1) \times 90 + (y_2 \times 9 + x_2)$$
  - **Action Masking**: Inference filters outputs strictly over the candidate set returned by the rule engine:
    $$a^* = \arg\max_{m \in \mathcal{M}_{\text{valid}}} \text{Logits}[\text{idx}(m)]$$
    completely preventing illegal move execution;
- **High-Availability Graceful Degradation**:
  If the pretrained weight file `aaa.pth` is absent, the decision engine degrades gracefully to a uniform random policy (`random.choice(valid_moves)`), preventing system lockups during demonstrations.

#### 4. Game Decision Optimization Roadmap
Future iterations will upgrade the convolutional evaluation model to a self-play **Reinforcement Learning (RL)** framework to expand tactical calculation depth in complex endgames.

---

## 5. Hardware Electrical Topology & Bus Definitions

The physical hardware electrical topology is organized below:

| Module | Physical Interface | Protocol | Host Pins | Parameters & Electrical Specs |
| :--- | :--- | :--- | :--- | :--- |
| **PCA9685 Servo Driver** | SDA / SCL | Hardware I2C (100kHz Standard) | CH347 D0(SCL) / D1(SDA) | 16-ch 12-bit PWM, 50Hz ( $T = 20$ ms ) |
| **Base Yaw (J1)** | PWM Channel 0 | 50Hz PWM | PCA9685 Output 0 | RDS3235 Metal Servo, Azimuth |
| **Shoulder Pitch (J2)** | PWM Channel 4 | 50Hz PWM | PCA9685 Output 4 | RDS3235 35kg·cm High-Torque Servo, Lift |
| **Elbow Pitch (J3)** | PWM Channel 8 | 50Hz PWM | PCA9685 Output 8 | 20kg·cm Digital Metal Servo, Radius & Elevation |
| **Wrist Pitch (J4)** | PWM Channel 12 | 50Hz PWM | PCA9685 Output 12 | 180° Metal Servo, Keeps Suction Vertical |
| **Electromagnet Relay** | Control Signal | CH347 Native GPIO | CH347 GPIO2 (Pin 10 / DTR0) | 5V DC relay driving electromagnet, mask 0x04 |
| **Overhead Camera** | USB 2.0 / UVC | DirectShow | PC USB 3.0 Port | 1280×720 @ 30 FPS, low-distortion lens |
| **WCH CH347 Bridge** | USB Type-C | 480Mbps High-Speed USB | PC USB Port | Hardware USB to I2C/GPIO bridge |

### PCA9685 12-Bit PWM Mapping Model
Operating at $T = 20$ ms (50Hz), the pulse width converts to register ticks via:

$$
\text{Tick} = \left\lfloor \frac{T_{\text{pulse}}}{20\text{ ms}} \times 4096 \right\rfloor
$$

For RDS3235 servos ( $\theta \in [0^\circ, 270^\circ]$ , pulse width 0.5 ~ 2.5 ms ):

$$
T_{\text{pulse}}(\theta) = 0.5\text{ ms} + \frac{\theta}{270^\circ} \times 2.0\text{ ms}
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
