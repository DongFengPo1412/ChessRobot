# ChessRobot technical notes

[简体中文](technical-notes.zh-CN.md) · **English** · [日本語](technical-notes.ja.md) · [Project overview](../README_EN.md)

**On this page:** [Vision](#vision-history) · [Models](#mathematical-models) · [Motion](#motion-hardware) · [Decision](#game-decision) · [Validation](#validation) · [Gallery](#hardware-gallery)
This document explains design choices, algorithms, and evidence. It distinguishes the course prototype, historical experiments, and fourth-generation development. Runtime instructions are in the [setup guide](setup.en.md).

## Implementation map

| Component | Code | Responsibility |
| :--- | :--- | :--- |
| Orchestration | [main.py](../main.py) | Coordinate conversion, vision adapter, queue, turns, and suppression of reports during motion. |
| Current vision | [find_chess_new.py](../localization/find_chess_new.py) | YOLO tracking, calibration, grid mapping, stabilization, and confirmation. |
| Earlier vision | [localization/](../localization/) | Experimental variants, not all invoked by the current entry point. |
| Acquisition and training | [recognition/](../recognition/) | Physical-image collection and YOLO fine-tuning. |
| Motion | [kinematics.py](../kinematics.py), [control.py](../control.py) | IK, compensation, interpolation, and pickup/transfer. |
| Hardware | [hardware.py](../hardware.py), [config.py](../config.py) | I2C/GPIO, PWM commands, geometry, and calibration. |
| Board and decision | [board.py](../game/board.py), [ai.py](../game/ai.py) | Movement constraints, state encoding, and move selection. |
| GUI prototype | [ui_main.py](../ui_main.py) | Controls and intended video/board displays; integration work remains. |

<a id="vision-history"></a>

## Vision development

### Generation 1 — Separate localization and character recognition

The initial approach detected circles, extracted crops, estimated color, and classified characters. Historical local project `PythonProject1` contains Hough-circle processing and `ChessResNet18`: a pretrained ResNet-18 adapted to one-channel input and an 11-class head with a 256-unit hidden layer and dropout.

Synthetic fonts made samples easy to generate, but did not reproduce overhead-camera blur, reflections, perspective, and low character resolution. Circle sensitivity also varied with lighting. These findings motivated real-image acquisition. The early local code is part of development history and is not fully included in this repository.

### Generation 2 — Physical data and multi-class YOLO

Chaoran Liu collected and manually annotated the physical vision data. Team records describe approximately 500 images of the original 40 × 40 cm dark-green mat, followed by 120 after changing to a 35 × 35 cm beige mat. The smaller board addressed the original mechanical reach problem, while the background change introduced domain shift.

Local dataset exports progressed through multiple Roboflow versions. A historical label-conversion script in `PythonProject2` later collapsed all piece classes into one for localization-focused detection.

The checked-in [training script](../recognition/train.py) specifies 1280-pixel input, up to 200 epochs, batch size 4, and GPU device 0. It requests rotation, translation, scale, shear, perspective, flips, Mosaic, MixUp, erasing, copy-paste, HSV perturbation, classification-loss weighting, and label smoothing. Configuration alone does not prove that every installed Ultralytics version applies every option to this task. Reproduction requires the package version, checkpoint, dataset, and run log.

Similar characters remained difficult at camera distance. Mapping a detection center to the board does not rectify the character crop itself; these are different uses of projective geometry.

![Physical-board dataset annotation](images/vision_pipeline_evolution.png)

The image illustrates annotation work, not a held-out recognition score.

<a id="training-records"></a>

### Historical training records and version mapping

The local `PythonProject2/recognition/runs/detect/` archive contains the following `args.yaml` and `results.csv` files. Logged epochs count CSV data rows; configured epochs are the requested upper limit.

| Experiment directory | Input size | Configured epoch limit | Logged epochs |
| :--- | ---: | ---: | ---: |
| `Chess_Robot_V3` | 640 | 100 | 100 |
| `Chess_V4_Final` | 640 | 100 | 100 |
| `Chess_V5_Ultimate_1280` | 1280 | 150 | 150 |
| `Chess_V6_Pro_1280` | 1280 | 200 | 75 |
| `Chess_V7_Ultra_Loc_1280` | 1280 | 150 | 137 |

These records support the history of repeated training and input-resolution changes. Run identifiers V3–V7 do not correspond to the four vision generations in this document. Directory names alone do not identify detector classes: the historical relabeling script converts labels to one class, but each checkpoint must be matched with its contemporary data configuration, labels, and metadata.

Some records were copied into `TianProject` and must not be counted as independent experiments. The full logs, old datasets, and models are not distributed here. This table documents training history; it does not turn detector metrics into physical-system accuracy.

<a id="vision-generation-3"></a>

### Generation 3 — Localization and temporal state tracking

The course-delivery approach simplified recognition to locating pieces and following changes from a known opening. The software board maintains character identities. Human confirmation selects the moment for comparing physical board snapshots.

| Mechanism | Function | Code or version |
| :--- | :--- | :--- |
| Four-point homography | Map camera coordinates to columns 0–8 and rows 0–9. | Current `set_black_view`, `cv2.findHomography`. |
| Expanded ROI | Exclude most off-board background and captured-piece areas; extend corners outward by 60 pixels. | Current `calibrate` and `run`. |
| Asynchronous capture | Keep the latest frame in a background thread; request driver buffer size 1. | Current `VideoStream`; no guaranteed zero-latency measurement. |
| LAB-channel CLAHE | Enhance contrast before detection. | Current `run`, clip limit 3.5, 8 × 8 tiles. |
| Low-luminance heuristic | Intended black-piece orientation estimation from cropped detections. | Current `is_black_piece`; issue below. Not a YOLO color model. |
| Manual orientation | Select which edge represents the black baseline. | Current keys `1` and `2`. |
| River-band rejection | Reject detections in the decorative river gap. | Current `4.35 < y < 4.65`. |
| Confidence thresholds | Require 0.50 for new tracks and retain existing tracks from 0.20. | Current detector/tracker logic. |
| Centroid smoothing | EMA coefficient 0.28 and 1.5-pixel displacement deadzone. | Current `smooth_centers`. |
| Motion gating | Clear the initialization window when centers move. | Current `global_motion`, squared threshold 25. |
| Track energy | Detection +40, miss −15, active above 50. | Current `piece_energy`. |
| History voting | Select the most frequent grid position from a 12-frame history. | Current `piece_history` and `Counter`. |
| Count and confirmation | Accept expected count or one fewer; submit after Space. Initialization requires more than 10 stable samples. | Current `run`. Buffer capacity 35 is not a 35-frame vote on every move. |
| Occupancy/ID difference | Infer source/destination from missing positions and changed track IDs. | Current `VisionAdapter.report_and_compare` in `main.py`. |
| Motion-time suppression | Block reports and reset the baseline after arm motion. | Current adapter/orchestrator; camera capture continues. |
| GUI coordinate scaling | Double clicks from a 640 × 360 display for intended 1280 × 720 input. | `ui_main.py`; depends on those dimensions and correct display mapping. |
| MOG2 hand-motion detection | Background subtraction and motion-region gating. | Earlier `localization/find_chess.py`, not current V15.4. |
| Fast-displacement bypass | Reset smoothing above squared displacement 2500. | Historical local `find2.txt`, absent from current V15.4. |
| Exposure and persistence | DirectShow/gain settings and saved calibration experiments. | Historical local variants; current entry point recalibrates. |

The table preserves useful experiments without implying they all run together. ROI masks cannot reject every distraction inside the polygon, and temporal filtering cannot guarantee correct identity after every capture.

### Generation 4 — Rectified board and grid classification

The development direction combines corner estimation with classification of a normalized board:

1. Prepare project-specific piece images and empty-board images.
2. Composite pieces onto intersections and update labels.
3. Estimate four corners with an RTMPose/MMPose-based model and rectify perspective.
4. Use Swin Transformer V2 and a spatial neck/head to classify 90 intersections into 16 states: seven red types, seven black types, empty, and other/obstacle.
5. Evaluate independent images and sequences before connecting state changes to game logic and motion.

The reference is [TheOne1006/chinese-chess-recognition](https://github.com/TheOne1006/chinese-chess-recognition), built on OpenMMLab. `CChessMixSinglePngCls`, `CChess10_9Neck`, and `CChessTableHead` provide relevant implementation examples. Project-specific data, adaptation, training, and evaluation should identify the team's changes separately from these components.

The reference pose configuration uses `KLDiscretLoss` and `SoftWeightSmoothL1Loss`. Its Swin classification configuration uses `LabelSmoothLoss`. These supervise different branches.

The robot entry points still use Generation 3. Fourth-generation training artifacts, evaluation records, and a robot adapter are not included here. The development plan therefore does not establish completed training or integrated autonomous gameplay.

The reference README reports `full-err-zero = 0.83` on its validation data: 83% of evaluated board images had every intersection correct. It does not mean every input is error-free, and it is not a ChessRobot measurement.

![Rectification and independent recognition example](images/vision_real_experiments.png)

This screenshot shows a rectification/inference interface and a displayed 0.54 s runtime for that example. It is not a repeatable latency benchmark, aggregate accuracy result, or evidence of robot integration.

<a id="motion-hardware"></a>

## Motion and hardware

### Board coordinates and motion targets

`control.py:board_to_physical` indexes columns a–i as c = 0–8 and rows as r = 0–9. The arm sits behind the black baseline: column a has positive x, and y increases toward the far side. Let w and h be grid spacing, h_r river spacing, y_0 the board offset, and z_b board height:

```math
x=(4-c)w,\qquad z=z_b,\qquad
y=y_0+\begin{cases}rh,&0\le r\le4\\4h+h_r+(r-5)h,&5\le r\le9\end{cases}
```

The separate river term avoids treating the entire board as a uniformly spaced rectangle. `main.py:Coord` handles orientation conversion between vision, game coordinates, and arm-facing coordinates; image-grid coordinates are not millimeter targets.

### Geometry and compensation

The checked-in link lengths are `L1 = 73`, `L2 = 231`, `L3 = 231`, `L4 = 66` mm. Board spacing is 34.2 mm, river spacing 34.0 mm, `BOARD_ORIGIN_Y = 200`, `BOARD_Z = −33`, and `SAFE_Z = 70` mm. Configuration values alone do not establish reach over the entire board.

The active IK applies empirical corrections before its geometric solution:

```text
comp_x = 1.125 * x
R_target = sqrt(comp_x² + y²)
R_comp = 0.94 * R_target
comp_z = z + 0.12 * R_target
delta_z = comp_z + L4 - L1
D = sqrt(R_comp² + delta_z²)
```

The solver checks two-link reach, applies the law of cosines to shoulder/elbow angles, and constrains the wrist toward a downward orientation. The height term is distance-dependent feedforward, not an identified elasticity model or feedback controller. No universal placement accuracy or IK runtime is claimed.

### Control choices and tradeoffs

| Engineering problem | Implemented method | Reason and scope |
| :--- | :--- | :--- |
| Fixed links and downward tool orientation | Two-link geometric IK, base rotation, and wrist constraint. | Direct geometric solution for this fixed mechanism, with a reach check first. |
| Reach-dependent placement offsets | Empirical lateral, radial, and height correction. | Easy to tune against physical offsets; coefficients depend on hardware and calibration. |
| Contact with neighboring pieces near the board | Clearance-height transfer, 8 mm hover, and slow vertical approach/withdrawal. | Explicit waypoints constrain approach direction; this is a prescribed path, not general obstacle search. |
| Large command jumps and tiny displacement increments | Cartesian linear interpolation and a 0.1 mm increment threshold. | Splits travel into small steps; S-curve velocity planning is not implemented. |

For a segment lasting T seconds, interpolation follows the equation below when the deadzone is ignored. Actual loop timing also includes computation, communication, and host scheduling.

```math
N=\max(1,\lfloor50T\rfloor),\qquad
\mathbf{p}_i=\mathbf{p}_0+\frac{i}{N}(\mathbf{p}_{\mathrm{target}}-\mathbf{p}_0),\quad i=1,\ldots,N
```

If the requested total displacement on an axis is at most 0.1 mm, its loop increment is zero; the final target is still commanded. This is a software threshold, not positioning accuracy. Pickup/placement sequences approach, attach, lift, transfer, release, and withdraw. Captures remove the target piece before reusing the normal transfer routine.

### Electrical interfaces

| Interface | Implementation |
| :--- | :--- |
| PC to servos | CH347 USB bridge, I2C address `0x40`, PCA9685. |
| Channels | Base 0, shoulder 4, elbow 8, wrist 12. |
| PWM timing | Prescaler 121, approximately 50 Hz; channel-specific pulse limits. |
| Pulse conversion | `tick = int(clamped_us / 20000 * 4096)`. |
| Electromagnet | GPIO2 mask `0x04` commands the relay. |

The PC sends I2C/GPIO commands through the USB bridge, coordinating servo and electromagnet interfaces without measured joint-position feedback.

<a id="mathematical-models"></a>

## Mathematical models and code

The following equations retain the main models needed to understand the implementation. They explain code behavior rather than establish accuracy, timing, or reliability.

### Homography and grid mapping

Let camera pixel coordinates be p = (x, y, 1) and board coordinates q = (u, v, 1), both homogeneous. Estimate H from four corners, then normalize to obtain grid coordinates. A homography corrects planar perspective; it does not replace lens-distortion calibration.

```math
\tilde{\mathbf{q}}=H\mathbf{p},\qquad u=\frac{\tilde q_1}{\tilde q_3},\quad v=\frac{\tilde q_2}{\tilde q_3}
```

### ROI and temporal stabilization

Expand each ROI corner radially from board center C with b = 60 pixels. The centroid EMA coefficient is 0.28; current code updates only beyond a 1.5-pixel displacement. Discrete grid positions use a 12-frame mode; energy and count gates are listed above.

```math
\mathbf{P}'_i=\mathbf{P}_i+b\frac{\mathbf{P}_i-\mathbf{C}}{\lVert\mathbf{P}_i-\mathbf{C}\rVert}
```

```math
\hat{\mathbf{c}}_t=(1-\lambda)\hat{\mathbf{c}}_{t-1}+\lambda\mathbf{c}_t,\qquad\lambda=0.28
```

### Analytical inverse kinematics

Set R = R_comp and z = delta_z from the compensation definitions above; D is the shoulder-to-effective-wrist distance. Check two-link reach before applying the law of cosines. Angles below are in radians; code converts them to degrees and applies servo-reference offsets. The additional wrist compensation coefficient is currently zero.

```math
D=\sqrt{R^2+z^2},\qquad |L_2-L_3|\le D\le L_2+L_3
```

```math
\alpha=\arccos\frac{L_2^2+D^2-L_3^2}{2L_2D},\qquad\gamma=\arccos\frac{L_2^2+L_3^2-D^2}{2L_2L_3}
```

```math
\beta=\mathrm{atan2}(z,R),\quad q_2=\beta+\alpha,\quad q_3=q_2-(\pi-\gamma),\quad q_4=-\frac{\pi}{2}-q_3
```

### PWM and move selection

In the PWM equation, u is the clamped pulse width in microseconds. Apply channel limits before converting to PCA9685 ticks. The network selects the highest score within the candidate set M returned by the rule module. Candidate validity is limited by that module implementation.

```math
n=\left\lfloor\frac{u}{20000}\times4096\right\rfloor,\qquad a^{\star}=\arg\max_{m\in\mathcal{M}} f(S)_{\mathrm{idx}(m)}
```

<a id="game-decision"></a>

## Game decision and integration

`game/board.py` implements the board, piece-specific movement constraints, candidate enumeration, turn switching, a history list, and termination when a general is absent. It does not implement a full check/checkmate adjudicator or general undo API. Candidate filtering inherits the rule module's limitations.

`ChessNet` uses three 3 × 3 convolutional layers with channels 14 → 64 → 128 → 128, then linear layers 11,520 → 1,024 → 8,100. Input is a 14 × 10 × 9 piece-location tensor. An output index is `(source_row * 9 + source_col) * 90 + destination_row * 9 + destination_col`. The highest-scoring candidate returned by the rule module is selected.

The current decision path contains no Minimax or Alpha–Beta search. Learned scoring needs `game/aaa.pth`; otherwise it explicitly chooses a random candidate. A network definition alone does not establish trained playing strength.

### State encoding and candidate filtering

Fourteen channels represent seven piece types for each side. Occupied intersections are 1 in the matching channel and 0 elsewhere. A square is row × 9 + column; source × 90 + destination gives the action index. The 8,100-dimensional output contains all source/destination pairs, including combinations that violate piece movement rules.

The network output therefore is not maximized over every action. The rule module generates candidates first, and inference compares only their logits. Raw logits are ranking scores, not calibrated probabilities of correct play. This combines explicit movement constraints with learned scoring rather than searching a multi-ply game tree.

### Historical supervised learning from game records

The team's local presentation archive contains `1ai(1).py`, a historical training path. It reads `CHESS_BOOKS`, replays moves, and constructs a 14 × 10 × 9 pre-move state and next-action label. A three-convolution network predicts 8,100 actions with `CrossEntropyLoss` and Adam. Function defaults are 20 epochs, batch size 32, and learning rate 0.001, with output `chess_ai_model.pth`. These are code settings, not proof of a completed training run or measured playing strength.

For logits z, game-record labels t_i, and batch size B, the cross-entropy objective is:

```math
\mathcal{L}=-\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp z_{i,t_i}}{\sum_{k=0}^{8099}\exp z_{i,k}}
```

That historical file is not distributed in this repository. Its layer names and checkpoint filename differ from current `game/ai.py`; renaming a checkpoint does not establish compatibility. Another local version, `TianProject/game/ai.py`, uses two convolutions, while the current entry point uses three and a hidden fully connected layer. Reproduction requires matching code, training data, and checkpoint structure. Historical supervised learning and future self-play reinforcement learning are separate stages.

<a id="system-coordination"></a>

### Orchestration and state consistency

In `main.py`, vision runs in a separate thread and submits human moves through `move_queue`. The main thread converts coordinates, checks and updates the board, and selects a response. It updates the software board before dispatching a transfer or capture.

During actuation, `suppress_detection` blocks visual move reports while camera capture continues. After the sequence returns, the program updates the expected count, rebuilds the vision baseline, and restores reporting. This timing coordination prevents modules from triggering one another; it is not camera-based measurement and correction of placement error. Software-state updates, returned command sequences, and successful physical placements require separate validation.

<a id="validation"></a>

## Validation and development boundaries

The team reported **18 center-covering placements out of 20 consecutive moves**. Retain this sample size and criterion when reporting the result. It does not define a millimeter tolerance, repeated test conditions, recognition accuracy, or complete-match reliability.

Dataset counts and historical behavior reflect team records. This repository does not publish every raw image, run log, or a fourth-generation held-out benchmark.

| Area | Observation | Consequence |
| :--- | :--- | :--- |
| Color/orientation | `is_black_piece` divides a single-channel count by three-channel `roi.size`; the ratio cannot exceed 1/3, below the 0.5 threshold. | Automatic orientation cannot pass. Use manual selection until corrected and tested. |
| GUI video | GUI assigns `ui_callback`, but current vision does not call it. | Embedded video is not fully wired; OpenCV windows remain relevant. |
| GUI board | Rendering expects `piece.name` and `piece.is_red`; board pieces are tuples. | The display interface needs adaptation. |
| Reach/configuration | `tools/calc_reach.py` has historical constants differing from active geometry/compensation; current far-edge targets can fail IK reach checks. | The tool cannot certify full-board reach for this checkout. Reconcile calibration before operation. |
| Motion outcome | Interpolation does not propagate every failed `execute_move` result as a verified physical outcome. | Completed command sequences do not prove successful placements. |

These observations concern the checked-in software and do not rewrite the historical demonstration record.

<a id="hardware-gallery"></a>

## Hardware gallery

These materials preserve physical evidence from the course prototype. They illustrate motion and hardware, rather than establish arbitrary-layout recognition or aggregate success rates.

![Second physical demonstration view](images/demo_robot_arm_2.gif)

| Electromagnetic pickup on the physical robot | Physical placement and release |
| :--- | :--- |
| ![Electromagnetic pickup on the physical robot](images/demo_live_arm_pickup.jpg) | ![Physical placement and release](images/demo_live_arm_place.jpg) |

[Full video](https://www.bilibili.com/video/BV1aXLn6HEQd) · [Overview demonstration](../README_EN.md). Dataset annotation and independent recognition examples appear in their respective generation sections.

## Next development milestones

- **Vision:** identify the dataset/transformations; separate physical train/test scenes; train or fine-tune; report intersection-level and board-level results; test captures/occlusion; validate a state adapter on the robot.
- **Motion:** align configuration and reach calculations; repeat placement tests over near/far and edge/center positions; compare smoother acceleration profiles with current interpolation.
- **Decision:** test movement constraints and establish checkpoint provenance/strength, then select improvements with the module owner. Self-play reinforcement learning remains an exploratory option.
- **Integration:** complete GUI interfaces and validate startup, stopping, capture recovery, and state synchronization.

New results should state version, hardware, data/test conditions, metric, sample count, and limitations. Referenced implementations retain their licenses; contributions should identify the actual project-specific work.
