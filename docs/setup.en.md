# ChessRobot setup and operation

[简体中文](setup.zh-CN.md) · **English** · [日本語](setup.ja.md) · [Project overview](../README_EN.md)

This guide describes the current Generation 3 entry point. It requires the physical robot and calibration for that robot. The repository has no integrated simulation-only switch. Fourth-generation training is a separate development task described in the [technical notes](technical-notes.en.md).

## Environment

- Windows 10/11 and 64-bit Python 3.10 or later.
- A CUDA-capable PyTorch installation and compatible NVIDIA environment. Current vision calls `model.to('cuda')` directly; it does not implement CPU fallback.
- A USB camera, WCH CH347 driver, matching 64-bit `CH347DLLA64.DLL`, PCA9685, four servos, and the electromagnet/relay assembly.

`requirements.txt` lists lower bounds for PyTorch, Ultralytics, OpenCV, NumPy, and PyQt5. It is not a tested lockfile. Preserve the exact working package versions when recording a reproduction. Historical vision scripts may also need Pillow and a font file.

From PowerShell:

```powershell
git clone https://github.com/TJS-Git-Hub/ChessRobot.git
cd ChessRobot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import torch, cv2; print(torch.__version__); print(torch.cuda.is_available()); print(cv2.__version__)"
```

If CUDA is unavailable, resolve the PyTorch/driver environment before launching the current vision code. Installing dependencies alone does not confirm hardware operation.

## Files to prepare

| Asset | Location | Notes |
| :--- | :--- | :--- |
| WCH DLL | `CH347DLLA64.DLL` in the project root | Obtain the matching vendor library/driver. DLLs are ignored by Git and are not supplied by a clean clone. |
| Piece detector | `recognition/models/best.pt` | A checkpoint is tracked, but its class mapping must match the chosen vision version. Do not substitute the root `yolo11n.pt` solely on its filename. |
| Optional move-selection weights | `game/aaa.pth` | Not supplied in the current tree. Without it, the decision engine chooses a random candidate. |
| Calibration arrays | `recognition/models/H_matrix.npy` and `ROI_corners.npy` | Used by earlier scripts; the current entry point still performs interactive calibration. |
| YOLO training data | `recognition/data/data.yaml` and referenced data folders | Raw images and run outputs are excluded from Git; check paths and label definitions before training. |

The Generation 3 detector is used for localization. Physical piece identities come from standard opening positions and game history. Checkpoint filenames do not prove a 14-class model, quantization, or training performance.

## Hardware and calibration

| Connection/setting | Code reference |
| :--- | :--- |
| CH347 device index 0; PCA9685 I2C address `0x40` | `hardware.py` |
| Servo channels 0 / 4 / 8 / 12: base / shoulder / elbow / wrist | `config.py` |
| CH347 GPIO2 mask `0x04` to relay control | `hardware.py` |
| Servo pulse references and per-channel limits | `SERVO_CONFIG` in `config.py` |
| Board origin, spacing, height, and clearance | `config.py` |
| Empirical X/radius/height compensation | `kinematics.py` |

Use the board, servo power distribution, common reference, and relay wiring appropriate to the actual assembly. Put the board and captured-piece area within a verified motion envelope. Controller construction calls `go_to_wait()` and can move the arm immediately.

The following tools have different effects:

```powershell
python tools/calibrate_servo.py
python tools/test_electromagnet.py
python tools/calc_reach.py
```

- `calibrate_servo.py` commands the servos, accepts coarse/fine pulse adjustments, and prints values to copy into `config.py`; it does not automatically save them.
- `test_electromagnet.py` operates the real relay/electromagnet.
- `calc_reach.py` reuses the current `board_to_physical` and `solve_ik` without opening hardware. It rejects 11/90 targets under the current configuration and exits nonzero when targets are rejected. Geometric reach does not certify servo limits, clearance or pickup success.

Reconcile geometry and test individual target coordinates with the active IK before attempting a full-board game. Current far-edge targets can fail reach checks. The checked-in parameters should not be treated as a calibration for another robot.

## CLI workflow

1. Prepare the detector, DLL, drivers, and calibrated hardware. Current `VideoStream` defaults to camera index `1` and requests 1280 × 720. Acquisition scripts use index `0`; align them with the intended camera.
2. Place the 32 pieces in the standard opening arrangement, with the arm behind the black baseline.
3. Start the robot:

   ```powershell
   python main.py
   ```

4. In the OpenCV window, click board corners clockwise: top-left, top-right, bottom-right, bottom-left.
5. Select orientation manually when needed: `1` means black is on the far/top image side; `2` means black is on the near/bottom side. The brightness ratio now counts single-channel pixels correctly; orientation under real lighting still requires testing.
6. Wait for initialization. Red/human moves first. Move a piece, remove the hand from view, focus the vision window, and press **Space** to submit the snapshot.
7. The robot chooses a response, moves, returns toward standby, and resets the visual baseline. Capture handling also transfers the captured piece to the off-board graveyard.
8. Press `q` in the vision window or interrupt the CLI to end the session. Shutdown requests a stop and closes the interface without starting a return-to-standby motion. Software stopping is not a hardware power-disconnect guarantee.

Do not change the piece count or position while a baseline is being locked. If a capture causes an ambiguous ID change, the program can request lifting the attacking piece briefly and replacing it before confirming again. Reinitialize a consistent board if physical and software states diverge; recovery is not fully autonomous.

## GUI prototype

```powershell
python ui_main.py
```

The interface has a **Start Match** button, a software stop button, logging, and Space/1/2 shortcuts. Clicking Start constructs the same hardware orchestrator.

The GUI now receives calibration/tracking frames, maps clicks using actual frame dimensions, scaled pixmap size and letterboxing, and renders tuple pieces from locked board snapshots. Stop requests end capture and interpolation; shutdown no longer starts a homing motion. Camera/Qt/OpenCV integration and calibration accuracy still require physical testing.

## Troubleshooting

| Symptom | Check |
| :--- | :--- |
| DLL cannot load / CH347 open fails | Windows/Python/DLL bitness, vendor driver, library location, USB connection, and device index. |
| Camera cannot open or wrong view appears | Index, camera occupancy, USB connection, and actual frame resolution. |
| CUDA error | PyTorch build, NVIDIA environment, and CUDA availability. There is no automatic CPU branch in current vision. |
| No orientation or baseline | Manual 1/2 selection, corner order, correct model, complete opening, and stable detections. |
| No move after a human turn | Vision-window focus, Space, piece count, track IDs, and the last baseline. |
| Implausible/random responses | `game/aaa.pth` availability and compatibility; rule checks are not tournament-complete. |
| Empty GUI video / board-render exception | Check captured frames, calibration click mapping and logs. Callback/tuple interfaces are fixed; live Qt/camera integration remains unverified. |
| Unreachable or offset placement | Active geometry, board origin/height, IK compensation, servo calibration, and pulse limits. Run the current `tools/calc_reach.py` and confirm the measured geometry. |

## Recording a reproduction

Record the code revision, Python/package versions, detector and decision-checkpoint identity, camera settings, board geometry, compensation values, and which interfaces were used. For placement tests, state sample count and the center-coverage or distance criterion. For recognition tests, keep held-out physical data separate and distinguish intersection accuracy from whole-board accuracy. See the [technical notes](technical-notes.en.md) for reported results and remaining milestones.

2026-10-24: pickup/drop, 8 mm hover and clearance waypoints are checked for both endpoints before a transfer. Captures also check the graveyard and attacking piece before moving or removing a piece. Each interpolation segment is still checked separately; this is not whole-action collision certification. Nine offline regressions pass. Calibration is unchanged and the current IK still rejects 11/90 targets, pending measured correction.
