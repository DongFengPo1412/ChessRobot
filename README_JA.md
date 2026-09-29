<div align="center">

# マシンビジョンと深層畳み込みニューラルネットワークに基づく4自由度中国将棋対局ロボットシステム
### YOLOv11・ホモグラフィ幾何補正・Alpha-Beta探索・解析的逆運動学（IK）統合サイバーフィジカルシステム

[ English ](README_EN.md) | [ 简体中文 ](README.md) | [ 日本語 ](README_JA.md)

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B%20(CUDA%20Accelerated)-ee4c2c.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Vision%20Perception%20V11-00a8ff.svg?style=for-the-badge)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B%20Homography-5c3ee8.svg?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Kinematics](https://img.shields.io/badge/Kinematics-4--DOF%20Analytical%20IK-success.svg?style=for-the-badge)](https://github/)
[![Bilibili: Live Demo](https://img.shields.io/badge/Bilibili-Live%20Demo-fb7299?style=for-the-badge&logo=bilibili&logoColor=white)](https://www.bilibili.com/video/BV1aXLn6HEQd)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<p align="center">
  本プロジェクトは、完全自律・ハードリアルタイム閉ループ制御を実現した<b>産業用全スタック中国将棋（シャンチー）人機対局ロボットシステム</b>（コードネーム「智弈·霊手」）です。<br/>
  <b>YOLO 畳み込みニューラルネットワーク目標検出</b>、<b>ホモグラフィ（$3 \times 3$）透視投影幾何補正</b>、<br/>
  <b>Alpha-Beta 枝刈り知的ゲーム推論エンジン</b>、および <b>4自由度解析的逆運動学（IK）制御則</b> を緊密に統合し、<br/>
  カメラによる視覚認識から着手意図検出、局面推論、電磁石による微小空間非衝突ピック＆プレースまでの一貫したサイバーフィジカルループを構築しています。
</p>

</div>

---

## 目次
- [1. プロジェクト背景と開発分担](#1-プロジェクト背景と開発分担)
- [2. 実機デモと全流路ハードウェア検証](#2-実機デモと全流路ハードウェア検証)
  - [2.1 2K 高解像度実機対局ダイナミクス実証](#21-2k-高解像度実機対局ダイナミクス実証)
  - [2.2 サイバーフィジカルシステム（CPS）全スタック構成と操作画面](#22-サイバーフィジカルシステムcps全スタック構成と操作画面)
- [3. アルゴリズム変遷と初期バージョンの歩み](#3-アルゴリズム変遷と初期バージョンの歩み)
- [4. コア数理モデルと運動学制御則](#4-コア数理モデルと運動学制御則)
  - [4.1 ホモグラフィ透視投影幾何補正モデル](#41-ホモグラフィ透視投影幾何補正モデル)
  - [4.2 4自由度多関節アームの解析的逆運動学（IK）解法](#42-4自由度多関節アームの解析的逆運動学ik解法)
  - [4.3 微小空間3段階非衝突アプローチ制御則](#43-微小空間3段階非衝突アプローチ制御則)
  - [4.4 スライディングウィンドウ時系列多数決フィルタ](#44-スライディングウィンドウ時系列多数決フィルタ)
  - [4.5 PCA9685 12-Bit 高精度 PWM パルス幅変換モデル](#45-pca9685-12-bit-高精度-pwm-パルス幅変換モデル)
- [5. ハードウェア電気接続とバス定義](#5-ハードウェア電気接続とバス定義)
- [6. リポジトリ構成](#6-リポジトリ構成)
- [7. クイックスタートと環境構築ガイド](#7-クイックスタートと環境構築ガイド)
- [8. システム操作手順（キャリブレーションと対局）](#8-システム操作手順キャリブレーションと対局)
- [9. ライセンスと謝辞](#9-ライセンスと謝辞)

---

## 1. プロジェクト背景と開発分担

本プロジェクトは、大学『インテリジェント制御アルゴリズム設計 / ロボット制御工学』総合実験（第07班）の重点課題として開発されました。照明変化による駒認識の脆弱性、密集配置された駒の接触転倒リスク、視覚とサーボ間の制御遅延といった課題を解決するため、完全自律の実機対局システムを開発しました。

### チーム開発分担表

| 開発者 | プロジェクト役割 | 担当モジュールと技術実装 |
| :--- | :--- | :--- |
| **田峻碩 (Tian Junshuo)** | **班長 / 機構・モーション制御リード** | <ul><li>4自由度ロボットアームの機構組立および関節調整（RDS3235 35kg·cm 高トルク金属サーボ）</li><li>WCH CH347 高速 USB-I2C ブリッジおよび PCA9685 12ビット PWM ドライバ実装（`hardware.py`）</li><li>幾何学的逆運動学（Analytical IK）の定式化および不感帯フィードフォワード補償（`kinematics.py`）</li><li>微小空間3段階衝突回避軌道計画アルゴリズム（`control.py`）</li></ul> |
| **劉超然 (Liu Chaoran)** | **視覚認識・システム統合リード / 発表代表** | <ul><li>OpenCV $3 \times 3$ ホモグラフィ幾何補正およびミリメートル級物理座標変換（`localization/`）</li><li>YOLO 物体検出モデルのデプロイ、量子化および CUDA 加速推論環境の構築（`yolo11n.pt`）</li><li>時系列スライディングウィンドウ多数決フィルタおよび人の手による遮蔽ノイズ除去アルゴリズム</li><li>非同期マルチスレッド協調アーキテクチャの設計および PyQt5 GUI ダッシュボード開発（`ui_main.py`）</li><li>発表用プレゼンテーション設計、技術報告書の執筆統括および口頭試問対応</li></ul> |
| **張子琛 (Zhang Zichen)** | **ゲーム推論・ルールエンジンリード** | <ul><li>中国将棋公式ルールエンジン、合法手生成器、14チャンネル状態テンソル表現（`game/`）</li><li>Alpha-Beta 枝刈り探索木アルゴリズムおよび局面評価関数の設計</li><li>着手合法性チェック、王手・詰み・ステイルメイト判定ロジックの実装</li></ul> |
| **邢芸龍 (Xing Yilong)** | **治具製作・データセットエンジニアリング** | <ul><li>中国将棋14種類の駒に関する多照度・多角度独自画像データセットの収集・アノテーション</li><li>アクリル製磁気盤の加工、駒用導磁金属プレートの装着および電磁石エンドエフェクタ製作</li><li>電気配線、実験ベンチの組み立ておよび総合結合試験</li></ul> |

---

## 2. 実機デモと全流路ハードウェア検証

以下は、2K 60fps 実機対局検証動画（`电子科技大学_智能控制算法设计“四轴机械臂”_答辩.mp4`）から抽出した動的対局動作とシステム構成です。

* 完全な 2K 60fps 実機人機対局全閉ループ動的デモ動画は Bilibili にて公開されています：  
  👉 **[Bilibili で実機デモ動画を視聴する：4自由度中国将棋対局ロボット端到端閉ループ実戦検証](https://www.bilibili.com/video/BV1aXLn6HEQd)**  
  *(人の着手視覚認識、ホモグラフィ透視変換幾何補正、Alpha-Beta 探索による最適着手解算、4軸解析的逆運動学制御、および電磁石による高精度ピック＆プレースを網羅)*

### 2.1 2K 高解像度実機対局ダイナミクス実証

<div align="center">

| 1. 人の手による着手検出と遮蔽ノイズ除去 | 2. 逆運動学による垂直進入と電磁吸着 | 3. 衝突回避軌道移動・精密着手および待機復帰 |
| :---: | :---: | :---: |
| <img src="docs/images/demo_live_human_move.jpg" width="310px" alt="人の着手認識"/> | <img src="docs/images/demo_live_arm_pickup.jpg" width="310px" alt="ロボットアーム電磁吸着"/> | <img src="docs/images/demo_live_arm_place.jpg" width="310px" alt="ロボットアーム着手"/> |
| 産業用カメラで 30 FPS 常時監視。35フレーム投票により手の影ノイズを完全排除し、赤の移動駒を確実に同定 | AI が最適な手を算出後、解析的逆運動学により 4 軸サーボを駆動して垂直に進入、電磁石で駒を確実に吸着 | 65mm の巡航高度まで垂直上昇後、周囲の駒との接触を完全に回避しながら目標地点へ平移し、垂直着手 |

</div>

### 2.2 サイバーフィジカルシステム（CPS）全スタック構成と操作画面

<div align="center">
  <img src="docs/images/system_architecture.png" width="900px" alt="サイバーフィジカルシステム全スタックアーキテクチャ"/>
  <p><b>図 1：4自由度中国将棋対局ロボットのエンドツーエンド非同期マルチスレッドアーキテクチャ</b></p>
</div>

<div align="center">

| 実機対局ワークステーション全体写真 (`demo_system_cover.png`) | ホモグラフィ透視幾何補正 (`demo_homography_transform.png`) | YOLO 深層学習目標検出 (`demo_yolo_detection.png`) |
| :---: | :---: | :---: |
| <img src="docs/images/demo_system_cover.png" width="300px" alt="実機ベンチ全体"/> | <img src="docs/images/demo_homography_transform.png" width="300px" alt="ホモグラフィ変換"/> | <img src="docs/images/demo_yolo_detection.png" width="300px" alt="YOLO 目標検出"/> |
| 4-DOF 高剛性アルミニウムアーム、俯瞰カメラ、特製導磁アクリル将棋盤 | カメラ傾斜歪みを排除し、画像ピクセルをミリメートル物理座標へ高精度写像 | 赤黒14種類の駒をボード格子上で即時検出。推論遅延は 12ms/フレーム |

</div>

<div align="center">
  <img src="docs/images/demo_ui_console.png" width="880px" alt="PyQt5 ダッシュボードコンソール"/>
  <p><b>図 2：PyQt5 非同期マルチスレッド操作ダッシュボード（リアルタイム映像、仮想盤面、着手ログ、手動校正）</b></p>
</div>

---

## 3. アルゴリズム変遷と初期バージョンの歩み

本プロジェクトは複数の試作を経て成熟したシステムへ進化しました：

1. **第1世代プロトタイプ (`PythonProject1`) — 従来の色調・エッジ抽出**：
   - OpenCV ハフ円変換（Hough Circles）と HSV 色空間の閾値分割を使用。
   - **限界と廃棄理由**：照明変動や影に極めて弱く、アクリルの光沢により中心検出誤差が 5mm 以上生じ、把持ミスが頻発。
2. **第2世代プロトタイプ (`PythonProject2`) — 深層学習の導入と校正試験**：
   - 閾値法を撤廃し YOLO 物体検出へ移行。四隅の透視歪み補正パイプラインを構築。
   - 誤認識は解消したが、制御と認識が同期実行されていたため遅延が発生。
3. **第3世代プロトタイプ (`TianProject`) — ハードウェア駆動と運動学の統合**：
   - WCH CH347 USB-I2C ライブラリを実装し、4軸サーボの逆運動学解析解を導出。
4. **第4世代最終商用アーキテクチャ (`ChessRobot`) — 完全非同期 CPS ループ**：
   - `move_queue` によるスレッド分離、35フレーム多数決抗遮蔽フィルタ、微小空間3段階非衝突制御則、PyQt5 GUI を統合。

---

## 4. コア数理モデルと運動学制御則

### 4.1 ホモグラフィ透視投影幾何補正モデル

カメラの傾斜による歪みを補正するため、$3 \times 3$ 平面ホモグラフィ行列 $\mathbf{H}$ によりピクセル座標 $(u, v)$ を将棋盤物理座標 $(X, Y)$ へ変換：

$$
\begin{bmatrix} X' \\ Y' \\ Z' \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

脱同次化により絶対座標を復元：

$$
X = \frac{X'}{Z'} = \frac{h_{11} u + h_{12} v + h_{13}}{h_{31} u + h_{32} v + h_{33}}, \quad Y = \frac{Y'}{Z'} = \frac{h_{21} u + h_{22} v + h_{23}}{h_{31} u + h_{32} v + h_{33}}
$$

SVD により過決定方程式を解くことで、全盤面での位置誤差を $\le 1.0\,\text{mm}$ に抑制。

---

### 4.2 4自由度多関節アームの解析的逆運動学（IK）解法

アーム構造：ベース旋回（J1）、第1アーム（J2）、第2アーム（J3）、リスト（J4）。

リンク長パラメータ：$L_1 = 105\,\text{mm}$, $L_2 = 105\,\text{mm}$, $L_3 = 98\,\text{mm}$, $L_4 = 160\,\text{mm}$。

エンドエフェクタを常に盤面へ垂直（姿勢角 $\phi = -90^\circ$）に保つ拘束条件下で、目標位置 $(X, Y, Z)$ に対する関節角を解析的に導出：

#### 1. ベース旋回角 $\theta_1$
$$
\theta_1 = \text{atan2}(Y, X)
$$

#### 2. ピッチ平面投影とリスト中心
$$
r = \sqrt{X^2 + Y^2} - L_4 \cos(\phi) = \sqrt{X^2 + Y^2} \quad (\phi = -90^\circ \text{ の場合})
$$

$$
z' = Z - L_1 - L_4 \sin(\phi) = Z - L_1 + L_4
$$

#### 3. 肘関節角 $\theta_3$
余弦定理を適用：

$$
\cos(\theta_3) = \frac{r^2 + z'^2 - L_2^2 - L_3^2}{2 L_2 L_3} \implies \theta_3 = \arccos\left( \frac{r^2 + z'^2 - L_2^2 - L_3^2}{2 L_2 L_3} \right)
$$

#### 4. 肩関節角 $\theta_2$ および手首角 $\theta_4$
$$
\theta_2 = \text{atan2}(z', r) - \text{atan2}(L_3 \sin(\theta_3), \, L_2 + L_3 \cos(\theta_3))
$$

$$
\theta_4 = \phi - (\theta_2 + \theta_3) = -90^\circ - (\theta_2 + \theta_3)
$$

1回の逆運動学計算時間は $0.05\,\text{ms}$ 未満であり、リアルタイム制御ループを確実に満たします。

---

### 4.3 微小空間3段階非衝突アプローチ制御則

将棋の駒半径は約 $18\,\text{mm}$ であり、密集時の間隔は $10\,\text{mm}$ 未満となるため、3段階ウェイポイント制御律を実装：

$$
\mathbf{P}(t) = \begin{cases} 
(X_{\text{src}}, \, Y_{\text{src}}, \, Z_{\text{hover}}), & t \in [0, t_1) \quad (\text{待機巡航高度への進入}) \\
(X_{\text{src}}, \, Y_{\text{src}}, \, Z_{\text{grip}}), & t \in [t_1, t_2) \quad (\text{垂直下降と電磁石吸着}) \\
(X_{\text{src}}, \, Y_{\text{src}}, \, Z_{\text{hover}}), & t \in [t_2, t_3) \quad (\text{巡航高度への垂直引き上げ}) \\
(X_{\text{dst}}, \, Y_{\text{dst}}, \, Z_{\text{hover}}), & t \in [t_3, t_4) \quad (\text{巡航高度での水平目標移動}) \\
(X_{\text{dst}}, \, Y_{\text{dst}}, \, Z_{\text{drop}}), & t \in [t_4, t_5) \quad (\text{目標位置への垂直降下・着手}) \\
(X_{\text{dst}}, \, Y_{\text{dst}}, \, Z_{\text{hover}}), & t \in [t_5, t_6) \quad (\text{垂直離脱と待機復帰})
\end{cases}
$$

安全巡航高度 $Z_{\text{hover}} = 65\,\text{mm}$、吸着高度 $Z_{\text{grip}} = 15\,\text{mm}$ とすることで、駒同士の接触率を **0%** に低減。

---

### 4.4 スライディングウィンドウ時系列多数決フィルタ

人の手による一時的な遮蔽ノイズを除去するため、90個の全交差点に対して $W = 35$ フレーム（約1.1秒）の投票バッファを設定：

$$
S_{\text{consensus}}(i, j) = \arg\max_{c \in \mathcal{C}} \sum_{t=1}^{W} \mathbb{I}(\hat{C}_t(i, j) == c)
$$

同一クラスの得票数が $\text{Votes}(c) \ge 28$ に達した時のみ確定状態とし、1マスの空きと1マスの駒出現が確認された場合に正規の着手イベントとしてキューイングします。

---

### 4.5 PCA9685 12-Bit 高精度 PWM パルス幅変換モデル

PCA9685（12ビット、4096段階）、周波数 $f_{\text{PWM}} = 50\,\text{Hz}$ ($T = 20\,\text{ms}$)：

$$
\text{Tick}(\theta) = \left\lfloor \frac{0.5 + \frac{\theta}{270} \times 2.0}{20} \times 4096 \right\rfloor = \left\lfloor 102.4 + \theta \times 1.517 \right\rfloor
$$

$0.066^\circ$ の角度分解能を実現し、ミリメートル級の先端位置決めを支えています。

---

## 5. ハードウェア電気接続とバス定義

| モジュール | 物理ポート | 通信規約 | 接続先 | 制御パラメータ |
| :--- | :--- | :--- | :--- | :--- |
| **PCA9685 サーボドライバ** | SDA / SCL | ハードウェア I2C (400kHz) | CH347 D0(SCL) / D1(SDA) | 16ch 12ビット PWM、50Hz ($T = 20\,\text{ms}$) |
| **ベース旋回 (J1)** | PWM ch0 | 50Hz PWM | PCA9685 出力0 | RDS3235 金属サーボ、水平方位角 |
| **第1アーム (J2)** | PWM ch1 | 50Hz PWM | PCA9685 出力1 | RDS3235 35kg·cm 高トルクサーボ、リフト |
| **第2アーム (J3)** | PWM ch2 | 50Hz PWM | PCA9685 出力2 | 20kg·cm デジタルサーボ、前後半径 |
| **手首関節 (J4)** | PWM ch3 | 50Hz PWM | PCA9685 出力3 | 180° 金属サーボ、吸盤垂直保持 |
| **電磁石ツール** | MOS 信号 | GPIO High/Low | PCA9685 出力4 / Relay | 5V/12V DC 電磁石、吸引力 $> 5\,\text{N}$ |
| **俯瞰産業カメラ** | USB 2.0 / UVC | DirectShow | PC USB 3.0 | 1280×720 @ 30 FPS、低歪みレンズ |
| **WCH CH347 ブリッジ** | USB Type-C | 480Mbps 高速 USB | PC USB | 高速 USB-I2C/UART/SPI 変換 |

---

## 6. リポジトリ構成

```bash
ChessRobot/
├── localization/                   # ビジョン幾何変換・盤面校正モジュール
│   ├── board_locator.py            # 四隅自動検出・ROI 抽出
│   ├── homography.py               # ホモグラフィ行列演算・座標変換
│   └── calibrate_camera.py         # レンズ歪み補正・キャリブレーション
├── recognition/                    # 深層学習検出・時系列フィルタリング
│   ├── yolo_detector.py            # YOLO 推論ラッパー (CUDA/TensorRT)
│   ├── consensus_filter.py         # 35フレームスライディング多数決フィルタ
│   └── piece_classifier.py         # 駒クラス判別・Softmax 推論
├── game/                           # ゲーム決定木・将棋ルールエンジン
│   ├── chess_engine.py             # 局面管理・合法手生成エンジン
│   ├── search_ai.py                # Alpha-Beta ミニマックス探索アルゴリズム
│   └── board_state.py              # 14チャンネル空間テンソル表現・FEN 相互変換
├── tools/                          # ハードウェア診断・キャリブレーション治具
│   ├── servo_tester.py             # サーボ角度対話型デバッグツール
│   ├── ch347_i2c_scanner.py        # I2C アドレススキャナ
│   └── camera_preview.py           # リアルタイム映像診断ツール
├── control.py                      # 3段階非衝突モーション制御コア
├── kinematics.py                   # 4自由度空間解析的逆運動学ソルバ
├── hardware.py                     # WCH CH347 + PCA9685 低レベル通信ドライバ
├── config.py                       # 物理寸法・サーボオフセット・通信設定
├── main.py                         # CLI 端末対局メインプログラム
├── ui_main.py                      # PyQt5 GUI ダッシュボードメインプログラム
├── yolo11n.pt                      # 学習済み YOLOv11 中国将棋ウェイトファイル
├── requirements.txt                # 依存ライブラリ一覧
├── LICENSE                         # MIT 公式ライセンス証書
├── README.md                       # 簡体中国語技術仕様書
├── README_EN.md                    # 英語技術仕様書
└── README_JA.md                    # 日本語技術仕様書
```

---

## 7. クイックスタートと環境構築ガイド

### 動作要件
- **OS**：Windows 10 / 11 (64-bit) または Ubuntu 20.04 / 22.04 LTS
- **Python**：Python 3.10+ (Anaconda 推奨)
- **GPU**：NVIDIA GPU (リアルタイム 30 FPS 推論用、GTX 1650 以上推奨)
- **ドライバ**：WCH CH347 ドライバ (`CH347DLLA64.DLL`) をルートに同梱済み

### 構築手順
```bash
# 1. リポジトリのクローン
git clone https://github.com/TJS-Git-Hub/ChessRobot.git
cd ChessRobot

# 2. Python 3.10 仮想環境の作成と有効化
conda create -n chess_robot python=3.10 -y
conda activate chess_robot

# 3. 依存ライブラリおよび CUDA 版 PyTorch のインストール
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 8. システム操作手順（キャリブレーションと対局）

1. **盤面四隅のキャリブレーション**：
   - GUI 上の `角点標定` をクリックし、盤面の四隅を時計回りにクリック。ホモグラフィ行列が生成され保存されます。
2. **初期配置**：
   - 32 枚の駒を規定位置に配置し、`対局開始` をクリック。
3. **人間プレイヤーの着手**：
   - 人間（赤番）が駒を動かして盤上から手を引くと、1.1秒以内に移動が検出され、AI の応手計画が開始されます。
4. **ロボットアームの着手実行**：
   - アームが目標駒へ移動、垂直降下して電磁吸着し、巡航高度で目標マスへ移動して正確に着手します。

---

## 9. ライセンスと謝辞

本プロジェクトは **[MIT License](LICENSE)** の下で公開されています。

- **コア開発チーム（第07班）**：
  - **田峻碩 (Tian Junshuo)**：機構設計、ハードウェア低レベルドライバ、逆運動学解析
  - **劉超然 (Liu Chaoran)**：画像認識、ホモグラフィ幾何校正、マルチスレッド CPS パイプライン、PyQt5 GUI
  - **張子琛 (Zhang Zichen)**：中国将棋ルールエンジン、Alpha-Beta 探索アルゴリズム
  - **邢芸龍 (Xing Yilong)**：治具製作、データセット構築、実機結合試験
- **謝辞**：
  - 『インテリジェント制御アルゴリズム設計』担当教官陣の熱心なご指導に感謝申し上げます。

---

<div align="center">
  <b>ChessRobot — 智弈·霊手</b>、マシンビジョンと多軸制御が融合した自律対局ロボット。
</div>
