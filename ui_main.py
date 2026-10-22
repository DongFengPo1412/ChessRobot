import sys
import os
import time
import threading
import queue
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QPushButton, QLabel, QGridLayout, QFrame)
from PyQt5.QtCore import pyqtSignal, QObject, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

# 导入你现有的模块
from main import ChessRobotOrchestrator, Coord

# ============================================================
# 1. 信号中转站
# ============================================================
class UiComm(QObject):
    log_signal = pyqtSignal(str)       # 日志信号
    video_signal = pyqtSignal(np.ndarray) # 视频帧信号
    board_signal = pyqtSignal(list)    # 棋盘数据信号
    status_signal = pyqtSignal(str)    # 状态栏信号

# [新增] 支持鼠标点击事件的 QLabel
class ClickableVideoLabel(QLabel):
    clicked_sig = pyqtSignal(int, int)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked_sig.emit(event.x(), event.y())

# 重定向 sys.stdout 到 UI
class LogRedirector:
    def __init__(self, signal):
        self.signal = signal
    def write(self, text):
        if text.strip(): self.signal.emit(str(text))
    def flush(self): pass
    def isatty(self): return False  # [新增] 欺骗底层库这不是真正的控制台
    def fileno(self): return 1      # [新增]

# ============================================================
# 2. UI 主界面
# ============================================================
class ChessRobotUI(QMainWindow):
    def __init__(self, model_path):
        super().__init__()
        self.comm = UiComm()
        self.model_path = model_path
        
        # 初始化 UI
        self.setWindowTitle("UESTC 象棋机器人控制台 v2.0")
        self.resize(1200, 800)
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")
        
        self.init_layout()
        
        # 绑定信号
        self.comm.log_signal.connect(self.update_log)
        self.comm.video_signal.connect(self.update_video)
        self.comm.board_signal.connect(self.draw_board)
        
        # 接管系统日志
        sys.stdout = LogRedirector(self.comm.log_signal)
        
        # 启动后端引擎线程
        self.engine_thread = threading.Thread(target=self.run_engine, daemon=True)
        
        # 强制开启全局按键监听
        self.setFocusPolicy(Qt.StrongFocus)

    def init_layout(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ---- 左侧：实时视频与虚拟棋盘 ----
        left_panel = QVBoxLayout()
        
        # Click mapping uses the actual frame/pixmap dimensions, including letterboxing.
        self.video_label = ClickableVideoLabel("正在初始化摄像头...")
        self.video_label.setFixedSize(640, 360) 
        self.video_label.setAlignment(Qt.AlignCenter)
        self._video_mapping = None
        self.video_label.setStyleSheet("border: 2px solid #555; background: black;")
        self.video_label.clicked_sig.connect(self.on_video_clicked)
        left_panel.addWidget(self.video_label)

        self.board_grid = QGridLayout()
        self.board_grid.setSpacing(2)
        self.cells = [[None for _ in range(9)] for _ in range(10)]
        self.init_board_ui()
        left_panel.addLayout(self.board_grid)
        
        main_layout.addLayout(left_panel, stretch=2)

        # ---- 右侧：控制台与状态监控 ----
        right_panel = QVBoxLayout()
        
        # 状态卡片
        self.status_box = QLabel("系统状态: 等待启动")
        self.status_box.setStyleSheet("font-size: 18px; color: #00ff00; padding: 10px; background: #333; border-radius: 5px;")
        right_panel.addWidget(self.status_box)

        # 按钮区
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始对弈")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setStyleSheet("background-color: #4CAF50; font-weight: bold;")
        self.start_btn.clicked.connect(self.start_logic)
        
        self.stop_btn = QPushButton("🚨 紧急停止")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setStyleSheet("background-color: #f44336; font-weight: bold;")
        self.stop_btn.clicked.connect(self.emergency_stop)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        right_panel.addLayout(btn_layout)

        # 日志台
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Consolas", 10))
        self.console.setStyleSheet("background: #1e1e1e; color: #dcdcdc; border: 1px solid #444;")
        right_panel.addWidget(self.console)
        
        main_layout.addLayout(right_panel, stretch=1)
    

    def on_video_clicked(self, x, y):
        """将 UI 点击坐标还原为物理坐标并投喂给视觉底层"""
        if hasattr(self, 'orchestrator') and self.orchestrator.vision:
            vision = self.orchestrator.vision
            # 如果处于标定阶段 (点数没满 4 个)
            if len(vision.roi_corners) < 4:
                if self._video_mapping is None:
                    return
                left, top, pw, ph, fw, fh = self._video_mapping
                if not (left <= x < left + pw and top <= y < top + ph):
                    return
                orig_x = min(fw - 1, int((x - left) * fw / pw))
                orig_y = min(fh - 1, int((y - top) * fh / ph))
                vision.roi_corners.append([orig_x, orig_y])
                if len(vision.roi_corners) == 4:
                    print("✅ UI 端完成四点标定，已下发至算法层。")
                    
    def init_board_ui(self):
        """创建虚拟棋盘格子"""
        for r in range(10):
            for c in range(9):
                label = QLabel("")
                label.setFixedSize(45, 45)
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("background: #d2b48c; border: 1px solid #8b4513; color: black; font-weight: bold;")
                self.board_grid.addWidget(label, r, c)
                self.cells[r][c] = label

    # ============================================================
    # 3. 核心逻辑连接
    # ============================================================
    def run_engine(self):
        """下棋机器人后端线程"""
        try:
            self.orchestrator = ChessRobotOrchestrator(self.model_path)
            # 劫持视觉输出到 UI 信号
            self.orchestrator.vision.ui_callback = self.comm.video_signal.emit
            
            # 开启循环监听棋盘更新（定时信号）
            def poll_board():
                while not self.orchestrator.stop_event.is_set():
                    self.comm.board_signal.emit(self.orchestrator.board.snapshot())
                    time.sleep(0.5)
            threading.Thread(target=poll_board, daemon=True).start()

            self.orchestrator.run()
        except Exception as e:
            print(f"❌ 引擎线程崩溃: {e}")
        finally:
            if hasattr(self, 'orchestrator'):
                self.orchestrator.shutdown()

    def start_logic(self):
        if not self.engine_thread.is_alive():
            self.engine_thread.start()
            self.status_box.setText("系统状态: 正在运行")
            self.start_btn.setEnabled(False)

    def emergency_stop(self):
        print("🚨 急停信号发出！")
        if hasattr(self, 'orchestrator'):
            self.orchestrator.stop_event.set()
            self.orchestrator.arm.request_stop()
            self.orchestrator.vs.stop()
        self.status_box.setText("系统状态: 已请求停止，请检查实物后重新启动")

    def closeEvent(self, event):
        self.emergency_stop()
        if self.engine_thread.is_alive():
            self.engine_thread.join(timeout=2.0)
        event.accept()

    # ============================================================
    # 4. 界面刷新方法 (由信号触发)
    # ============================================================
    def update_video(self, frame):
        # BGR 转 RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        rect = self.video_label.contentsRect()
        pixmap = QPixmap.fromImage(qt_image).scaled(rect.size(), Qt.KeepAspectRatio)
        self._video_mapping = (rect.x() + (rect.width() - pixmap.width()) // 2,
                               rect.y() + (rect.height() - pixmap.height()) // 2,
                               pixmap.width(), pixmap.height(), w, h)
        self.video_label.setPixmap(pixmap)

    def update_log(self, text):
        self.console.append(text)
        # 自动滚动到底部
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def draw_board(self, board_matrix):
        """实时更新 UI 上的虚拟棋盘"""
        for r in range(10):
            for c in range(9):
                piece = board_matrix[r][c]
                label = self.cells[r][c]
                if piece:
                    side, kind = piece
                    names = {"ju":"车", "ma":"马", "pao":"炮", "xiang":"相" if side == "red" else "象",
                             "shi":"仕" if side == "red" else "士", "jiang":"将", "shuai":"帅", "zu":"卒", "bing":"兵"}
                    label.setText(names.get(kind, kind))
                    color = "red" if side == "red" else "black"
                    label.setStyleSheet(f"background: #f0d5a0; border: 2px solid #8b4513; color: {color}; font-size: 20px; border-radius: 20px;")
                else:
                    label.setText("·")
                    label.setStyleSheet("background: #d2b48c; border: 1px solid #8b4513; color: #8b4513;")

    # ============================================================
    # 5. 全局按键监听 (解决空格无效问题)
    # ============================================================
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if hasattr(self, 'orchestrator'):
                print("🚀 [UI快捷键] 收到空格信号 -> 触发视觉检测")
                self.orchestrator.vision.force_detect_move = True
        elif event.key() == Qt.Key_1:
             if hasattr(self, 'orchestrator'): self.orchestrator.vision.set_black_view(1)
        elif event.key() == Qt.Key_2:
             if hasattr(self, 'orchestrator'): self.orchestrator.vision.set_black_view(2)

if __name__ == "__main__":
    # 获取路径
    B_DIR = os.path.dirname(os.path.abspath(__file__))
    M_PATH = os.path.join(B_DIR, "recognition", "models", "best.pt")

    app = QApplication(sys.argv)
    ui = ChessRobotUI(M_PATH)
    ui.show()
    sys.exit(app.exec_())
