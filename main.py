"""
下棋机器人中枢主程序 (main.py)
==============================
架构：视觉感知(独立线程) → move_queue → 逻辑决策(主线程) → 硬件控制(主线程)

坐标系约定（黑方第一视角，机械臂在棋盘正后方）：
  - a 列（索引 0）在最右侧，i 列（索引 8）在最左侧
  - row 0 最近（黑方底线），row 9 最远（红方底线）
  - 视觉输出 "a0" 格式，游戏内部 (x,y) 纯数字，硬件接收 "a0" 格式

数据流：
  VideoStream(异步) → YOLO追踪 → 稳态判定 → move_queue → 游戏逻辑 → AI → 机械臂
"""

import os
import sys
import time
import threading
import queue

import cv2

# ---- 三层模块导入 ----
from localization.find_chess_new import ChessVisionV15_4, VideoStream
from game.board import ChessBoard
from game.ai import ChessAI
from control import ChessArmController


# ============================================================
# 1. 统一坐标适配层
# ============================================================
class Coord:
    """三套坐标体系之间的无损双向转换

    视觉格式  "a0":       列字母(a-i) + 行数字(0-9)
    游戏格式  (x, y):     整数元组, x=0..8 列索引(0=a), y=0..9 行索引
    游戏走法  "x1y1x2y2": 4位纯数字 — board.parse_move() 的输入格式
    硬件格式  "a0":       与视觉相同 — control.board_to_physical() 消费此格式
    """
    COL_CHARS = "abcdefghi"
    _col_to_idx = {c: i for i, c in enumerate(COL_CHARS)}

    @classmethod
    def vision_to_game(cls, coord: str):
        """ "a0" → (0, 0) """
        return (cls._col_to_idx[coord[0].lower()], int(coord[1:]))

    @classmethod
    def game_to_vision(cls, x: int, y: int):
        """ (0, 0) → "a0" """
        if not (0 <= x <= 8 and 0 <= y <= 9):
            raise ValueError(f"坐标越界: ({x}, {y})")
        return f"{cls.COL_CHARS[x]}{y}"

    @classmethod
    def vision_move_to_game_move(cls, from_vis: str, to_vis: str):
        """ ("e6", "e4") → "6646" """
        x1, y1 = cls.vision_to_game(from_vis)
        x2, y2 = cls.vision_to_game(to_vis)
        return f"{x1}{y1}{x2}{y2}"

    @classmethod
    def game_move_to_vision_move(cls, move_str: str):
        """ "6646" → ("e6", "e4") """
        x1, y1, x2, y2 = (int(c) for c in move_str)
        return cls.game_to_vision(x1, y1), cls.game_to_vision(x2, y2)


# ============================================================
# 2. 视觉适配器 — 将独立视觉模块改造成可被主线程调度的组件
# ============================================================
class VisionAdapter(ChessVisionV15_4):
    """继承 V15.4，覆写关键方法以支持外部编排

    改造点：
      1. report_and_compare() — 检测到移动时推送到线程安全队列
      2. reset_baseline()    — 修复原代码的缩进 bug，加入线程锁保护
      3. 新增 suppress_detection 标志 — 机械臂运动期间抑制视觉上报
    """

    def __init__(self, model_path, move_queue: queue.Queue, ready_event: threading.Event):
        super().__init__(model_path)
        self.move_queue = move_queue
        self.ready_event = ready_event
        self._lock = threading.Lock()
        self.suppress_detection = False

    # ── 覆写：移动检测 → 推送到队列 ──────────────────────
    def report_and_compare(self, current_board):
        with self._lock:
            if self.suppress_detection:
                return None, None

            if not self.last_stable_board:
                self.last_stable_board = current_board.copy()
                print("🟢 [视觉] 基准盘面已锁定，等待人类走棋...")
                self.ready_event.set()
                return None, None

            lost = [p for p in self.last_stable_board if p not in current_board]
            gain = [p for p, tid in current_board.items() if
                    p not in self.last_stable_board or self.last_stable_board.get(p) != tid]

            if lost and gain:
                from_pos, to_pos = lost[0], gain[0]
                print(f"\n🚀 [视觉] 检测到移动: {from_pos} → {to_pos}")
                self.last_stable_board = current_board.copy()
                self.move_queue.put((from_pos, to_pos))
                return from_pos, to_pos
            else:
                # 【核心修复】：增加 else 分支，彻底消除静默无反应的假象
                print(f"\n⚠️ [视觉警告] 数量校验通过，但无法解析空间轨迹！(Lost: {lost}, Gain: {gain})")
                print("👉 物理纠正：吃子时产生了 ID 粘连。请将己方棋子拿起，等待 1 秒让系统确认该位置为空，再重新放下并按空格。")
                return None, None

    # ── 覆写：线程安全重置基准 ──────────────────────────
    def reset_baseline(self, new_expected_count: int = None):
        """机械臂动作完成后调用，丢弃机械臂运动期间的干扰帧"""
        with self._lock:
            if new_expected_count is not None:
                self.expected_count = new_expected_count
            self.last_stable_board = {}
            self.board_window.clear()
            self.is_ready = False
            self.ready_event.clear()
            self.smooth_centers.clear()
            self.piece_energy.clear()
            self.piece_history.clear()
        print(f"🔄 [视觉] 基准已重置，预期棋子总数: {self.expected_count}")


# ============================================================
# 3. 中枢调度器
# ============================================================
class ChessRobotOrchestrator:
    """持有三个模块的引用，编排完整的「人类(红) vs AI(黑)」对弈流程

    线程模型:
      - 视觉线程: 摄像头采集 + YOLO 推理 + cv2 显示 (daemon)
      - 主线程:   状态机编排、AI 计算、机械臂控制
    """

    def __init__(self, model_path: str):
        # ---- 硬件层 ----
        self.arm = ChessArmController()

        # ---- 逻辑层 ----
        self.board = ChessBoard()
        self.ai = ChessAI(self.board)

        # ---- 视觉层 (线程间通信原语) ----
        self.move_queue = queue.Queue()
        self.vision_ready = threading.Event()
        self.stop_event = threading.Event()

        self.vs = VideoStream().start()
        time.sleep(1.2)

        self.vision = VisionAdapter(model_path, self.move_queue, self.vision_ready)

        # ---- 运行时状态 ----
        self.piece_count = 32
        self.vision_thread = None

    # ── 视觉线程入口 ───────────────────────────────────────
    def _vision_thread_func(self):
        try:
            self.vision.run(self.vs)
        except Exception as exc:
            print(f"❌ [视觉] 线程异常: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_event.set()

    # ── 人类走棋 (红方) ────────────────────────────────────
    def _wait_for_human_move(self) -> bool:
        print("\n--- 【等待对手走棋 (红方)】 ---")
        
        # 1. 阻塞等待视觉线程推送坐标 (恢复丢失的通信链路)
        try:
            # 采用轮询机制防止主线程死锁，确保能响应系统退出指令
            while not self.stop_event.is_set():
                try:
                    from_vis, to_vis = self.move_queue.get(timeout=0.5)
                    break
                except queue.Empty:
                    continue
            if self.stop_event.is_set():
                return False
        except Exception as e:
            print(f"❌ 获取人类走棋异常: {e}")
            return False

        game_move = Coord.vision_move_to_game_move(from_vis, to_vis)

        # 2. 校验游戏规则合法性
        if not self.board.is_move_valid(game_move):
            print(f"⚠️ 视觉检测到非法走法: {from_vis} → {to_vis}")
            self.vision.reset_baseline(self.piece_count)
            return False

        # 3. 拦截并处理吃子动作
        x2, y2 = int(game_move[2]), int(game_move[3])
        if self.board.board[y2][x2] is not None:
            self.piece_count -= 1
            print(f"⚔️ [系统] 确认人类执行吃子！全局棋子总数同步递减至: {self.piece_count}")

        # 4. 执行状态更新
        self.board.make_move(game_move)
        print(f"✅ 红方走棋: {from_vis} → {to_vis}  (游戏格式: {game_move})")
        return True

    # ── AI 决策 + 机械臂执行 (黑方) ─────────────────────────
    def _robot_move(self) -> bool:
        print("\n--- 【AI 正在思考 (黑方)】 ---")

        ai_move = self.ai.get_best_move()
        if not ai_move:
            print("🏆 黑方无路可走，红方胜利!")
            return False

        from_vis, to_vis = Coord.game_move_to_vision_move(ai_move)
        x1, y1, x2, y2 = (int(c) for c in ai_move)
        is_capture = self.board.board[y2][x2] is not None

        if not self.board.is_move_valid(ai_move):
            print("AI returned an invalid move; stopping before hardware commands")
            self.stop_event.set()
            return False
        action = "吃子" if is_capture else "移位"
        print(f"🤖 AI决策: {ai_move} ({from_vis} → {to_vis}) [{action}]")

        # 抑制视觉上报，防止机械臂运动干扰检测
        self.vision.suppress_detection = True

        try:
            if is_capture:
                self.arm.capture(from_vis, to_vis)
            else:
                self.arm.pick_and_place(from_vis, to_vis)
            self.board.make_move(ai_move)
            if is_capture:
                self.piece_count -= 1
        except Exception as exc:
            print(f"Motion failed: {exc}. Inspect/reconcile the physical board before restarting.")
            self.stop_event.set()
            self.arm.request_stop()
            return False
        finally:
            if not self.stop_event.is_set():
                time.sleep(0.3)
                self.vision.reset_baseline(self.piece_count)
                self.vision.suppress_detection = False

        return True

    # ── 主循环 ────────────────────────────────────────────
    def run(self):
        print("=" * 55)
        print("        中国象棋对弈机器人 v1.0")
        print("   黑方 (AI / 机械臂)  vs  红方 (人类)")
        print("=" * 55)
        print("📍 视觉线程启动后，请在窗口中完成 ROI 标定")
        print("   点击顺序: 1.左上 → 2.右上 → 3.右下 → 4.左下")

        # 视觉线程负责标定 + 追踪 (Windows 上 cv2.imshow 可在非主线程运行)
        self.vision_thread = threading.Thread(
            target=self._vision_thread_func,
            daemon=True,
            name="VisionThread"
        )
        self.vision_thread.start()

        # 阻塞等待视觉完成标定 + 锁定初始棋盘
        print("⏳ 等待视觉锁定初始棋盘...")
        deadline = time.monotonic() + 180
        while not self.vision_ready.wait(timeout=0.1):
            if self.stop_event.is_set() or time.monotonic() >= deadline:
                break
        if self.stop_event.is_set() or not self.vision_ready.is_set():
            print("❌ 视觉初始化超时，请检查摄像头和棋盘")
            self.shutdown()
            return

        print("🎮 对弈开始! 红方(人类)先手。\n")
        self.board.print_board()

        # ── 对弈主循环 ──
        while not self.stop_event.is_set():
            result = self.board.check_game_over()
            if result:
                print(f"\n{'=' * 40}")
                print(f"  🏆 {result}")
                print(f"{'=' * 40}")
                break

            if self.board.red_turn:
                ok = self._wait_for_human_move()
                if not ok:
                    if self.stop_event.is_set():
                        break
                    continue
            else:
                ok = self._robot_move()
                if not ok:
                    break

            self.board.print_board()

        self.shutdown()

    # ── 安全关闭 ──────────────────────────────────────────
    def shutdown(self):
        print("\n🛑 正在关闭系统...")
        self.stop_event.set()

        try:
            self.arm.shutdown()
        except Exception:
            pass

        try:
            self.vs.stop()
        except Exception:
            pass

        cv2.destroyAllWindows()

        if self.vision_thread and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=2.0)

        print("✅ 系统已安全关闭。")


# ============================================================
# 4. 入口
# ============================================================
if __name__ == "__main__":
    MODEL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "recognition", "models", "best.pt"
    )

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("   请将训练好的 best.pt 放到 recognition/models/ 目录下")
        sys.exit(1)

    orchestrator = ChessRobotOrchestrator(MODEL_PATH)
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断 (Ctrl+C)")
    finally:
        orchestrator.shutdown()
