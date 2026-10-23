import time
import threading
import math
import config
from hardware import PCA9685Driver
from kinematics import KinematicsSolver


class MotionError(RuntimeError):
    """A command failed; physical board recovery requires inspection."""


class ChessArmController:

    def __init__(self, hardware=None, solver=None, auto_home=True):

        self.col_map = {char: i for i, char in enumerate("abcdefghi")}

        self.hw = hardware if hardware is not None else PCA9685Driver()
        self._stop = threading.Event()
        self._closed = False

        self.solver = solver if solver is not None else KinematicsSolver()



        # 预设位置

        self.WAIT_XYZ = (220, 50, 150)  # 右侧高处待命 (倒U)

        self.GRAVEYARD_XYZ = (220, 120, config.BOARD_Z+20)  # 棋盘外弃子区



        # 当前坐标初始化

        self.cur_x, self.cur_y, self.cur_z = self.WAIT_XYZ

        if auto_home:
            try:
                self.go_to_wait()
            except Exception:
                self.hw.close()
                raise



    @staticmethod
    def board_to_physical(pos_str):

        """

        输入: 'a0' (黑方视角右下角黑车)

        输出: (x, y, z) 物理坐标

        """

        col_char = pos_str[0].lower()

        row = int(pos_str[1:])

        if col_char not in "abcdefghi" or row not in range(10):
            raise ValueError("Board coordinate must be a0..i9")
        col_idx = ord(col_char) - ord('a')



        # 1. X 轴计算 (左右)

        # 机械臂在 e0 后方。黑方视角下：a列在最右侧(正)，i列在最左侧(负)

        x = (4 - col_idx) * config.SQUARE_WIDTH



        # 2. Y 轴计算 (前后)

        # 机械臂在黑方底线 (row 0) 后方，row 越大，距离 y 越远

        y = config.BOARD_ORIGIN_Y



        if row <= 4:

            # 黑方半场 (row: 0, 1, 2, 3, 4)，距离较近

            y += row * config.SQUARE_HEIGHT

        else:

            # 红方半场 (row: 5, 6, 7, 8, 9)，距离较远

            # 必须跨越：黑方4格距离 + 1个楚河汉界 + 红方剩余格数

            y += (4 * config.SQUARE_HEIGHT) + config.RIVER_HEIGHT + ((row - 5) * config.SQUARE_HEIGHT)



        # 3. Z 轴

        z = config.BOARD_Z



        return round(x, 2), round(y, 2), z



    def execute_move(self, x, y, z):
        if self._stop.is_set():
            raise MotionError("Motion stopped; inspect the physical board before restarting")
        res = self.solver.solve_ik(x, y, z)

        if res:

            pwms = self.solver.angle_to_pwm(*res)

            for i, channel in enumerate([0, 4, 8, 12]):

                self.hw.set_pwm_us(channel, pwms[i])

            return True

        return False



 



    def go_to_wait(self):

        """平滑收回到待命位置"""

        print(">>> 移动至右侧待命区...")

        self.move_to_smooth(self.cur_x, self.cur_y, config.SAFE_Z, 0.8)  # 先升起

        self.move_to_smooth(self.WAIT_XYZ[0], self.WAIT_XYZ[1], self.WAIT_XYZ[2], 1.5)



    def move_to_smooth(self, tx, ty, tz, duration=1.2):

        """线性插补规划，确保平滑（增加死区防止浮点漂移）"""

        if not all(math.isfinite(v) for v in (tx, ty, tz, duration)) or duration < 0:
            raise MotionError("Invalid motion target/duration")
        steps = max(1, int(duration * 50))
        origin = self.cur_x, self.cur_y, self.cur_z
        target = tx, ty, tz
        delta = tuple(target[j] - origin[j] if abs(target[j] - origin[j]) > 0.1 else 0.0 for j in range(3))
        path = [tuple(origin[j] + delta[j] * i / steps for j in range(3))
                for i in range(1, steps + 1)]
        # Preserve the small-axis dead zone, then finish at the exact target.
        path[-1] = target
        if any(self.solver.solve_ik(*point) is None for point in path):
            raise MotionError("Unreachable interpolation path; no commands sent for this segment")
        for point in path:
            if not self.execute_move(*point):
                raise MotionError("Motion command failed")
            self.cur_x, self.cur_y, self.cur_z = point
            if self._stop.wait(duration / steps):
                raise MotionError("Motion interrupted")
        return True

    def request_stop(self):
        self._stop.set()

    def _preflight_pick_targets(self, *positions, include_graveyard=False):
        """Reject known unreachable pickup/drop waypoints before touching pieces.

        Interpolated paths are still checked separately before each segment.
        This geometric check does not certify servo limits or collision clearance.
        """
        if self._stop.is_set():
            raise MotionError("Motion stopped; inspect the physical board before restarting")
        targets = []
        for position in positions:
            x, y, z = self.board_to_physical(position)
            targets.extend(((position, (x, y, config.SAFE_Z)),
                            (position, (x, y, config.BOARD_Z + 8.0)),
                            (position, (x, y, z))))
        if include_graveyard:
            x, y, z = self.GRAVEYARD_XYZ
            targets.extend((("graveyard", (x, y, config.SAFE_Z)),
                            ("graveyard", (x, y, z))))
        for name, target in targets:
            if self.solver.solve_ik(*target) is None:
                raise MotionError(f"Unreachable {name} waypoint {target}; no pickup started")

    def pick_and_place(self, start, end, return_to_wait=True):
        """
        常规移位：s->e
        引入【微距垂直阶段】强制小臂优先拉升，防止刮蹭
        """
        self._preflight_pick_targets(start, end)
        s_xyz = self.board_to_physical(start)
        e_xyz = self.board_to_physical(end)
        
        # 定义微距安全区：棋子表面上方 8mm
        MICRO_Z = config.BOARD_Z + 8.0

        # ================= 1. 起点吸取 =================
        # A. 快速移动至起点上方并降至微距悬停点
        self.move_to_smooth(s_xyz[0], s_xyz[1], config.SAFE_Z, 1.2)
        self.move_to_smooth(s_xyz[0], s_xyz[1], MICRO_Z, 0.6)
        time.sleep(0.3) # 停顿：等待手腕姿态绝对竖直
        
        # B. 极慢速垂直切入
        self.move_to_smooth(s_xyz[0], s_xyz[1], s_xyz[2], 0.5)
        self.hw.set_magnet(True)
        time.sleep(0.5)
        
        # C. 【核心要求】：微距强制抬升 (小臂主导)
        # 这一段 8mm 的位移极慢且纯垂直，能确保磁铁离开棋盘前不发生任何角度偏移
        self.move_to_smooth(s_xyz[0], s_xyz[1], MICRO_Z, 0.6)
        time.sleep(0.2) # 稳住姿态
        
        # D. 升至安全高度
        self.move_to_smooth(s_xyz[0], s_xyz[1], config.SAFE_Z, 0.8)

        # ================= 2. 终点放下 =================
        # E. 平移至终点并降至微距悬停点
        self.move_to_smooth(e_xyz[0], e_xyz[1], config.SAFE_Z, 1.2)
        self.move_to_smooth(e_xyz[0], e_xyz[1], MICRO_Z, 0.6)
        time.sleep(0.3) # 停顿：等待姿态回正
        
        # F. 极慢速垂直切入放下
        self.move_to_smooth(e_xyz[0], e_xyz[1], e_xyz[2], 0.5)
        time.sleep(0.1)
        self.hw.set_magnet(False)
        time.sleep(0.6) # 退磁延时
        
        # G. 【核心要求】：微距强制抬升
        self.move_to_smooth(e_xyz[0], e_xyz[1], MICRO_Z, 0.6)
        time.sleep(0.2)
        
        # H. 升至安全高度
        self.move_to_smooth(e_xyz[0], e_xyz[1], config.SAFE_Z, 0.8)

        if return_to_wait:
            self.go_to_wait()

    def capture(self, start, end):
        """吃子动作：全流程应用微距垂直保护"""
        self._preflight_pick_targets(start, end, include_graveyard=True)
        print(f">>> 执行吃子序列: 移除 {end}, 移动 {start}")
        e_xyz = self.board_to_physical(end)
        MICRO_Z = config.BOARD_Z + 8.0

        # 1. 移除对方棋子
        self.move_to_smooth(e_xyz[0], e_xyz[1], config.SAFE_Z, 1.2)
        self.move_to_smooth(e_xyz[0], e_xyz[1], MICRO_Z, 0.6)
        time.sleep(0.3)
        
        self.move_to_smooth(e_xyz[0], e_xyz[1], e_xyz[2], 0.5)
        self.hw.set_magnet(True)
        time.sleep(0.5)
        
        # 强制微抬脱离
        self.move_to_smooth(e_xyz[0], e_xyz[1], MICRO_Z, 0.6)
        time.sleep(0.2)
        self.move_to_smooth(e_xyz[0], e_xyz[1], config.SAFE_Z, 0.8)

        # 2. 丢弃到墓地 (保持现有逻辑)
        self.move_to_smooth(self.GRAVEYARD_XYZ[0], self.GRAVEYARD_XYZ[1], config.SAFE_Z, 1.2)
        self.move_to_smooth(self.GRAVEYARD_XYZ[0], self.GRAVEYARD_XYZ[1], self.GRAVEYARD_XYZ[2], 0.8)
        self.hw.set_magnet(False)
        time.sleep(0.5)
        self.move_to_smooth(self.GRAVEYARD_XYZ[0], self.GRAVEYARD_XYZ[1], config.SAFE_Z, 0.8)

        # 3. 执行己方移位 (内部已包含微抬逻辑)
        self.pick_and_place(start, end, return_to_wait=True)



    def shutdown(self):
        if self._closed:
            return
        self.request_stop()
        self._closed = True
        # Do not start a home motion after a failed/partial action.
        self.hw.close()


if __name__ == "__main__":
    arm = ChessArmController()
    try:
        arm.pick_and_place("a0", "a6")
        #arm.pick_and_place("a0", "a2")
        #arm.pick_and_place("h0","g2")
        # 示例：吃掉 c6 的子并移动 e5 到 c6
        #arm.capture("e5", "c6")
    finally:
        arm.shutdown()
