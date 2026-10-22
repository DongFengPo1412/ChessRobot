from game.constants import INIT_BOARD, X_RANGE, Y_RANGE
import threading


class ChessBoard:
    def __init__(self):
        self._lock = threading.RLock()
        # 初始化棋盘
        self.board = [row.copy() for row in INIT_BOARD]
        # 记录历史走法（用于AI匹配棋谱）
        self.move_history = []
        # 当前回合：红方先手(True=红方走, False=黑方走)
        self.red_turn = True

    def print_board(self):
        """打印带数字坐标的棋盘"""
        print("\n  0 1 2 3 4 5 6 7 8 (X轴)")
        for y in range(Y_RANGE + 1):
            print(f"{y}", end=" ")
            for x in range(X_RANGE + 1):
                piece = self.board[y][x]
                if piece:
                    # 映射棋子类型为中文显示
                    piece_map = {
                        "ju": "车", "ma": "马", "xiang": "象", "shi": "士",
                        "jiang": "将", "zu": "卒", "pao": "炮",
                        "shuai": "帅", "相": "相", "仕": "士", "兵": "兵"
                    }
                    print(piece_map.get(piece[1], piece[1]), end=" ")
                else:
                    print("□", end=" ")
            print(f"{y} (Y轴)")
        print("  0 1 2 3 4 5 6 7 8 (X轴)\n")

    def parse_move(self, move_str: str) -> tuple:
        """解析4位数字走法：原坐标(x1,y1) → 目标坐标(x2,y2)"""
        if len(move_str) != 4 or not move_str.isdigit():
            raise ValueError("走法必须是4位数字！")
        x1 = int(move_str[0])
        y1 = int(move_str[1])
        x2 = int(move_str[2])
        y2 = int(move_str[3])
        return (x1, y1), (x2, y2)

    def is_in_board(self, x: int, y: int) -> bool:
        """判断坐标是否在棋盘内"""
        return 0 <= x <= X_RANGE and 0 <= y <= Y_RANGE

    def is_own_piece(self, x: int, y: int, is_red: bool) -> bool:
        """判断是否是己方棋子"""
        piece = self.board[y][x]
        return piece is not None and piece[0] == ("red" if is_red else "black")

    def is_path_clear(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """检查车/炮的行走路径是否无遮挡"""
        if x1 == x2:  # 竖走
            step = 1 if y2 > y1 else -1
            for y in range(y1 + step, y2, step):
                if self.board[y][x1] is not None:
                    return False
        elif y1 == y2:  # 横走
            step = 1 if x2 > x1 else -1
            for x in range(x1 + step, x2, step):
                if self.board[y1][x] is not None:
                    return False
        return True

    def is_ma_valid(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """马走日校验（别马腿）"""
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        if not ((dx == 2 and dy == 1) or (dx == 1 and dy == 2)):
            return False
        # 别马腿判断
        if dx == 2:
            return self.board[y1][(x1 + x2) // 2] is None
        else:
            return self.board[(y1 + y2) // 2][x1] is None

    def is_xiang_valid(self, x1: int, y1: int, x2: int, y2: int, is_red: bool) -> bool:
        """相/象走田校验（塞象眼+不越河）"""
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        if dx != 2 or dy != 2:
            return False
        # 塞象眼
        if self.board[(y1 + y2) // 2][(x1 + x2) // 2] is not None:
            return False
        # 红相不过河（y≥5），黑象不过河（y≤4）【修正：原逻辑颠倒】
        if is_red:
            return y2 >= 5  # 红相只能在己方半场（y5-y9）
        else:
            return y2 <= 4  # 黑象只能在己方半场（y0-y4）

    def is_shi_valid(self, x1: int, y1: int, x2: int, y2: int, is_red: bool) -> bool:
        """士/仕走斜线校验（不出九宫）"""
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        if dx != 1 or dy != 1:
            return False
        # 九宫范围
        if is_red:
            return 3 <= x2 <= 5 and 7 <= y2 <= 9  # 红帅九宫
        else:
            return 3 <= x2 <= 5 and 0 <= y2 <= 2  # 黑将九宫

    def is_jiang_valid(self, x1: int, y1: int, x2: int, y2: int, is_red: bool) -> bool:
        """将/帅走一步校验（不出九宫）"""
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        if dx + dy != 1:  # 只能走一步（横/竖）
            return False
        # 九宫范围
        if is_red:
            return 3 <= x2 <= 5 and 7 <= y2 <= 9
        else:
            return 3 <= x2 <= 5 and 0 <= y2 <= 2

    def is_zu_valid(self, x1: int, y1: int, x2: int, y2: int, is_red: bool) -> bool:
        """卒/兵走法校验【核心修复：方向完全修正】"""
        dx = abs(x1 - x2)
        dy = y2 - y1  # 位移差：y2-y1 为正=向下，为负=向上

        if is_red:  # 红兵规则
            # 红兵初始位置y6，未过河（y > 4）：只能向上走（dy=-1），不能左右
            if y1 > 4:  # 未过河（y5-y6）
                return dx == 0 and dy == -1  # 向上走1格
            else:  # 已过河（y0-y4）：可向上/左右走1格
                return (dx == 0 and dy == -1) or (dy == 0 and dx == 1)
        else:  # 黑卒规则
            # 黑卒初始位置y3，未过河（y < 5）：只能向下走（dy=1），不能左右
            if y1 < 5:  # 未过河（y0-y4）
                return dx == 0 and dy == 1  # 向下走1格
            else:  # 已过河（y5-y9）：可向下/左右走1格
                return (dx == 0 and dy == 1) or (dy == 0 and dx == 1)

    def is_pao_valid(self, x1: int, y1: int, x2: int, y2: int, is_red: bool) -> bool:
        """炮走法校验（严格规则：必须有炮架才能吃子，无炮架只能走空）"""
        target_piece = self.board[y2][x2]

        # 1. 炮只能走直线（横/竖），斜线无效
        if not (x1 == x2 or y1 == y2):
            return False

        # 2. 路径无遮挡（无炮架）：只能走空位置，不能吃任何棋子
        if self.is_path_clear(x1, y1, x2, y2):
            # 无炮架时，目标位置必须为空（不能有任何棋子）
            return target_piece is None

        # 3. 路径有遮挡（可能有炮架）：必须满足「炮架数=1 + 目标有敌方棋子」才能吃子
        else:
            # 统计路径中的炮架数量（所有非空棋子）
            count = 0
            if x1 == x2:  # 竖线走
                step = 1 if y2 > y1 else -1
                for y in range(y1 + step, y2, step):
                    if self.board[y][x1] is not None:
                        count += 1
            else:  # 横线走
                step = 1 if x2 > x1 else -1
                for x in range(x1 + step, x2, step):
                    if self.board[y1][x] is not None:
                        count += 1

            # 吃子条件：炮架数=1 + 目标有棋子 + 目标是敌方棋子
            return (count == 1
                    and target_piece is not None
                    and not self.is_own_piece(x2, y2, is_red))

    def _is_pseudo_legal_move(self, move_str: str) -> bool:
        """核心：校验走法是否符合象棋规则【修复所有匹配错误】"""
        try:
            (x1, y1), (x2, y2) = self.parse_move(move_str)
        except:
            return False

        # 基础校验：坐标在棋盘内
        if not self.is_in_board(x1, y1) or not self.is_in_board(x2, y2):
            return False

        # 基础校验：原位置有棋子
        piece = self.board[y1][x1]
        if piece is None:
            return False

        # 基础校验：目标位置不是己方棋子
        if self.is_own_piece(x2, y2, self.red_turn):
            return False

        # 基础校验：当前回合匹配棋子颜色
        color, typ = piece
        is_red = (color == "red")
        if is_red != self.red_turn:
            return False

        # 分棋子校验规则【修复：统一用英文类型匹配】
        if typ == "ju":  # 车
            return (x1 == x2 or y1 == y2) and self.is_path_clear(x1, y1, x2, y2)
        elif typ == "ma":  # 马
            return self.is_ma_valid(x1, y1, x2, y2)
        elif typ == "pao":  # 炮
            return self.is_pao_valid(x1, y1, x2, y2, is_red)
        elif typ == "xiang":  # 象/相（统一用xiang，显示时映射）
            return self.is_xiang_valid(x1, y1, x2, y2, is_red)
        elif typ == "shi":  # 士/仕（统一用shi，显示时映射）
            return self.is_shi_valid(x1, y1, x2, y2, is_red)
        elif typ in ["jiang", "shuai"]:  # 将/帅
            return self.is_jiang_valid(x1, y1, x2, y2, is_red)
        elif typ in ["zu", "bing"]:  # 卒/兵
            return self.is_zu_valid(x1, y1, x2, y2, is_red)

        return False

    def is_in_check(self, red):
        side = "red" if red else "black"
        kings = {piece[0]: (x, y) for y, row in enumerate(self.board)
                 for x, piece in enumerate(row) if piece and piece[1] in ("jiang", "shuai")}
        if side not in kings:
            return True
        x, y = kings[side]
        opponent = kings.get("black" if red else "red")
        if opponent and opponent[0] == x and self.is_path_clear(x, y, *opponent):
            return True
        old_turn = self.red_turn
        try:
            self.red_turn = not red
            for ey, row in enumerate(self.board):
                for ex, piece in enumerate(row):
                    if piece and piece[0] != side and self._is_pseudo_legal_move(f"{ex}{ey}{x}{y}"):
                        return True
            return False
        finally:
            self.red_turn = old_turn

    def snapshot(self):
        with self._lock:
            return [row[:] for row in self.board]

    def is_move_valid(self, move_str: str) -> bool:
        with self._lock:
            return self._validate_move(move_str)

    def _validate_move(self, move_str: str) -> bool:
        if not self._is_pseudo_legal_move(move_str):
            return False
        (x1, y1), (x2, y2) = self.parse_move(move_str)
        moving, captured = self.board[y1][x1], self.board[y2][x2]
        self.board[y1][x1], self.board[y2][x2] = None, moving
        try:
            return not self.is_in_check(self.red_turn)
        finally:
            self.board[y1][x1], self.board[y2][x2] = moving, captured

    def make_move(self, move_str: str):
        with self._lock:
            return self._make_move(move_str)

    def _make_move(self, move_str: str):
        """执行合法走法"""
        if not self.is_move_valid(move_str):
            raise ValueError("无效走法！")
        (x1, y1), (x2, y2) = self.parse_move(move_str)
        # 移动棋子
        self.board[y2][x2] = self.board[y1][x1]
        self.board[y1][x1] = None
        # 记录走法 + 切换回合
        self.move_history.append(move_str)
        self.red_turn = not self.red_turn

    def check_game_over(self) -> str:
        """胜负判定：将/帅被吃则游戏结束"""
        has_black_jiang = any(piece and piece[1] == "jiang" for row in self.board for piece in row)
        has_red_shuai = any(piece and piece[1] == "shuai" for row in self.board for piece in row)
        if not has_black_jiang:
            return "红方胜利！"
        if not has_red_shuai:
            return "黑方胜利！"
        return ""

    def get_state_tensor(self):
        """将当前棋盘状态转换为 14x10x9 的特征矩阵 (用于输入神经网络)"""
        import numpy as np
        # 14个通道对应: 黑(将车马炮象士卒) 红(帅车马炮相仕兵)
        channel_map = {
            ("black", "jiang"): 0, ("black", "ju"): 1, ("black", "ma"): 2,
            ("black", "pao"): 3, ("black", "xiang"): 4, ("black", "shi"): 5, ("black", "zu"): 6,
            ("red", "shuai"): 7, ("red", "ju"): 8, ("red", "ma"): 9,
            ("red", "pao"): 10, ("red", "xiang"): 11, ("red", "shi"): 12, ("red", "bing"): 13,
        }

        tensor = np.zeros((14, 10, 9), dtype=np.float32)
        for y in range(10):
            for x in range(9):
                piece = self.board[y][x]
                if piece:
                    # 兼容可能存在的中文命名异常
                    color, typ = piece
                    # 将中文映射回英文(以防万一)
                    type_en_map = {"相": "xiang", "仕": "shi", "兵": "bing"}
                    typ = type_en_map.get(typ, typ)

                    if (color, typ) in channel_map:
                        channel = channel_map[(color, typ)]
                        tensor[channel][y][x] = 1.0
        return tensor

    def get_all_legal_moves(self, is_red):
        """获取指定方所有的合法走法列表"""
        valid_moves = []
        target_color = "red" if is_red else "black"
        for y1 in range(10):
            for x1 in range(9):
                piece = self.board[y1][x1]
                if piece and piece[0] == target_color:
                    for y2 in range(10):
                        for x2 in range(9):
                            move = f"{x1}{y1}{x2}{y2}"
                            # 临时保存回合状态，避免 is_move_valid 内部的回合校验拦截
                            temp_turn = self.red_turn
                            self.red_turn = is_red
                            if self.is_move_valid(move):
                                valid_moves.append(move)
                            self.red_turn = temp_turn
        return valid_moves
