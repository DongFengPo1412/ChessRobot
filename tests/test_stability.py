"""No CH347, camera, model, GUI, or physical movement is opened."""
import ast
import copy
import importlib
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from game.board import ChessBoard


def extract(path, name, scope):
    p = Path(__file__).parents[1] / path
    t = ast.parse(p.read_text(encoding='utf-8'))
    n = next(n for n in t.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    exec(compile(ast.Module(body=[n], type_ignores=[]), str(p), 'exec'), scope)
    return scope[name]


class FakeHardware:
    def __init__(self): self.moves=[]; self.magnets=[]; self.closed=0
    def set_pwm_us(self, channel, pulse): self.moves.append((channel,pulse))
    def set_magnet(self, enabled): self.magnets.append(enabled)
    def close(self): self.closed+=1


class Tests(unittest.TestCase):
    def board(self):
        b=ChessBoard();b.board=[[None]*9 for _ in range(10)]
        b.board[0][4]=('black','jiang');b.board[9][3]=('red','shuai');b.red_turn=True
        return b

    def test_move_cannot_expose_own_general(self):
        b=self.board();b.board[0][3]=('black','ju');b.board[8][3]=('red','ju')
        before=copy.deepcopy(b.board)
        self.assertFalse(b.is_move_valid('3828'))
        self.assertEqual(b.board,before);self.assertTrue(b.red_turn)

    def test_generals_cannot_face_after_move(self):
        b=self.board();b.board[9][3]=None;b.board[9][4]=('red','shuai');b.board[5][4]=('red','ju')
        self.assertFalse(b.is_move_valid('4535'))
        self.assertTrue(b.is_move_valid('4544'))

    def test_initial_moves_remain_legal(self):
        b=ChessBoard();self.assertTrue(b.is_move_valid('0605'))

    def test_motion_preflight_prevents_partial_segment(self):
        import math,config
        solver=types.SimpleNamespace(solve_ik=lambda x,y,z:None if x>220.5 else (0,0,0,0),angle_to_pwm=lambda *a:(1500,)*4)
        scope={'time':types.SimpleNamespace(sleep=lambda _:None),'threading':threading,'math':math,'config':config,'MotionError':RuntimeError}
        cls=extract('control.py','ChessArmController',scope);hw=FakeHardware()
        arm=cls(hardware=hw,solver=solver,auto_home=False)
        before=arm.cur_x,arm.cur_y,arm.cur_z
        with self.assertRaises(RuntimeError):arm.move_to_smooth(222,50,150,.1)
        self.assertEqual(hw.moves,[]);self.assertEqual((arm.cur_x,arm.cur_y,arm.cur_z),before)

    def test_shutdown_never_moves_and_is_idempotent(self):
        import math,config
        scope={'time':types.SimpleNamespace(sleep=lambda _:None),'threading':threading,'math':math,'config':config,'MotionError':RuntimeError}
        cls=extract('control.py','ChessArmController',scope);hw=FakeHardware()
        arm=cls(hardware=hw,solver=object(),auto_home=False);arm.shutdown();arm.shutdown()
        self.assertEqual(hw.closed,1);self.assertEqual(hw.moves,[])
        with self.assertRaises(RuntimeError):arm.execute_move(0,0,0)

    def test_unreachable_destination_rejected_before_pickup(self):
        import math, config
        from kinematics import KinematicsSolver
        scope={'time':types.SimpleNamespace(sleep=lambda _:None),'threading':threading,'math':math,'config':config,'MotionError':RuntimeError}
        cls=extract('control.py','ChessArmController',scope);hw=FakeHardware()
        arm=cls(hardware=hw,solver=KinematicsSolver(),auto_home=False)
        with self.assertRaisesRegex(RuntimeError, 'no pickup started'):
            arm.pick_and_place('a0', 'a9')
        self.assertEqual(hw.moves,[]);self.assertEqual(hw.magnets,[])

    def test_capture_checks_attacker_before_removing_target(self):
        import math, config
        from kinematics import KinematicsSolver
        scope={'time':types.SimpleNamespace(sleep=lambda _:None),'threading':threading,'math':math,'config':config,'MotionError':RuntimeError}
        cls=extract('control.py','ChessArmController',scope);hw=FakeHardware()
        arm=cls(hardware=hw,solver=KinematicsSolver(),auto_home=False)
        with self.assertRaisesRegex(RuntimeError, 'no pickup started'):
            arm.capture('a9', 'a0')
        self.assertEqual(hw.moves,[]);self.assertEqual(hw.magnets,[])

    def test_failed_robot_action_does_not_commit_board(self):
        scope={'threading':threading,'time':types.SimpleNamespace(sleep=lambda _:None),'print':lambda *a:None}
        cls=extract('main.py','ChessRobotOrchestrator',scope);o=cls.__new__(cls)
        o.board=self.board();o.board.red_turn=False;o.board.board[1][0]=('black','ju')
        o.ai=types.SimpleNamespace(get_best_move=lambda:'0102')
        o.arm=types.SimpleNamespace(pick_and_place=lambda *a:(_ for _ in ()).throw(RuntimeError('driver failed')),request_stop=lambda:None)
        o.vision=types.SimpleNamespace(suppress_detection=False,reset_baseline=lambda n:None)
        o.stop_event=threading.Event();o.piece_count=3
        scope['Coord']=extract('main.py','Coord',{})
        before=copy.deepcopy(o.board.board)
        self.assertFalse(o._robot_move());self.assertEqual(o.board.board,before);self.assertTrue(o.stop_event.is_set())

    def test_black_pixel_ratio_uses_pixels_not_channels(self):
        p=Path(__file__).parents[1]/'localization/find_chess_new.py'
        cls=next(n for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='ChessVisionV15_4')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='is_black_piece')
        scope={'np':np,'cv2':cv2};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),scope)
        self.assertTrue(scope['is_black_piece'](None,np.zeros((64,64,3),dtype=np.uint8),(0,0,64,64)))
        self.assertFalse(scope['is_black_piece'](None,np.full((64,64,3),255,dtype=np.uint8),(0,0,64,64)))


if __name__=='__main__':unittest.main()
