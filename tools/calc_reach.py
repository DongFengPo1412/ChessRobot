"""Check active Cartesian mapping and IK without opening the CH347 device.

This geometric check does not certify servo limits, collisions or pickup success.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from control import ChessArmController
from kinematics import KinematicsSolver


def check_board():
    for column in 'abcdefghi':
        for row in range(10):
            position = f'{column}{row}'
            xyz = ChessArmController.board_to_physical(position)
            yield position, xyz, KinematicsSolver.solve_ik(*xyz)


if __name__ == '__main__':
    unreachable = []
    for position, xyz, angles in check_board():
        print(f'{position}: {xyz}, IK={angles if angles is not None else "UNREACHABLE"}')
        if angles is None:
            unreachable.append(position)
    print(f'Geometric IK rejected {len(unreachable)}/90 positions: {unreachable}')
    print('No hardware opened; command reach is not a physical validation.')
    sys.exit(1 if unreachable else 0)
