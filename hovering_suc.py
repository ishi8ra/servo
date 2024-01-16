#!/user/bin/env python
# coding: UTF-8

import sys
import time
import numpy as np
import pygame
from pygame.locals import *
import readchar
import fcntl
import termios
import os

# from ros_ws.src.crazyswarm.scripts.pycrazyswarm.crazyswarm import Crazyswarm
from pycrazyswarm import *

from Env_experiment_suc_lowpass import Env_Experiment
# from Env_experiment_directEuler import Env_Experiment
from Exp_Controller.Controllers import Controllers


def getkey():
    fno = sys.stdin.fileno()

    # stdinの端末属性を取得
    attr_old = termios.tcgetattr(fno)

    # stdinのエコー無効、カノニカルモード無効
    attr = termios.tcgetattr(fno)
    attr[3] = attr[3] & ~termios.ECHO & ~termios.ICANON  # & ~termios.ISIG
    termios.tcsetattr(fno, termios.TCSADRAIN, attr)

    # stdinをNONBLOCKに設定
    fcntl_old = fcntl.fcntl(fno, fcntl.F_GETFL)
    fcntl.fcntl(fno, fcntl.F_SETFL, fcntl_old | os.O_NONBLOCK)

    chr = 0

    try:
        # キーを取得
        c = sys.stdin.read(1)
        if len(c):
            while len(c):
                chr = (chr << 8) + ord(c)
                c = sys.stdin.read(1)
    finally:
        # stdinを元に戻す
        fcntl.fcntl(fno, fcntl.F_SETFL, fcntl_old)
        termios.tcsetattr(fno, termios.TCSANOW, attr_old)

    return chr


def Experiment(Texp, Tsam):
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    Env = Env_Experiment(Texp, Tsam, 0)
    Tsam_lock = Tsam

    zero = np.zeros(3)

    Drone_env = Env_Experiment(Texp, Tsam, 0)
    Drone_ctrl = Controllers(Tsam, "mellinger")

    # Drone_env.quad_takeoff(Drone_ctrl, Pinit=np.array([-1.0, 0.0, 0.0]))
    Drone_env.takeoff_50cm(Drone_ctrl)
    timeHelper.sleep(2)

    exp_flag = True
    set_flag = True
    stop_flag = True
    land_flag = True
    emergency_flag = False
    pendulum_flag = True
    espendulum_flag = True
    translate_flag = True
    runonce = True

    takeoff_time = 5
    Texp = Texp + takeoff_time
    stop_time = Texp + 30
    land_time = stop_time + 2

    Ts = timeHelper.time()
    Te = -Tsam
    t = 0

    while True:
        Env.set_clock(t)
        Drone_env.set_clock(t)
        Drone_env.take_log(Drone_ctrl)

        # flagの考え方があっているなら、takeoff_time<tだけでもいいはず
        # if takeoff_time < t < Texp:
        #     if translate_flag:
        #         Drone_env.translate(Drone_ctrl, False)
        #         translate_flag = False

        key = getkey()
        if key == ord('s'):
            if pendulum_flag:
                Drone_env.pendulum(Drone_ctrl, False)
                pendulum_flag = False

        if key == ord('e'):
            if espendulum_flag:
                Drone_env.espendulum(Drone_ctrl, False)
                espendulum_flag = False

        if t > stop_time:
            if land_flag:
                Drone_env.land_track_50cm(Drone_ctrl)
                land_flag = False

        # if Drone_env.P[2] > 2.0 or Drone_env.P[0] < -1.7 or Drone_env.P[0] > 1.7:
        #   emergency_flag = True
        # runonceはEnv_experimentのモード定義の所にある（関係ない）
        #  mellinger内で、元の制御入力書き換えてるから消しても関係ないはず
        if emergency_flag and runonce:
            Drone_env.quad_land(Drone_ctrl, controller_type="mellinger")
            land_flag = False
            stop_flag = False
            exp_flag = False

            runonce = False

        # if stop_time < t:
        #     if land_flag:
        #         Drone_env.quad_land(Drone_ctrl)
        #         land_flag = False

        Drone_ctrl.set_state(Drone_env)
        Drone_ctrl.get_output(t)

        input_acc = Drone_ctrl.input_acc
        input_W = np.array(
            [
                Drone_ctrl.input_Wb[0] * 1,
                Drone_ctrl.input_Wb[1] * 1,
                Drone_ctrl.input_Wb[2],
            ]
        )
        cf.cmdFullState(
            zero, zero, np.array([0.0, 0.0, input_acc / 100.0]), 0.0, input_W
        )

        # Te = t
        # t = timeHelper.time() - Ts
        # end_flag = Env.time_check(t - Te, land_time)
        # if end_flag:
        #     cf.cmdFullState(zero, zero, zero, 0.0, zero)
        #     break

        # Drone_env.set_dt(dt=Tsam_lock)
        # if Drone_env.update_state():
        #     cf.cmdFullState(zero, zero, zero, 0.0, zero)
        #     exit()

        t = timeHelper.time() - Ts  # ループ周期を一定に維持
        Tsam = t - Te
        Te = t
        if Env.time_check(t - Te, land_time):
            break

        Drone_env.set_dt(dt=Tsam_lock)
        Drone_env.update_state()  # 状態を更新

    cf.cmdFullState(zero, zero, zero, 0.0, zero)
    exit()


if __name__ == "__main__":
    Experiment(3, 0.02)
    # Experiment(3, 0.005)
