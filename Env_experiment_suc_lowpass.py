#!/user/bin/env python
# coding: UTF-8

import sys
import numpy as np
import math
import datetime
import time
import termios
from timeout_decorator import timeout, TimeoutError
import pandas as pd
import matplotlib.pyplot as plt

import rospy
import tf2_ros
import tf_conversions
import tf
from crazyswarm.msg import GenericLogData
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool

# 関連モジュールのインポート
from tools.Decorator import run_once
from frames_setup import Frames_setup
from tools.Mathfunction import LowPath_Filter, Mathfunction
from tools.Log import Log_data
from models import quadrotor_with_50cm_cable as model


# 定値制御
class Env_Experiment(Frames_setup):
    # このクラスの初期設定を行う関数
    def __init__(self, Texp, Tsam, num):
        # frames_setup, vel_controller の初期化
        super(Frames_setup, self).__init__()

        self.Tend = Texp
        self.Tsam = Tsam
        self.t = 0

        self.mathfunc = Mathfunction()

        # ! Initialization Lowpass Filter
        self.LowpassP = LowPath_Filter()
        self.LowpassP.Init_LowPass2D(fc=5)
        self.LowpassV = LowPath_Filter()
        self.LowpassV.Init_LowPass2D(fc=5)
        self.LowpassE = LowPath_Filter()
        self.LowpassE.Init_LowPass2D(fc=5)
        self.LowpassL = LowPath_Filter()
        self.LowpassL.Init_LowPass2D(fc=5)
        self.LowpassVl = LowPath_Filter()
        self.LowpassVl.Init_LowPass2D(fc=5)
        self.Lowpassdq = LowPath_Filter()
        self.Lowpassdq.Init_LowPass2D(fc=5)
        self.LowpassVpan = LowPath_Filter()
        self.LowpassVpan.Init_LowPass2D(fc=5)

        self.set_frame()
        self.set_key_input()
        self.set_log_function()
        self.set_paylaod_position_function()
        time.sleep(0.5)
        self.init_state()

        self.log = Log_data(num)

        rospy.Subscriber("/pos_command", Twist, callback=self.pos_callback)
        rospy.Subscriber("/land_command", Bool, callback=self.land_callback)
        self.pos_pub = rospy.Publisher(
            "/cf21/pos_LPF", PoseStamped, queue_size=10)
        self.cf21_pos = PoseStamped()

        rospy.Rate(100)

    def pos_callback(self, msg):
        self.position_command = msg.linear

    def land_callback(self, msg):
        self.land_command = msg.data

    # * set frame of crazyflie and paylad
    def set_frame(self):
        self.world_frame = Frames_setup().world_frame
        self.child_frame = Frames_setup().children_frame[0]
        self.tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(self.tfBuffer)
        time.sleep(0.5)

    # * keybord input function
    def set_key_input(self):
        self.fd = sys.stdin.fileno()

        self.old = termios.tcgetattr(self.fd)
        self.new = termios.tcgetattr(self.fd)

        self.new[3] &= ~termios.ICANON
        self.new[3] &= ~termios.ECHO

    def set_log_function(self):
        self.cmd_sub = rospy.Subscriber(
            "/cf20/log1", GenericLogData, self.log_callback)

    def set_paylaod_position_function(self):
        self.cmd_sub2 = rospy.Subscriber(
            "/payload_pose",
            PoseStamped,
            self.payload_pos_callback,
        )

    def init_state(self):
        self.P = np.zeros(3)
        self.Ppre = np.zeros(3)
        self.Vrow = np.zeros(3)
        self.V = np.zeros(3)
        self.Vpre = np.zeros(3)
        self.A = np.zeros(3)
        self.R = np.zeros((3, 3))
        self.Euler = np.zeros(3)

        # 方向ベクトルdpenの定義（毎回変える）（多分行ベクトル）
        # self.dpen = np.array([-0.00656295,-0.01319998,0.0189808])
        # self.dpen = np.array([-0.00636,-0.01328,0.01643])
        self.dpen = np.array([-0.00368491, -0.01511539, 0.022109])
        # 振子の根元から振子の重心に向かう方向ベクトル追加
        self.d = np.zeros(3)
        # 振子の方向ベクトルの大きさを追加
        self.dsize = 0
        # 振子の方向ベクトルを正規化したものを追加
        self.dunit = np.zeros(3)
        # 振子の根本位置座標追加（ドローン重心に方向ベクトルdpen足したもの）
        self.Pba = np.zeros(3)
        self.Pbapre = np.zeros(3)
        # 振子の角度追加(一行目にθ、二行目にΦ)
        self.Pan = np.zeros(2)
        self.Panpre = np.zeros(2)
        # 振子の角速度(フィルター前)
        self.Vpanrow = np.zeros(3)
        # 振子の角速度追加（一行目にdθ、二行目にdΦ）
        self.Vpan = np.zeros(3)
        self.Vpanpre = np.zeros(3)
        # 振子の重心位置
        self.Pl = np.zeros(3)
        self.Plrow = np.zeros(3)
        self.Plpre = np.zeros(3)
        # 振子の重心速度
        self.Vl = np.zeros(3)
        self.Vrow_pre = np.zeros(3)
        self.Vl_filterd = np.zeros(3)

        self.Height_drone = np.array([0.0, 0.0, 0.00])
        self.q = np.zeros(3)
        self.qpre = np.zeros(3)
        self.dqrow = np.zeros(3)
        self.dqrowpre = np.zeros(3)
        self.dq_filtered = np.zeros(3)

        self.M = np.array([0.0, 0.0, 0.0, 0.0])

        try:
            quad = self.tfBuffer.lookup_transform(
                self.world_frame, self.child_frame, rospy.Time(0)
            )

        # 取得できなかった場合は0.5秒間処理を停止し処理を再開する
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            rospy.logerr("LookupTransform Error !")
            rospy.sleep(0.5)
            exit()

        self.P[0] = quad.transform.translation.x
        self.P[1] = quad.transform.translation.y
        self.P[2] = quad.transform.translation.z
        self.Quaternion = (
            quad.transform.rotation.x,
            quad.transform.rotation.y,
            quad.transform.rotation.z,
            quad.transform.rotation.w,
        )
        self.Euler = tf_conversions.transformations.euler_from_quaternion(
            self.Quaternion
        )
        self.Eulerpre = self.Euler
        self.R = self.mathfunc.Euler2Rot(self.Euler)

        self.Plrow[0] = self.load.pose.position.x
        self.Plrow[1] = self.load.pose.position.y
        self.Plrow[2] = self.load.pose.position.z
        self.Pl[0] = self.load.pose.position.x
        self.Pl[1] = self.load.pose.position.y
        self.Pl[2] = self.load.pose.position.z

    # ------------------------------- ここまで　初期化関数 ---------------------

    def update_state(self):
        end_flag = False
        try:
            quad = self.tfBuffer.lookup_transform(
                self.world_frame, self.child_frame, rospy.Time(0)
            )

        # 取得できなかった場合は0.5秒間処理を停止し処理を再開する
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            rospy.logerr("LookupTransform Error !")
            end_flag = True

        # ! state of quadrotor
        # position
        self.P[0] = quad.transform.translation.x
        self.P[1] = quad.transform.translation.y
        self.P[2] = quad.transform.translation.z
        # velocity
        self.Vrow = self.mathfunc.deriv(self.P, self.Ppre, self.dt)
        self.V = self.LowpassV.LowPass2D(self.Vrow, self.Tsam)
        self.A = self.mathfunc.deriv(self.V, self.Vpre, self.dt)
        # attitude
        self.Quaternion = (
            quad.transform.rotation.x,
            quad.transform.rotation.y,
            quad.transform.rotation.z,
            quad.transform.rotation.w,
        )
        self.Euler = self.LowpassE.LowPass2D(
            tf_conversions.transformations.euler_from_quaternion(
                self.Quaternion),
            self.Tsam,
        )
        self.R = self.mathfunc.Euler2Rot(self.Euler)
        # previous states update
        self.Ppre[0] = self.P[0]
        self.Ppre[1] = self.P[1]
        self.Ppre[2] = self.P[2]
        self.Eulerpre = self.Euler

        # ! state of payload
        # 振子の重心位置
        self.Pl[0] = self.load.pose.position.x
        self.Pl[1] = self.load.pose.position.y
        self.Pl[2] = self.load.pose.position.z
        # position
        self.Vlrow = self.mathfunc.deriv(self.Pl, self.Plpre, self.dt)
        self.Vl_filterd = self.LowpassVl.LowPass2D(self.Vlrow, self.Tsam)

        # 振子の根本位置（ドローンの重心 + dpen)(エラー出たら個別で足す)
        self.Pba = self.P + self.dpen
        # 振子の根元から重心への方向ベクトルの作成
        self.d = self.Pl - self.Pba
        # 方向ベクトルを正規化
        self.dsize = np.sqrt(self.d[0]*self.d[0] +
                             self.d[1]*self.d[1]+self.d[2]*self.d[2])
        self.dunit[0] = self.d[0]/self.dsize
        self.dunit[1] = self.d[1]/self.dsize
        self.dunit[2] = self.d[2]/self.dsize
        # 振子の角度追加（[0]がθ、[1]がΦ)
        self.Pan[0] = math.asin(self.dunit[0])
        self.Pan[1] = -math.asin(self.dunit[1]/math.cos(self.Pan[0]))
        # 振子の角速度(フィルター前)更新([0]がdθ、[1]がdΦ)
        self.Vpanrow[0] = self.mathfunc.deriv(
            self.Pan[0], self.Panpre[0], self.dt)
        self.Vpanrow[1] = self.mathfunc.deriv(
            self.Pan[1], self.Panpre[1], self.dt)
        # 振子の角速度更新([0]がdθ、[1]がdΦ)
        # self.Vpan[0] = self.LowpassVpan.LowPass2D(self.Vpanrow[0],self.Tsam)
        # self.Vpan[1] = self.LowpassVpan.LowPass2D(self.Vpanrow[1],self.Tsam)
        self.Vpan = self.LowpassVpan.LowPass2D(self.Vpanrow, self.Tsam)
        # vector and vector velocity
        self.q = (self.Pl - (self.P - self.Height_drone)) / np.linalg.norm(
            (self.Pl - (self.P - self.Height_drone))
        )
        self.dqrow = self.mathfunc.deriv(self.q, self.qpre, self.dt)
        self.dq_filtered = self.Lowpassdq.LowPass2D(self.dqrow, self.Tsam)

        # previous state update
        self.Plpre[0] = self.Pl[0]
        self.Plpre[1] = self.Pl[1]
        self.Plpre[2] = self.Pl[2]
        self.qpre[0] = self.q[0]
        self.qpre[1] = self.q[1]
        self.qpre[2] = self.q[2]
        self.Vpre[0] = self.V[0]
        self.Vpre[1] = self.V[1]
        self.Vpre[2] = self.V[2]
        # 振子の角度追加
        self.Panpre[0] = self.Pan[0]
        self.Panpre[1] = self.Pan[1]

        return end_flag

    def set_dt(self, dt):
        self.dt = dt

    def set_clock(self, t):
        self.t = t

    def log_callback(self, log):
        self.M = log.values

    def payload_pos_callback(self, msg):
        self.load = msg

    def set_reference(
        self,
        controller,
        P=np.array([0.0, 0.0, 0.0]),
        V=np.array([0.0, 0.0, 0.0]),
        R=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        Euler=np.array([0.0, 0.0, 0.0]),
        Wb=np.array([0.0, 0.0, 0.0]),
        Euler_rate=np.array([0.0, 0.0, 0.0]),
        traj="land",
        controller_type="mellinger",
        init_controller=True,
        tmp_P=np.zeros(3),
    ):
        if init_controller:
            controller.select_controller()
        if controller_type == "pid":
            controller.set_reference(
                P, V, R, Euler, Wb, Euler_rate, controller_type)
        elif controller_type == "mellinger" or "QCSL" or "DI":
            controller.set_reference(traj, self.t, tmp_P)

    def take_log(self, ctrl):
        self.log.write_state(
            self.t,
            self.P,
            self.V,
            self.R,
            self.Euler,
            np.zeros(3),
            np.zeros(3),
            self.M,
            self.Plrow,
            self.Pl,
            self.Vl_filterd,
            self.q,
            self.dq_filtered,
            self.Pan,
            self.Vpan,
            self.Pba,
        )
        ctrl.log(self.log, self.t)

    def save_log(self):
        self.log.close_file()

    @timeout(0.01)
    def input_with_timeout(self, msg=None):
        termios.tcsetattr(self.fd, termios.TCSANOW, self.new)
        return sys.stdin.read(1)

    def get_input(self):
        quit_flag = False
        try:
            input = self.input_with_timeout("key:")
            if input == "q":
                quit_flag = True
        except TimeoutError:
            quit_flag = False

        return quit_flag

    def time_check(self, Tint, Tend):
        if Tint < self.Tsam:
            time.sleep(self.Tsam - Tint)
        if self.t > Tend:
            return True
        return False

    # takeoff and land command
    def quad_takeoff(
        self, controller, controller_type="mellinger", Pinit=np.array([0.0, 0.0, 0.0])
    ):
        controller.switch_controller(controller_type)
        self.set_reference(
            controller=controller,
            traj="takeoff",
            controller_type=controller_type,
            tmp_P=Pinit,
            init_controller=False,
        )
        self.land_P = np.array([0.0, 0.0, 0.0])

    def quad_takeoff_from_125cm(self, controller, Pinit=np.array([0.0, 0.0, 0.0])):
        self.set_reference(
            controller=controller,
            traj="takeoff_from_125cm",
            controller_type="mellinger",
            tmp_P=Pinit,
        )
        self.land_P = np.array([0.0, 0.0, 0.0])

    def quad_land(self, controller, controller_type="mellinger"):
        controller.switch_controller(controller_type)
        self.set_reference(
            controller=controller,
            traj="land",
            controller_type=controller_type,
            tmp_P=np.array([self.P[0], self.P[1], 0.0]),
            init_controller=False,
        )

    def quad_land_at_125cm(self, controller, controller_type="mellinger"):
        controller.switch_controller(controller_type)
        self.set_reference(
            controller=controller,
            traj="land_at_125cm",
            controller_type="mellinger",
            tmp_P=np.array([self.P[0], self.P[1], 0.0]),
        )

    def quad_land_from_50cm(self, controller, controller_type="mellinger"):
        controller.switch_controller(controller_type)
        self.set_reference(
            controller=controller,
            traj="land_50cm",
            controller_type="mellinger",
            init_controller=False,
            tmp_P=np.array([self.P[0], self.P[1], 0.0]),
        )

    # quadrotor trajectory tracking command
    def quad_tack_circle(self, controller, controller_type="mellinger"):
        controller.switch_controller(controller_type)
        self.set_reference(
            controller=controller,
            traj="circle",
            controller_type=controller_type,
            init_controller=False,
        )

    def quad_track_move_to_zero(
        self, controller, controller_type="mellinger", flag=False
    ):
        controller.switch_controller(controller_type)
        self.set_reference(
            controller=controller,
            traj="move_to_1.0_from_0.0",
            controller_type=controller_type,
            init_controller=flag,
        )

    def quad_stop_track(self, controller, controller_type="mellinger"):
        controller.switch_controller(controller_type)
        self.set_reference(
            controller=controller,
            traj="stop",
            controller_type=controller_type,
            init_controller=False,
            tmp_P=np.array([self.P[0], self.P[1], self.P[2] + 0.2]),
        )
        self.land_P[0:2] = self.P[0:2]

    # payload trajectory tracking command
    def payload_track_circle(self, controller, controller_type="QCSL", flag=False):
        controller.switch_controller(controller_type)
        self.set_reference(
            controller=controller,
            traj="circle",
            controller_type=controller_type,
            init_controller=flag,
        )

    def payload_track_straight(self, controller, controller_type="QCSL", flag=False):
        controller.switch_controller(controller_type)
        self.set_reference(
            controller=controller,
            traj="straight",
            controller_type=controller_type,
            init_controller=flag,
        )

    def payload_track_circle_from_center(
        self, controller, controller_type="QCSL", flag=False
    ):
        controller.switch_controller(controller_type)
        self.set_reference(
            controller=controller,
            traj="circle_from_center",
            controller_type=controller_type,
            init_controller=flag,
        )

    def payload_stop_track(self, controller, controller_type="QCSL"):
        controller.switch_controller(controller_type)
        self.set_reference(
            controller=controller,
            traj="stop",
            controller_type=controller_type,
            init_controller=False,
            tmp_P=np.array([self.Pl[0], self.Pl[1], self.Pl[2]]),
        )
        self.land_P[0:2] = self.P[0:2]

    def paylaod_track_hover_payload(
        self, controller, controller_type="QCSL", flag=False
    ):
        controller.switch_controller(controller_type)
        self.set_reference(
            controller=controller,
            traj="hover",
            controller_type=controller_type,
            init_controller=flag,
        )

    def payload_takeoff(
        self, controller, controller_type="QCSL", Pinit=np.array([0.0, 0.0, 0.0])
    ):
        controller.switch_controller(controller_type)
        self.set_reference(
            controller=controller,
            traj="takeoff",
            controller_type=controller_type,
            tmp_P=Pinit,
        )
        self.land_P = np.array([0.0, 0.0, 0.1])

    def payload_land(self, controller, controller_type="QCSL"):
        controller.switch_controller(controller_type)
        self.set_reference(
            controller=controller,
            traj="land",
            controller_type=controller_type,
            init_controller=False,
            tmp_P=np.array([self.P[0], self.P[1], 0.0]),
        )

    # Env_experiment3.pyから引用
    @run_once
    def land(self, controller):
        controller.switch_controller("pid")
        self.set_reference(
            controller=controller, command="land", init_controller=True, P=self.land_P
        )

    @run_once
    def hovering(self, controller, P, Yaw=0.0):
        self.set_reference(
            controller=controller,
            command="hovering",
            P=P,
            Euler=np.array([0.0, 0.0, Yaw]),
        )
        self.land_P = np.array([0.0, 0.0, 0.1])

    def takeoff(self, controller, Pinit=np.array([0.0, 0.0, 0.0])):
        self.set_reference(
            controller=controller,
            traj="takeoff",
            controller_type="mellinger",
            tmp_P=Pinit,
        )
        self.land_P = np.array([0.0, 0.0, 0.1])

    def land_track(self, controller):
        self.set_reference(
            controller=controller,
            traj="land",
            controller_type="mellinger",
            init_controller=False,
            tmp_P=np.array([self.P[0], self.P[1], 0.0]),
        )

    def takeoff_50cm(self, controller, Pinit=np.array([0.0, 0.0, 0.0])):
        self.set_reference(
            controller=controller,
            traj="takeoff_50cm",
            controller_type="mellinger",
            tmp_P=Pinit,
            init_controller=True,
        )
        self.land_P = np.array([0.0, 0.0, 0.1])

    def land_track_50cm(self, controller):
        self.set_reference(
            controller=controller,
            traj="land_50cm",
            controller_type="mellinger",
            init_controller=False,
            tmp_P=np.array([0.0, 0.0, 0.0]),
        )

    # 倒立振子用の定義
    def pendulum(self, controller, flag=False):
        self.set_reference(
            controller=controller,
            traj="pendulum",
            controller_type="mellinger",
            tmp_P=np.array([1.0, 0.0, 0.0]),
            init_controller=flag,
        )

    # 倒立振子終了の定義
    def espendulum(self, controller, flag=False):
        self.set_reference(
            controller=controller,
            traj="espendulum",
            controller_type="mellinger",
            tmp_P=np.array([2.0, 0.0, 0.0]),
            init_controller=flag,
        )

    # zの高さ変更の定義
    def translate(self, controller, flag=False):
        self.set_reference(
            controller=controller,
            traj="translate",
            controller_type="mellinger",
            tmp_P=np.array([2.0, 0.0, 0.0]),
            init_controller=flag,
        )

    def track_straight(self, controller, flag):
        self.set_reference(
            controller=controller,
            traj="straight",
            controller_type="mellinger",
            init_controller=flag,
        )

    def track_circle(self, controller, flag=False):
        self.set_reference(
            controller=controller,
            traj="circle",
            controller_type="mellinger",
            init_controller=flag,
        )

    def stop_track(self, controller):
        self.set_reference(
            controller=controller,
            traj="stop",
            controller_type="mellinger",
            init_controller=False,
            tmp_P=np.array([self.P[0], self.P[1], self.P[2]]),
        )
        self.land_P[0:2] = self.P[0:2]
