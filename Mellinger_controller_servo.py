"""
This controller designed to track trajectory for multirotor UAV
references
  - Geometric Tracking Control of a Quadrotor UAV on SE(3)
  - Minimum Snap Trajectory Generation and Control for Quadrotors
  - Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag for Accurate Tracking of High-Speed Trajectories
"""

from tools.Mathfunction import Mathfunction
from Exp_Controller.Trajectory import Trajectory
from scipy import linalg
import math
import numpy as np
import sys

sys.path.append("../")


# 初期設定
# r = (2/3)*0.3  # r(λ)=2/3Lp 長:0.3m
# r = (2/3)*0.2
r = (2/3)*1
g = 9.8  # 重力加速度

A1 = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, g, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, -g, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

Bl1 = np.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 1],
    ]
)

# 重みの決定
# Q1 = np.diag([15, 1, 1, 15, 1, 1, 20, 1])
Q1 = np.diag([15, 1, 50, 15, 1, 50, 80, 1])
R1 = np.diag([50, 50, 1])


# lqr法
def lqr1(A, B, Q, R):
    P1 = linalg.solve_continuous_are(A, B, Q, R)
    K1 = linalg.inv(R).dot(B.T).dot(P1)
    E1 = linalg.eigvals(A - B.dot(K1))

    return P1, K1, E1


P1, K1, E1 = lqr1(A1, Bl1, Q1, R1)

A2 = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, g, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, g/r, 0, -g/r, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, -g, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, g/r, 0, -g/r, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

Bl2 = np.array([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 1]])

C = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# servo系への拡大版
At = np.array([
    [A2, np.zeros((12, 1))],
    [C]
])
Bt = np.array([
    [Bl2],
    [np.zeros(3)]
])

# 重みの決定
# Q2 = np.diag([15, 1, 1, 1, 1, 15, 1, 1, 1, 1, 20, 1])
# matsuda
Q2 = np.diag([15, 1, 1, 1, 50, 15, 1, 1, 1, 50, 80, 1])
R2 = np.diag([50, 50, 1])
# servo拡大系での重み
Qt = np.diag([1, 1, 1, 100, 1, 1, 50, 1, 100, 1, 1, 1, 1000])
Rt = np.diag([1, 1, 1])

# tadokoro
# Q2 = np.diag([10, 1, 1, 1, 10, 10, 1, 1, 1, 10, 1, 1])
# R2 = np.diag([1, 1, 0.3])

# lqr法


def lqr2(A, B, Q, R):
    P2 = linalg.solve_continuous_are(A, B, Q, R)
    K2 = linalg.inv(R).dot(B.T).dot(P2)
    E2 = linalg.eigvals(A - B.dot(K2))

    return P2, K2, E2


P2, K2, E2 = lqr2(A2, Bl2, Q2, R2)
Pt, Kt, Et = lqr2(At, Bt, Qt, Rt)

# サーボ系での目標値
r0 = 0.5


class Mellinger(Mathfunction):
    def __init__(self, dt):
        self.dt = dt

    def set_dt(self, dt):
        self.dt = dt

    def mellinger_init(self):
        print("Init Mellinger Controller")

        # init trajectory
        self.q_ref = np.array([[0], [0], [0], [0], [0], [0], [0], [0]])
        self.A = A1
        self.Bl = Bl1
        self.K = K1

        self.kp = np.array([4.3, 4.3, 3.0])
        self.kv = np.array([3.8, 3.8, 2.5])
        self.ki = np.array([0.0, 0.0, 0.0])
        self.ka = np.array([0.0, 0.0, 0.0])
        self.kR = np.array([8.5, 8.5, 0.5])
        # * initialize gain
        self.kp = np.array([4.3, 4.3, 4.0])
        self.kv = np.array([3.8, 3.8, 2.5])
        self.ka = np.array([0.0, 0.0, 0.0])
        self.ki = np.array([0.0, 0.0, 0.0])
        self.kR = np.array([7.0, 7.0, 0.7])

        # * initialize nominal state values
        self.Euler_nom = np.array([0.0, 0.0, 0.0])
        self.Euler_rate_nom = np.array([0.0, 0.0, 0.0])
        self.traj_W = np.zeros(3)
        self.Pi = np.array([0.0, 0.0, 0.0])
        # * intialize input values
        self.input_acc = 0.0
        self.input_Wb = np.zeros(3)

        # * initialize trajectory module and tmporary position command
        self.trajectory = Trajectory()
        self.tmp_pos = np.zeros(3)
        self.traj_plan = None

    # ! set reference trajectory and temporary position command
    def set_reference(self, traj_plan, t, tmp_P=np.zeros(3)):
        self.trajectory.set_clock(t)
        self.trajectory.set_traj_plan(traj_plan)

        self.tmp_pos = np.zeros(3)
        self.Pi = np.zeros(3)
        self.ki = np.array([0.0, 0.0, 0.0])

        # * set takeoff position for polynominal land trajectory
        if traj_plan == "takeoff" or traj_plan == "takeoff_50cm":
            self.tmp_pos = tmp_P
            self.ki[2] = 0.5
            self.q_ref = np.array([[0], [0], [0], [0], [0], [0], [0], [0]])
            self.A = A1
            self.Bl = Bl1
            self.K = K1
        # * set pendulum
        elif traj_plan == "pendulum":
            self.tmp_pos = tmp_P
            self.ki[2] = 0.5
            self.q_ref = np.array(
                [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
            self.A = A2
            self.Bl = Bl2
            self.K = K2
        # * set servo_pendulum
        elif traj_plan == "servo_pendulum":
            self.tmp_pos = tmp_P
            self.ki[2] = 0.5
            self.q_ref = np.array(
                [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
            self.A = At
            self.Bl = Bt
            self.K = Kt
        # * escape pendulum
        elif traj_plan == "espendulum":
            self.tmp_pos = tmp_P
            self.ki[2] = 0.5
            self.q_ref = np.array([[0], [0], [0], [0], [0], [0], [0], [0]])
            self.A = A1
            self.Bl = Bl1
            self.K = K1
        # * set translate
        elif traj_plan == "translate":
            self.tmp_pos = tmp_P
            self.ki[2] = 0.5
            self.q_ref = np.array([[0], [0], [0], [0], [0], [0], [0], [0]])
            self.A = A1
            self.Bl = Bl1
            self.K = K1

        # * set landing position for polynominal land trajectory（振子載せたまま着地する場合）
        # elif traj_plan == "land" or traj_plan == "land_50cm":
        #     self.tmp_pos = tmp_P
        #     self.ki[2] = 1.5
        #     self.q = np.array(
        #     [
        #         [self.P[0]],
        #         [self.V[0]],
        #         [self.Pan[0]],
        #         [self.Vpan[0]],
        #         [self.Euler[1]],
        #         [self.P[1]],
        #         [self.V[1]],
        #         [self.Pan[1]],
        #         [self.Vpan[1]]
        #         [self.Euler[0]],
        #         [self.P[2]],
        #         [self.V[2]],
        #     ]
        #     )
        #     self.q_ref = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
        #     self.A = A2
        #     self.Bl = Bl2
        #     self.K = K2
        elif traj_plan == "land" or traj_plan == "land_50cm":
            self.tmp_pos = tmp_P
            self.ki[2] = 1.5
            self.q_ref = np.array([[0], [0], [0], [0], [0], [0], [-0.2], [0]])
            self.A = A1
            self.Bl = Bl1
            self.K = K1
        # * set stop position when stop tracking trajectory
        elif traj_plan == "stop":
            self.tmp_pos = tmp_P

        elif traj_plan == "move_to_zero_from_x_1.3":
            self.ki[2] = 2.0

    def set_state(self, state):
        self.P = state.P
        self.V = state.V
        self.R = state.R
        self.Euler = state.Euler
        # 振子追加
        # self.Pe = state.Pe
        self.Pan = state.Pan
        self.Vpan = state.Vpan
        self.y = state.diff_with_ref

    def Position_controller(self):
        # * set desired state of trajectory

        # traj_pos = self.trajectory.traj_pos + self.tmp_pos
        # traj_vel = self.trajectory.traj_vel
        # traj_acc = self.trajectory.traj_acc

        # # * calculate nominal acceleration in 3 dimentions
        # self.Pi += (traj_pos - self.P) * self.dt
        # self.ref_acc = (
        #     self.kp * (traj_pos - self.P)
        #     + self.kv * (traj_vel - self.V)
        #     + self.ki * self.Pi
        #     + traj_acc
        # )

        # * calculate input acceleration
        # q = [x,dx,θ,dθ,β,y,dy,Φ,dΦ,γ,z,dz]
        if (self.tmp_pos == np.array([0.0, 0.0, 0.0])).all():
            q = np.array(
                [
                    [self.P[0]],
                    [self.V[0]],
                    [self.Euler[1]],
                    [self.P[1]],
                    [self.V[1]],
                    [self.Euler[0]],
                    [self.P[2]-0.2],
                    [self.V[2]],
                ]
            )
        elif (self.tmp_pos == np.array([2.0, 0.0, 0.0])).all():
            q = np.array(
                [
                    [self.P[0]],
                    [self.V[0]],
                    [self.Euler[1]],
                    [self.P[1]],
                    [self.V[1]],
                    [self.Euler[0]],
                    [self.P[2]-0.2],
                    [self.V[2]],
                ]
            )
        else:
            self.y = r0 - q*C
            q = np.array(
                [
                    [self.P[0]],
                    [self.V[0]],
                    [self.Pan[0]],
                    [self.Vpan[0]],
                    [self.Euler[1]],
                    [self.P[1]],
                    [self.V[1]],
                    [self.Pan[1]],
                    [self.Vpan[1]],
                    [self.Euler[0]],
                    [self.P[2]-0.2],
                    [self.V[2]],
                    [self.y],
                ]
            )

        q_div = q - self.q_ref
        u = -np.dot(self.K, q_div)

        # nominal acceleraion
        ad = u[2, 0]
        a = (
            ad
            + g
            + (g / 2) * (self.Euler[1] * self.Euler[1] +
                         self.Euler[0] * self.Euler[0])
        )

        self.input_acc = a

    def Attitude_controller(self):
        # # * set desired state of trajectory
        # traj_acc = self.trajectory.traj_acc
        # traj_jer = self.trajectory.traj_jer
        # traj_yaw = self.trajectory.traj_yaw
        # traj_yaw_rate = self.trajectory.traj_yaw_rate

        # # * calculate nominal Rotation matrics
        # traj_R = np.zeros((3, 3))
        # traj_Rxc = np.array([np.cos(traj_yaw), np.sin(traj_yaw), 0.0])
        # traj_Ryc = np.array([-np.sin(traj_yaw), np.cos(traj_yaw), 0.0])
        # traj_Rz = self.ref_acc / np.linalg.norm(self.ref_acc)
        # traj_Rx = np.cross(traj_Ryc, traj_Rz) / np.linalg.norm(
        #     np.cross(traj_Ryc, traj_Rz)
        # )
        # traj_Ry = np.cross(traj_Rz, traj_Rx)

        # traj_R[:, 0] = traj_Rx
        # traj_R[:, 1] = traj_Ry
        # traj_R[:, 2] = traj_Rz

        # * calculate nominal Angular velocity
        # traj_wy = np.dot(traj_Rx, traj_jer) / np.dot(traj_Rz, traj_acc)
        # traj_wx = -np.dot(traj_Ry, traj_jer) / np.dot(traj_Rz, traj_acc)
        # traj_wz = (
        #     traj_yaw_rate * np.dot(traj_Rxc, traj_Rx)
        #     + traj_wy * np.dot(traj_Ryc, traj_Rz)
        # ) / np.linalg.norm(np.cross(traj_Ryc, traj_Rz))
        # self.traj_W[0] = traj_wx
        # self.traj_W[1] = traj_wy
        # self.traj_W[2] = traj_wz

        # # * calculate input Body angular velocity
        # self.input_Wb = self.traj_W + self.kR * self.Wedge(
        #     -(np.matmul(traj_R.T, self.R) - np.matmul(self.R.T, traj_R)) / 2.0
        # )

        # * calculate nominal Euler angle and Euler angle rate
        # q = [x,dx,θ,dθ,β,y,dy,Φ,dΦ,γ,z,dz]
        if (self.tmp_pos == np.array([0.0, 0.0, 0.0])).all():
            q = np.array(
                [
                    [self.P[0]],
                    [self.V[0]],
                    [self.Euler[1]],
                    [self.P[1]],
                    [self.V[1]],
                    [self.Euler[0]],
                    [self.P[2]-0.2],
                    [self.V[2]],
                ]
            )
        elif (self.tmp_pos == np.array([2.0, 0.0, 0.0])).all():
            q = np.array(
                [
                    [self.P[0]],
                    [self.V[0]],
                    [self.Euler[1]],
                    [self.P[1]],
                    [self.V[1]],
                    [self.Euler[0]],
                    [self.P[2]-0.2],
                    [self.V[2]],
                ]
            )
        else:
            y = r0 - q*C
            q = np.array(
                [
                    [self.P[0]],
                    [self.V[0]],
                    [self.Pan[0]],
                    [self.Vpan[0]],
                    [self.Euler[1]],
                    [self.P[1]],
                    [self.V[1]],
                    [self.Pan[1]],
                    [self.Vpan[1]],
                    [self.Euler[0]],
                    [self.P[2]-0.2],
                    [self.V[2]],
                    [self.y],
                ]
            )

        q_div = q - self.q_ref
        u = -np.dot(self.K, q_div)

        # ωxとωy(ωxとωyが逆の可能性あり)
        ox = u[1]
        oy = u[0]
        # oz = -10 * self.Euler[2]
        oz = -1 * self.Euler[2]

        """
        # calculate input Body angular velocity
        self.input_Wb = self.traj_W + self.kR * self.Wedge(
            -(np.matmul(traj_R.T, self.R) - np.matmul(self.R.T, traj_R)) / 2.0
        )
        """

        # calculate input Body angular velocity

        # self.input_Wb = np.array([[ox], [oy], [oz]], dtype=object)
        # self.input_Wb = np.array([ox, oy, oz])
        self.input_Wb[0] = ox
        self.input_Wb[1] = oy
        self.input_Wb[2] = oz
        """
        ten1 = np.array([ox, oy, oz])
        ten2 = ten1.tolist()
        self.input_Wb = ten2
        """

        # self.Euler_nom[1] = np.arctan(
        #     (traj_acc[0] * np.cos(traj_yaw) + traj_acc[1] * np.sin(traj_yaw))
        #     / (traj_acc[2])
        # )
        # self.Euler_nom[0] = np.arctan(
        #     (traj_acc[0] * np.sin(traj_yaw) - traj_acc[1] * np.cos(traj_yaw))
        #     / np.sqrt(
        #         (traj_acc[2]) ** 2
        #         + (traj_acc[0] * np.cos(traj_yaw) + traj_acc[2] * np.sin(traj_yaw)) ** 2
        #     )
        # )
        # self.Euler_nom[2] = traj_yaw

        # self.input_Euler_rate = self.BAV2EAR(self.Euler_nom, self.input_Wb)
        # self.Euler_rate_nom = self.BAV2EAR(self.Euler_nom, self.traj_W)

    # ! control flow calculate input accelaretion and input body angular velocity
    def mellinger_ctrl(self, t):
        self.trajectory.set_clock(t)
        self.trajectory.set_traj()
        self.Position_controller()
        self.Attitude_controller()

    # ! take logs of desired states and input
    def log_nom(self, log, t):
        log.write_nom(
            t=t,
            input_acc=self.input_acc,
            input_Wb=self.input_Wb,
            P=self.trajectory.traj_pos + self.tmp_pos,
            V=self.trajectory.traj_vel,
            Euler=self.Euler_nom,
            Wb=self.traj_W,
            Euler_rate=self.Euler_rate_nom,
            L=np.zeros(3),
            VL=np.zeros(3),
            q=np.zeros(3),
            dq=np.zeros(3),
        )
