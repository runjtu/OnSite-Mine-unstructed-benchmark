import math
import numpy as np
import scipy.linalg as la

# State 对象表示自车的状态，位置x、y，以及横摆角yaw、速度v
class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

class LQRController:

    def __init__(self):
        self.dt = 0.1  # time tick[s]，采样时间
        self.L = 5.6  # Wheel base of the vehicle [m]，车辆轴距
        self.max_steer =1#np.deg2rad(90)  # maximum steering angle[rad]
        # LQR parameter
        self.lqr_Q = np.eye(5)
        # self.lqr_Q[0][0] = 5
        # self.lqr_Q[1][1] = 5
        # self.lqr_Q[2][2] = 10
        self.lqr_R = np.eye(2)
        self.pe, self.pth_e = 0.0, 0.0
        self.keyind = 0 
        self.isback = False     

    def pi_2_pi(self,angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    # 实现离散Riccati equation 的求解方法
    def solve_dare(self,A, B, Q, R):
        """
        solve a discrete time_Algebraic Riccati equation (DARE)
        """
        x = Q
        x_next = Q
        max_iter = 150
        eps = 0.01

        for i in range(max_iter):
            x_next = A.T @ x @ A - A.T @ x @ B @ \
                    la.inv(R + B.T @ x @ B) @ B.T @ x @ A + Q
            if (abs(x_next - x)).max() < eps:
                break
            x = x_next

        return x_next

    # 返回值K 即为LQR 问题求解方法中系数K的解
    def dlqr(self,A, B, Q, R):
        """Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        # ref Bertsekas, p.151
        """

        # first, try to solve the ricatti equation
        X = self.solve_dare(A, B, Q, R)

        # compute the LQR gain
        K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

        eig_result = la.eig(A - B @ K)

        return K, X, eig_result[0]

    # 计算距离自车当前位置最近的参考点
    def calc_nearest_index(self,state, cx, cy, cyaw):
        dx = [state.x - icx for icx in cx]
        dy = [state.y - icy for icy in cy]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        mind = min(d)

        ind = d.index(mind)

        mind = math.sqrt(mind)

        dxl = cx[ind] - state.x
        dyl = cy[ind] - state.y
        angle = self.pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        return ind, mind

    def process(self,vehicle_info,path_waypoints,speed_profile,mode):
        # if self.keyind != 0 and self.isback:
        #     path_waypoints = path_waypoints[self.keyind+1:]
        #     speed_profile = speed_profile[self.keyind+1:]
        cx,cy,cyaw,ck,cgear=[],[],[],[],[]
        for i in path_waypoints:
            cx.append(i[0])
            cy.append(i[1])
            cyaw.append(i[2])
            cgear.append(i[3])
            ck.append(i[4])
        _ego_v = vehicle_info['ego']['v_mps']
        _ego_x = vehicle_info['ego']['x']
        _ego_y = vehicle_info['ego']['y']
        _ego_yaw = vehicle_info['ego']['yaw_rad']
        state = State(_ego_x, _ego_y, _ego_yaw, _ego_v)

        ind, e = self.calc_nearest_index(state, cx, cy, cyaw)
        
        if cgear[ind]==True:
            gear = 1
        else:
            gear = 3
        # if (cgear[ind]==2 or cgear[ind]==False) and not self.isback:
        #     for i in range(len(path_waypoints)):
        #         if path_waypoints[i][-1]==2:
        #             self.keyind = i
        #             break
        #     self.isback = True
        sp = speed_profile
        tv = sp[ind]
        k = ck[ind]
        v_state = state.v
        th_e = self.pi_2_pi(state.yaw - cyaw[ind])

        # 构建LQR表达式，X(k+1) = A * X(k) + B * u(k), 使用Riccati equation 求解LQR问题
    #     dt表示采样周期，v表示当前自车的速度
    #     A = [1.0, dt, 0.0, 0.0, 0.0
    #          0.0, 0.0, v, 0.0, 0.0]
    #          0.0, 0.0, 1.0, dt, 0.0]
    #          0.0, 0.0, 0.0, 0.0, 0.0]
    #          0.0, 0.0, 0.0, 0.0, 1.0]
        A = np.zeros((5, 5))
        A[0, 0] = 1.0
        A[0, 1] = self.dt
        A[1, 2] = v_state
        A[2, 2] = 1.0
        A[2, 3] = self.dt
        A[4, 4] = 1.0

        # 构建B矩阵，L是自车的轴距
        # B = [0.0, 0.0
        #     0.0, 0.0
        #     0.0, 0.0
        #     v/L, 0.0
        #     0.0, dt]
        B = np.zeros((5, 2))
        B[3, 0] = v_state / self.L
        B[4, 1] = self.dt

        K, _, _ = self.dlqr(A, B, self.lqr_Q, self.lqr_R)

        # state vector，构建状态矩阵
        # x = [e, dot_e, th_e, dot_th_e, delta_v]
        # e: lateral distance to the path， e是自车到轨迹的距离
        # dot_e: derivative of e， dot_e是自车到轨迹的距离的变化率
        # th_e: angle difference to the path， th_e是自车与期望轨迹的角度偏差
        # dot_th_e: derivative of th_e， dot_th_e是自车与期望轨迹的角度偏差的变化率
        # delta_v: difference between current speed and target speed，delta_v是当前车速与期望车速的偏差
        X = np.zeros((5, 1))
        X[0, 0] = e
        X[1, 0] = (e - self.pe) / self.dt
        X[2, 0] = th_e
        X[3, 0] = (th_e - self.pth_e) / self.dt
        X[4, 0] = v_state - tv
        # input vector，构建输入矩阵u
        # u = [delta, accel]
        # delta: steering angle，前轮转角
        # accel: acceleration，自车加速度
        ustar = -K @ X

        # calc steering input
        ff = math.atan2(self.L * k, 1)  # feedforward steering angle
        fb = self.pi_2_pi(ustar[0, 0])  # feedback steering angle
        delta = ff + fb
        if abs(delta)>self.max_steer:
            if delta>0:
                delta = self.max_steer
            else:
                delta = -self.max_steer
        if gear == 3 and mode=="complex":
            # calc accel input
            accel = -ustar[1, 0]
        else:
            accel = ustar[1, 0]
        if abs(accel)>15:
            if accel>0:
                accel = 15
            else:
                accel = -15
        return (accel,delta,gear)
    