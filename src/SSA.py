import numpy as np
import math
import cvxopt
import sys
import collections

class SafeSetAlgorithm():
    def __init__(self, max_speed, dmin = 0.12, k = 0.1, max_acc = 0.04):
        """
        Args:
            dmin: dmin for phi
            k: k for d_dot in phi
        """
        self.dmin = 0.04 #0.06#dmin #
        self.k = k
        self.max_speed = max_speed
        self.max_acc = max_acc
        self.forecast_step = 3
        self.records = collections.deque(maxlen = 10)
        self.acc_reward_normal_ssa = 0
        self.acc_reward_qp_ssa = 0
        self.acc_phi_dot_ssa = 0
        self.acc_phi_dot_qp = 0

    def get_safe_control(self, robot_state, obs_states, f, g, u0):
        """
        Args:
            robot_state <x, y, vx, vy>
            obs_state: np array closest static obstacle state <x, y, vx, vy, ax, ay>
        """
        u0 = np.array(u0).reshape((2,1))
        robot_vel = np.linalg.norm(robot_state[-2:])
        
        L_gs = []
        L_fs = []
        obs_dots = []
        reference_control_laws = []
        is_safe = True
        phis = []
        warning_indexs = []
        danger_indexs = []
        danger_obs = []
        record_data = {}
        record_data['obs_states'] = [obs[:2] for obs in obs_states]
        record_data['robot_state'] = robot_state
        record_data['phi'] = []
        record_data['phi_dot'] = []
        record_data['is_safe_control'] = False
        record_data['is_multi_obstacles'] = True if len(obs_states) > 1 else False
        for i, obs_state in enumerate(obs_states):
            d = np.array(robot_state - obs_state[:4])
            d_pos = d[:2] # pos distance
            d_vel = d[2:] # vel 
            d_abs = np.linalg.norm(d_pos)
            d_dot = self.k * (d_pos @ d_vel.T) / np.linalg.norm(d_pos)
            phi = np.power(self.dmin, 2) - np.power(np.linalg.norm(d_pos), 2) - d_dot
            record_data['phi'].append(phi)
            
            # calculate Lie derivative
            # p d to p robot state and p obstacle state
            p_d_p_robot_state = np.hstack([np.eye(2), np.zeros((2,2))]) # shape (2, 4)
            p_d_p_obs_state = np.hstack([-1*np.eye(2), np.zeros((2,2))]) # shape (2, 4)
            p_d_pos_p_d = np.array([d_pos[0], d_pos[1]]).reshape((1,2)) / d_abs # shape (1, 2)
            p_d_pos_p_robot_state = p_d_pos_p_d @ p_d_p_robot_state # shape (1, 4)
            p_d_pos_p_obs_state = p_d_pos_p_d @ p_d_p_obs_state # shape (1, 4)

            # p d_dot to p robot state and p obstacle state
            p_vel_p_robot_state = np.hstack([np.zeros((2,2)), np.eye(2)]) # shape (2, 4)
            p_vel_p_obs_state = np.hstack([np.zeros((2,2)), -1*np.eye(2)]) # shape (2, 4)
            p_d_dot_p_vel = d_pos.reshape((1,2)) / d_abs # shape (1, 2)
            
            p_pos_p_robot_state = np.hstack([np.eye(2), np.zeros((2,2))]) # shape (2, 4)
            p_pos_p_obs_state = np.hstack([-1*np.eye(2), np.zeros((2,2))]) # shape (2, 4)
            p_d_dot_p_pos = d_vel / d_abs - 0.5 * (d_pos @ d_vel.T) * d_pos / np.power(d_abs, 3) 
            p_d_dot_p_pos = p_d_dot_p_pos.reshape((1,2)) # shape (1, 2)

            p_d_dot_p_robot_state = p_d_dot_p_pos @ p_pos_p_robot_state + p_d_dot_p_vel @ p_vel_p_robot_state # shape (1, 4)
            p_d_dot_p_obs_state = p_d_dot_p_pos @ p_pos_p_obs_state + p_d_dot_p_vel @ p_vel_p_obs_state # shape (1, 4)

            p_phi_p_robot_state = -2 * np.linalg.norm(d_pos) * p_d_pos_p_robot_state - \
                            self.k * p_d_dot_p_robot_state # shape (1, 4)
            p_phi_p_obs_state = -2 * np.linalg.norm(d_pos) * p_d_pos_p_obs_state - \
                            self.k * p_d_dot_p_obs_state # shape (1, 4)
        
            L_f = p_phi_p_robot_state @ (f @ robot_state.reshape((-1,1))) # shape (1, 1)
            L_g = p_phi_p_robot_state @ g # shape (1, 2) g contains x information
            obs_dot = p_phi_p_obs_state @ obs_state[-4:]
            L_fs.append(L_f)
            phis.append(phi)  
            obs_dots.append(obs_dot)

            if (phi > 0):
                L_gs.append(L_g)                                              
                reference_control_laws.append( -0.5*phi - L_f - obs_dot)
                is_safe = False
                danger_indexs.append(i)
                danger_obs.append(obs_state[:2])
                # constrain_obs.append(obs_state[:2])

        if (not is_safe):
            # Solve safe optimization problem
            # min_x (1/2 * x^T * Q * x) + (f^T * x)   s.t. Ax <= b
            u0 = u0.reshape(-1,1)
            u, reference_control_laws = self.solve_qp(robot_state, u0, L_gs, reference_control_laws, phis, danger_indexs, warning_indexs)
            reward_qp_ssa = robot_state[1] + (robot_state[3] + u[1]) + 1
            self.acc_reward_qp_ssa += reward_qp_ssa
            unavoid_collision = False
            record_data['control'] = u
            record_data['is_safe_control'] = True
            #self.records.append(record_data)
            return u, True, unavoid_collision, danger_obs                            
        u0 = u0.reshape(1,2)
        u = u0
        record_data['control'] = u[0]
        self.records.append(record_data)     
        return u[0], False, False, danger_obs

    def solve_qp(self, robot_state, u0, L_gs, reference_control_laws, phis, danger_indexs, warning_indexs):
        q = np.eye(2)
        Q = cvxopt.matrix(q)
        u_prime = -u0
        p = cvxopt.matrix(u_prime) #-u0
        G = cvxopt.matrix(np.vstack([np.eye(2), -np.eye(2), np.array([[1,0],[-1,0]]), np.array([[0,1],[0,-1]])]))
        S_saturated = cvxopt.matrix(np.array([self.max_acc, self.max_acc, self.max_acc, self.max_acc, \
                                    self.max_speed-robot_state[2], self.max_speed+robot_state[2], \
                                    self.max_speed-robot_state[3], self.max_speed+robot_state[3]]).reshape(-1, 1))
        #G = cvxopt.matrix(np.vstack([np.eye(2), -np.eye(2)]))
        #S_saturated = cvxopt.matrix(np.array([self.max_acc, self.max_acc, self.max_acc, self.max_acc]).reshape(-1, 1))
        L_gs = np.array(L_gs).reshape(-1, 2)
        reference_control_laws = np.array(reference_control_laws).reshape(-1,1)
        A = cvxopt.matrix([[cvxopt.matrix(L_gs), G]])
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['maxiters'] = 600
        while True:
            try:
                b = cvxopt.matrix([[cvxopt.matrix(reference_control_laws), S_saturated]])
                sol = cvxopt.solvers.qp(Q, p, A, b)
                u = sol["x"]
                break
            except ValueError:
                # no solution, relax the constraint                  
                for i in range(len(reference_control_laws)):
                    reference_control_laws[i][0] += 0.01
                #print(f"relax reference_control_law, reference_control_laws {reference_control_laws}")
        u = np.array([u[0], u[1]])
        return u, reference_control_laws