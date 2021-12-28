import numpy as np
import pdb

class GlobalPlanner:
    def __init__(self, dist, global_dist, horizon, global_horizon, model, planner, use_global_planner = False):
        self.dist = dist # the distance between intermedia waypoints
        self.global_dist = global_dist
        self.horizon = horizon # the planning horizon
        self.global_horizon = global_horizon
        self.model = model # dynamic model of the robot
        self._state_dimension = 2
        self.planner = planner
        self.use_global_planner = use_global_planner
        pass

    def IntermediaGoalPlanner(self, dt: float, robot_state: np.ndarray, goal: np.ndarray, sensor_data: dict):
        '''
        Return:
            intermedia_goal: np.array, []
            traj: np.array, trajectory from the current robot state to the intermedia goal
        '''        
        if (self.use_global_planner):
            goal_rel_pos = goal - np.array(robot_state[:2])
            goal_rel_pos = self.global_dist * goal_rel_pos / (np.linalg.norm(goal_rel_pos, 2))  
            gloabl_goal = np.vstack(np.array([goal_rel_pos[0]+robot_state[0], goal_rel_pos[1]+robot_state[1], 0, 0]))
            global_traj = self.IntegratorPlanner(dt, robot_state, gloabl_goal, self.global_horizon)
            safe_global_traj = self.planner(dt, gloabl_goal, global_traj, sensor_data)
            
            intermedia_goal = None
            min_dist = float("inf")
            for point in safe_global_traj:
                dist = np.linalg.norm(np.array(robot_state[:2]) - point[:2])
                if abs(dist - self.dist) < 0.01:
                    intermedia_goal = point
                    break
                elif (dist < min_dist):
                    min_dist = dist
                    intermedia_goal = point
            #pdb.set_trace()
            intermedia_goal = np.vstack(intermedia_goal)
        else:
            goal_rel_pos = goal - np.array(robot_state[:2])
            goal_rel_pos = self.dist * goal_rel_pos / (np.linalg.norm(goal_rel_pos, 2))        
            intermedia_goal = np.vstack(np.array([goal_rel_pos[0]+robot_state[0], goal_rel_pos[1]+robot_state[1], 0, 0]))
        traj = self.IntegratorPlanner(dt, robot_state, intermedia_goal, self.horizon)
        return intermedia_goal, traj


    def IntegratorPlanner(self, dt, cur_state, goal_state, N):
        # assume integrater uses first _state_dimension elements from est data
        state = np.vstack(cur_state.ravel())
        xd = self._state_dimension
        # both state and goal is in [pos, vel, etc.]' with shape [T, ?, 1]
        A = self.model.A(dt=dt)
        B = self.model.B(dt=dt)

        # lifted system for tracking last state
        Abar = np.vstack([np.linalg.matrix_power(A, i) for i in range(1,N+1)])
        Bbar = np.vstack([
            np.hstack([
                np.hstack([np.linalg.matrix_power(A, p) @ B for p in range(row, -1, -1)]),
                np.zeros((xd, N-1-row))
            ]) for row in range(N)
        ])
        #pdb.set_trace()
        # tracking each state dim
        n_state_comp = 2 #len(self.model.state_component) # number of pos, vel, etc.
        traj = np.zeros((N, xd * n_state_comp, 1))
        for i in range(xd):
            # vector: pos, vel, etc. of a single dimension
            x = np.vstack([ state[ j * xd + i, 0 ] for j in range(n_state_comp) ])
            xref = np.vstack([ goal_state[ j * xd + i, 0 ] for j in range(n_state_comp) ])

            ubar = np.linalg.lstsq(
                a = Bbar[-xd:, :], b = xref - np.linalg.matrix_power(A, N) @ x)[0] # get solution

            xbar = (Abar @ x + Bbar @ ubar).reshape(N, n_state_comp, 1)

            for j in range(n_state_comp):
                traj[:, j * xd + i] = xbar[:, j]
        traj = traj.squeeze()        
        return traj
