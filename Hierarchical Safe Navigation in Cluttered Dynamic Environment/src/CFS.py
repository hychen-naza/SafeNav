from abc import ABC, abstractmethod
from operator import concat
from matplotlib.pyplot import axis
import numpy as np
from cvxopt import matrix, solvers
import cvxopt
import pdb
from utils import *

class Planner(ABC):
    def __init__(self, spec, model) -> None:
        self.spec = spec
        self.model = model
        self.replanning_cycle = spec["replanning_cycle"]
        self.horizon = spec["horizon"]
        self._state_dimension = spec["state_dimension"]
        self.dT = spec["dT"]
    
    @property
    def state_dimension(self):
        return self._state_dimension

    @abstractmethod
    def _plan(self, dt: float, goal: dict, traj: np.array, est_data: dict) -> np.array:
        '''
            Implementation of planner
        '''
        pass

    def __call__(self, dt: float, goal: dict, traj: np.array, est_data: dict) -> np.array:
        '''
            Public interface
        '''
        return self._plan(dt, goal, traj, est_data)

class CFSPlanner(Planner):

    def __init__(self, spec, model, safety_dist = 0.02) -> None:
        super().__init__(spec, model)
        self.max_speed = 0.02
        self.max_acc = 0.04
        self.D = safety_dist

    def _CFS(self, 
        x_ref,
        n_ob,
        obs_traj,
        cq = [10,0,10], 
        cs = [0,1,0.1], 
        minimal_dis = 0, 
        ts = 1, 
        maxIter = 30,
        stop_eps = 1e-3
    ):
        # has obstacle, the normal CFS procedure 
        x_rs = np.array(x_ref)

        # planning parameters 
        h = x_rs.shape[0]    
        dimension = x_rs.shape[1]

        # flatten the trajectory to one dimension
        # flatten to one dimension for applying qp, in the form of x0,y0,x1,y1,...
        x_rs = np.reshape(x_rs, (x_rs.size, 1))
        x_origin = x_rs
        
        # objective terms 
        # identity
        Q1 = np.identity(h * dimension)
        S1 = Q1
        # velocity term 
        Vdiff = np.identity(h*dimension) - np.diag(np.ones((1,(h-1)*dimension))[0],dimension)
        Q2 = np.matmul(Vdiff.transpose(),Vdiff) 
        # Acceleration term 
        Adiff = Vdiff - np.diag(np.ones((1,(h-1)*dimension))[0],dimension) + np.diag(np.ones((1,(h-2)*dimension))[0],dimension*2)
        Q3 = np.matmul(Adiff.transpose(),Adiff)
        # Vdiff = eye(nstep*dim)-diag(ones(1,(nstep-1)*dim),dim);
        #pdb.set_trace()
        # objective 
        Q = Q1*cq[0]+Q2*cq[1]+Q3*cq[2]
        S = S1*cs[0]+Q2*cs[1]+Q3*cs[2]

        # quadratic term
        H =  Q + S 
        # linear term
        f = -1 * np.dot(Q, x_origin)

        b = np.ones((h * n_ob, 1)) * (-minimal_dis)
        H = matrix(H,(len(H),len(H[0])),'d')
        f = matrix(f,(len(f), 1),'d')
        # b = matrix(b,(len(b),1),'d')

        # reference trajctory cost 
        J0 =  np.dot(np.transpose(x_rs - x_origin), np.dot(Q, (x_rs - x_origin))) + np.dot(np.transpose(x_rs), np.dot(S, x_rs))
        J = float('inf')
        dlt = float('inf')
        cnt = 0

        # equality constraints 
        # start pos and end pos remain unchanged 
        Aeq = np.zeros((dimension*2, len(x_rs)))
        for i in range(dimension):
            Aeq[i,i] = 1
            Aeq[dimension*2-i-1, len(x_rs)-i-1] = 1
        
        beq = np.zeros((dimension*2, 1))
        beq[0:dimension,0] = x_rs[0:dimension,0]
        beq[dimension:dimension*2, 0] = x_rs[dimension*(h-1): dimension*h, 0] 
        # transform to convex optimization matrix 
        Aeq_array = Aeq
        beq_array = beq
        Aeq = matrix(Aeq,(len(Aeq),len(Aeq[0])),'d')
        beq = matrix(beq,(len(beq),1),'d')

        # set the safety margin 
        

        # fig, ax = plt.subplots()
        # main CFS loop
        #while dlt > stop_eps:
        #    cnt += 1
        Lstack, Sstack = [], []
        # inequality constraints 
        # l * x <= s
        Constraint = np.zeros((h * n_ob, len(x_rs)))
        
        for i in range(h):
            # get reference pos at time step i
            if i < h-1 and i > 0:
                x_r = x_rs[i * dimension : (i + 1) * dimension] 

                # get inequality value (distance)
                # get obstacle at this time step 
                for j in range(int(len(obs_traj[i])/2)):
                    obs_p = obs_traj[i,j*2:(j+1)*2]                      
                    dist = self._ineq(x_r,obs_p)
                    # get gradient 
                    ref_grad = jac_num(self._ineq, x_r, obs_p)
                    # compute
                    s = dist - self.D - np.dot(ref_grad, x_r)
                    l = -1 * ref_grad
                    # update 
                    Sstack = vstack_wrapper(Sstack, s)
                    l_tmp = np.zeros((1, len(x_rs)))
                    l_tmp[:,i*dimension:(i+1)*dimension] = l
                    Lstack = vstack_wrapper(Lstack, l_tmp)
                '''
                obs_p = obs_traj[i,:2]                      
                dist = self._ineq(x_r,obs_p)
                #pdb.set_trace()
                # print(dist)

                # get gradient 
                ref_grad = jac_num(self._ineq, x_r, obs_p)
                # print(ref_grad)

                # compute
                s = dist - D - np.dot(ref_grad, x_r)
                l = -1 * ref_grad
                '''
            if i == h-1 or i == 0: # don't need inequality constraints for lst dimension 
                s = np.zeros((1,1))
                l = np.zeros((1,2))

            # update 
            '''
            Sstack = vstack_wrapper(Sstack, s)
            l_tmp = np.zeros((1, len(x_rs)))
            l_tmp[:,i*dimension:(i+1)*dimension] = l
            Lstack = vstack_wrapper(Lstack, l_tmp)
            '''
        # QP solver 
        Lstack = matrix(Lstack,(len(Lstack),len(Lstack[0])),'d')
        cvxopt.solvers.options['show_progress'] = False            
        while True:
            try:
                #b = np.vstack([Sstack, VA_saturated])        
                Sstack_matrix = matrix(Sstack,(len(Sstack),1),'d')
                sol = solvers.qp(H, f, Lstack, Sstack_matrix, Aeq, beq)
                x_ts = sol['x']
                break
            except ValueError:
                # no solution, relax the constraint               
                for i in range(len(Sstack)):
                    Sstack[i][0] += 0.01                        
                #print(f"relax Sstack")    
            except ArithmeticError:
                #print(Lstack)
                print(f"Sstack {Sstack}, cnt {cnt}, maxIter {maxIter}")
                pdb.set_trace()
        
        x_ts = np.reshape(x_ts, (len(x_rs),1))
        J = np.dot(np.transpose(x_ts - x_origin), np.dot(Q, (x_ts - x_origin))) + np.dot(np.transpose(x_ts), np.dot(S, x_ts))
        dlt = min(abs(J - J0), np.linalg.norm(x_ts - x_rs))
        J0 = J
        x_rs = x_ts
        #if cnt >= maxIter:
        #        break
        
        # return the reference trajectory      
        x_rs = x_rs[: h * dimension]
        x_rs = x_rs.reshape(h, dimension)
        #print(f"traj {x_rs}")
        #pdb.set_trace()
        return x_rs

    def _New_CFS(self, 
        x_ref,
        n_ob,
        obs_traj,
    ):
        # has obstacle, the normal CFS procedure 
        x_rs = np.array(x_ref)

        # planning parameters 
        h = x_rs.shape[0]    
        dimension = x_rs.shape[1]

        # flatten the trajectory to one dimension
        # flatten to one dimension for applying qp, in the form of x0,y0,x1,y1,...
        x_rs = np.reshape(x_rs, (x_rs.size, 1))
        x_origin = x_rs
        #pdb.set_trace()

        # quadratic objective term
        Q = cvxopt.matrix(np.identity(h * dimension))
        p = cvxopt.matrix(-2 * x_origin) 

        # equality constraints 
        # start pos and end pos remain unchanged 
        # the end pos constraint can be removed NAZA
        Aeq = np.zeros((dimension*2, len(x_rs)))
        for i in range(dimension):
            Aeq[i,i] = 1
            Aeq[dimension*2-i-1, len(x_rs)-i-1] = 1
        
        beq = np.zeros((dimension*2, 1))
        beq[0:dimension,0] = x_rs[0:dimension,0]
        beq[dimension:dimension*2, 0] = x_rs[dimension*(h-1): dimension*h, 0] 
        # transform to convex optimization matrix 
        A = matrix(Aeq,(len(Aeq),len(Aeq[0])),'d')
        b = matrix(beq,(len(beq),1),'d')

        # inequality constraints 
        # velocity and acceleration constraints
        Vdiff = np.identity(h*dimension) - np.diag(np.ones((1,(h-1)*dimension))[0],dimension)  # NAZA FIXME
        Adiff = Vdiff - np.diag(np.ones((1,(h-1)*dimension))[0],dimension) + np.diag(np.ones((1,(h-2)*dimension))[0],dimension*2)
        Vdiff = -1 * 1/self.dT * Vdiff
        Adiff = 1/(self.dT**2) * Adiff
        Vdiff_ = -Vdiff
        Adiff_ = -Adiff
        VA_constraints = np.vstack([Vdiff, Vdiff_, Adiff, Adiff_])
        VA_saturated = np.concatenate((np.array([self.max_speed]*(h*dimension)), np.array([self.max_speed]*(h*dimension)), \
                        np.array([self.max_acc]*(h*dimension)), np.array([self.max_acc]*(h*dimension)))).reshape(-1, 1)
        #VA_saturated = cvxopt.matrix(VA_saturated)

        # set the safety margin 
        D = 0.02
        # main CFS loop        
        # inequality constraints 
        Lstack, Sstack = [], []
        
        for i in range(h):
            # get reference pos at time step i
            if i < h-1 and i > 0:
                x_r = x_rs[i * dimension : (i + 1) * dimension] 
                # get inequality value (distance)
                # get obstacle at this time step 
                #print(f"len(obs_traj[i])/2 {len(obs_traj[i])/2}")
                for j in range(int(len(obs_traj[i])/2)):
                    obs_p = obs_traj[i,j*2:(j+1)*2]                      
                    dist = self._ineq(x_r,obs_p)
                    # get gradient 
                    ref_grad = jac_num(self._ineq, x_r, obs_p)
                    # compute
                    s = dist - D - np.dot(ref_grad, x_r)
                    l = -1 * ref_grad
                    # update 
                    Sstack = vstack_wrapper(Sstack, s)
                    l_tmp = np.zeros((1, len(x_rs)))
                    l_tmp[:,i*dimension:(i+1)*dimension] = l
                    Lstack = vstack_wrapper(Lstack, l_tmp)
            if i == h-1 or i == 0: # don't need inequality constraints for lst dimension 
                s = np.zeros((1,1))
                l = np.zeros((1,2))
        
        #pdb.set_trace()
        #A = np.vstack([Lstack, VA_constraints])
        G = Lstack #VA_constraints
        G = cvxopt.matrix(G, (len(G),len(G[0])), 'd')
        cvxopt.solvers.options['show_progress'] = False
        while True:
            try:
                #b = np.vstack([Sstack, VA_saturated])        
                h_ineq = Sstack #VA_saturated
                h_ineq = cvxopt.matrix(h_ineq, (len(h_ineq),1), 'd')
                sol = solvers.qp(Q, p, G, h_ineq, A, b)
                #sol = solvers.qp(Q, p, A, b)
                x_ts = sol['x']
                break
            except ValueError:
                #pdb.set_trace()
                # no solution, relax the constraint               
                for i in range(len(Sstack)):
                    Sstack[i][0] += 0.01                        
                print(f"relax Sstack {Sstack}")     
        x_ts = np.reshape(x_ts, (len(x_rs),1))
        x_rs = x_ts
        # return the reference trajectory    
        #pdb.set_trace()  
        x_rs = x_rs[: h * dimension]
        print(f"eq {Aeq @ x_rs}, beq {beq}") # 
        x_rs = x_rs.reshape(h, dimension)
        print(f"after CFS traj {x_rs}")
        #pdb.set_trace()
        return x_rs

    def _ineq(self, x, obs):
        '''
        inequality constraints. 
        constraints: ineq(x) > 0
        '''
        # norm distance restriction
        obs_p = obs.flatten()
        obs_r = 0.04 # todo tune
        obs_r = np.array(obs_r)
        
        # flatten the input x 
        x = x.flatten()
        dist = np.linalg.norm(x - obs_p) - obs_r

        return dist


    def _plan(self, dt: float, goal: dict, traj: np.array, est_data: dict) -> np.array:
        
        xd = self.state_dimension
        N = self.horizon
        obs_pos_list = []
        obs_vel_list = []
        obs_traj = []
        for name, info in est_data['obstacle_sensor_est'].items():
            if 'obs' in name:
                obs_pos_list.append(info['pos'] - est_data['cartesian_sensor_est']['pos'])
                obs_vel_list.append(info['vel'])

        # Without obstacle, then collision free
        if (len(obs_pos_list) == 0):
            return traj  

        # Estimate the obstacle's trajectory
        for obs_pos, obs_vel in zip(obs_pos_list, obs_vel_list):
            one_traj = []
            for i in range(N):
                obs_waypoint = obs_pos + obs_vel * i * dt                
                obs_waypoint = obs_waypoint.reshape(1,-1).tolist()[0]
                one_traj.append(obs_waypoint) # [N, xd]
            obs_traj.append(one_traj)
        obs_traj = np.array(obs_traj)
        
        if len(obs_traj) > 1:
            obs_traj = np.concatenate(obs_traj, axis=-1) # [T, n_obs * xd]
        else:
            obs_traj = obs_traj[0]
        #print(f"before CFS traj {traj}, obs traj {obs_traj}")
        # CFS
        traj_pos_only = traj[:, :xd]
        #print(f"robot traj_pos_only {traj_pos_only}, obs traj {obs_traj}")
        traj_pos_safe = self._CFS(x_ref=traj_pos_only, n_ob=len(obs_pos_list), obs_traj=obs_traj)
        traj[:, :xd] = traj_pos_safe
        #pdb.set_trace()
        
        
        return traj

    def global_frame_plan(self, dt: float, goal: dict, traj: np.array, est_data: dict) -> np.array:
        
        xd = self.state_dimension
        N = self.horizon
        obs_pos_list = []
        obs_vel_list = []
        obs_traj = []
        for name, info in est_data['obstacle_sensor_est'].items():
            if 'obs' in name:
                obs_pos_list.append(info['pos'])
                obs_vel_list.append(info['vel'])

        # Without obstacle, then collision free
        if (len(obs_pos_list) == 0):
            return traj  

        # Estimate the obstacle's trajectory
        for obs_pos, obs_vel in zip(obs_pos_list, obs_vel_list):
            one_traj = []
            for i in range(N):
                obs_waypoint = obs_pos + obs_vel * i * dt                
                obs_waypoint = obs_waypoint.reshape(1,-1).tolist()[0]
                one_traj.append(obs_waypoint) # [N, xd]
            obs_traj.append(one_traj)
        obs_traj = np.array(obs_traj)
        
        if len(obs_traj) > 1:
            obs_traj = np.concatenate(obs_traj, axis=-1) # [T, n_obs * xd]
        else:
            obs_traj = obs_traj[0]
        # CFS
        traj_pos_only = traj[:, :xd]
        traj_pos_safe = self._CFS(x_ref=traj_pos_only, n_ob=len(obs_pos_list), obs_traj=obs_traj)
        traj[:, :xd] = traj_pos_safe
        #pdb.set_trace()        
        return traj



