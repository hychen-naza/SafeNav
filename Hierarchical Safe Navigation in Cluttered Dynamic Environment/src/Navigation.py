from __future__ import print_function
from __future__ import absolute_import

# python modules
import argparse
import math
import copy
import numpy as np
import pdb
import collections

# project files
import dynamic_obstacle
import bounds
import robot # double integrator robot
import simu_env
import runner
import param
from turtle_display import TurtleRunnerDisplay
from SSA import SafeSetAlgorithm
from CFS import CFSPlanner
from GlobalPlanner import GlobalPlanner
from DynamicModel import DoubleIntegrator 
from Controller import NaiveFeedbackController
from EgoCircle import EgoCircle


def display_for_name( dname ):
    # choose none display or visual display
    if dname == 'turtle':
        return TurtleRunnerDisplay(800,800)
    else:
        return runner.BaseRunnerDisplay()

dT = 0.1

def run_kwargs( params ):
    in_bounds = bounds.BoundsRectangle( **params['in_bounds'] )
    goal_bounds = bounds.BoundsRectangle( **params['goal_bounds'] )
    min_dist = params['min_dist']
    ret = { 'field': dynamic_obstacle.ObstacleField(dt = dT),
            'robot_state': robot.DoubleIntegratorRobot( **( params['initial_robot_state'] ) ),
            'in_bounds': in_bounds,
            'goal_bounds': goal_bounds,
            'noise_sigma': params['noise_sigma'],
            'min_dist': min_dist,
            'nsteps': 1000 }
    return ret

def parser():
    prsr = argparse.ArgumentParser()
    prsr.add_argument( '--display',
                       choices=('turtle','text','none'),
                       default='none' )
    prsr.add_argument('--pgap', dest='use_pgap', action='store_true')
    prsr.add_argument('--no-pgap', dest='use_pgap', action='store_false')
    prsr.add_argument('--dgap', dest='use_dgap', action='store_true')
    prsr.add_argument('--no-dgap', dest='use_dgap', action='store_false')
    prsr.set_defaults(feature=False)
    return prsr


def main(display_name, use_pgap, use_dgap):
    # testing env
    try:
        params = param.params
    except Exception as e:
        print(e)
        return
    display = display_for_name(display_name)
    env_params = run_kwargs(params)

    # parameters
    max_steps = int(1e6)
    episode_reward = 0
    episode_num = 0
    total_rewards = []
    total_steps = 0
    # dynamic model parameters
    fx = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])#dynamic_model.Fx(dt=dT) 
    gx = np.array([[1,0],[0,1],[1,0],[0,1]])#dynamic_model.Gx(dt=dT) 
    
    collision_num = 0
    failure_num = 0
    success_num = 0
    # Goal position
    goal_pos = np.array([episode_num/100, 1.05])
    # Parameters
    horizon = 30
    global_horizon = 60
    # CFS Replanning Parameters
    replanning_cycle = 10
    replanning_timer = replanning_cycle
    traj = None

    # Build the env
    env = simu_env.Env(display, **(env_params))
    sensor_dist = env.min_dist * 6
    # Init Safe Controller SSA
    safe_controller = SafeSetAlgorithm(max_speed = env.robot_state.max_speed)

    # Init CFS Planner
    dynamic_model = DoubleIntegrator({'dT': dT})
    cfs_planner = CFSPlanner({'horizon': horizon, 'replanning_cycle': replanning_cycle, 'state_dimension': 2, 'dT':dT}, dynamic_model)
    global_cfs_planner = CFSPlanner({'horizon': global_horizon, 'replanning_cycle': replanning_cycle, 'state_dimension': 2, 'dT':dT}, dynamic_model, safety_dist=0.04)
    # Init Feedback Controller for CFS
    controller = NaiveFeedbackController()
    global_controller = NaiveFeedbackController(is_global = True)

    # Init Global Planner
    global_planner = GlobalPlanner(dist = 0.1, global_dist = 0.25, horizon = horizon, global_horizon = global_horizon, \
                                    model = dynamic_model, planner = global_cfs_planner, use_global_planner = False)

    # Init EgoCircle
    ego_circle = EgoCircle(goal_pos, dist = 0.1, radius = sensor_dist, safety_dist = 0.05)
    #use_ego_circle = use_pgap #False
    state, done = env.reset(), False
    planning_init_state = state[:2]
    infeasible_traj_count = 0
    cfs_planning_count = 0
    for t in range(max_steps):     
      env.display_start()
      # Collect robot information and obstacles information
      sensor_data = {}
      sensor_data['cartesian_sensor_est'] = {'pos':np.vstack(state[:2]), 'vel':np.vstack(state[2:4])}
      unsafe_obstacle_ids, unsafe_obstacles = env.find_unsafe_obstacles(sensor_dist)
      sensor_data['obstacle_sensor_est'] = {}
      for i, obs in enumerate(unsafe_obstacles):
        sensor_data['obstacle_sensor_est']['obs'+str(i)] = {'pos':np.vstack(obs[:2]), 'vel':np.vstack(obs[2:4])}
      
      # High level planner and CFS planner      
      if replanning_timer == replanning_cycle:
        if (use_pgap):
          # transform to laser data form
          ego_circle.parse_sensor_data(sensor_data)
          # build gaps
          possible_gaps = ego_circle.build_gaps(sensor_data)
          if (use_dgap):   
            intermedia_goal = ego_circle.select_dynamic_gap(sensor_data, possible_gaps)
          else:
            intermedia_goal = ego_circle.select_gap(sensor_data, possible_gaps)
          traj = global_planner.IntermediaGoalPlannerv2(dT, state[:4], intermedia_goal)
          # plot gaps
          env.display.gaps_at_loc(possible_gaps, state[:2]) 
        else:
          intermedia_goal, traj = global_planner.IntermediaGoalPlanner(dT, state[:4], goal_pos, sensor_data)  
        # CFS trajectory optimization        
        safe_traj = cfs_planner(dT, intermedia_goal, copy.deepcopy(traj), sensor_data)            
        replanning_timer = 0
        planning_init_state = state[:2]        
        is_pdb = True
        env.display.traj_at_loc(safe_traj, state[:2])
        cfs_planning_count += 1
        # check the feasibility of the trajectory
        for i in range(len(safe_traj) - 1):
          cur_waypoint = safe_traj[i]
          next_waypoint = safe_traj[i+1]
          if (abs(cur_waypoint[0]) > 1 or abs(cur_waypoint[1]) > 1):
            infeasible_traj_count += 1
            break
          elif (abs(cur_waypoint[0] - next_waypoint[0]) > 1 or abs(cur_waypoint[1] - next_waypoint[1]) > 1):
            infeasible_traj_count += 1
            break
      
      # Generate control signal  
      sensor_data['planning_init_state'] = np.vstack(planning_init_state)      
      next_traj_point = safe_traj[min(replanning_timer, safe_traj.shape[0]-1)]
      next_traj_point = np.vstack(next_traj_point.ravel())      
      action = controller(dT, sensor_data, next_traj_point, 2)
      replanning_timer += 1 
      
      #action = [0, 0.04]
      # Monitor and modify the unsafe control signal
      #action, is_safe, _, _ = safe_controller.get_safe_control(state[:4], unsafe_obstacles, fx, gx, action)
      is_safe = False
      #if (is_safe):
      #  print(f"control_action {control_action}, action {action}")
      action = np.vstack(action.ravel())      
      s_new, reward, done, info = env.step(dT, action, is_safe, unsafe_obstacle_ids) 
      episode_reward += reward        
      env.display_end()
      state = s_new
      if is_pdb:
        #pdb.set_trace()
        is_pdb = False
      
      if (done and reward == -500):          
        collision_num += 1      
      elif (done and reward == 2000):
        success_num += 1
        total_steps += env.cur_step
      elif (done):
        failure_num += 1
      
      if (done):              
        print(f"Train: episode_num {episode_num}, total_steps {total_steps}, cfs_planning_count {cfs_planning_count}, infeasible_count {infeasible_traj_count}, reward {episode_reward}, last state {state[:2]}")
        total_rewards.append(episode_reward)
        episode_reward = 0
        episode_num += 1
        state, done = env.reset(), False
        if (episode_num >= 100):
          print(f"avg_succ_steps {total_steps/success_num}, collision_num {collision_num}, success_num {success_num}, failure_num {failure_num}")
          break


if __name__ == '__main__':
    args = parser().parse_args()
    for i in range(10):
      main(display_name = args.display, use_pgap = args.use_pgap, use_dgap = args.use_dgap)


