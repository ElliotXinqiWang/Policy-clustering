from collections import OrderedDict
from enum import IntEnum
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Dict
import chex
from flax import struct
from queue import PriorityQueue


@struct.dataclass
class State:
    agent_pos: chex.Array
    goal_pos: chex.Array
    wall_map: chex.Array
    time: int
    terminal: bool
    info: Dict[str, chex.Array] = struct.field(pytree_node=True)


class Actions(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    
class grid_Reacher_agent:
    """
    A simple rule-based agent for the gridworld environment.
    
    """
    def __init__(self, env):
        self.env = env
        self.map_cache = {} # cache the distance map for each wall_map
        
    def get_action(self, State):
        """
        simply approaching the goal position
        """
        agent_pos = State.agent_pos
        goal_pos = State.goal_pos
        wall_map = State.wall_map
        time = State.time
        terminal = State.terminal
        info = State.info
        special_pos = info["special_pos"]
        special_reached = info["special_reached"]
        
        return self.distance_between_points(goal_pos, agent_pos, wall_map)
    
    def distance_between_points(self, point1, point2, wall_map):
        """
        Use BFS to find the shortest path from point2 to point1. Also return the action to take to reach the point.
        return inf if there is no path.
        """
        point1 = np.array(point1)
        point2 = np.array(point2)
        
        # check if the distance map is already in the cache
        if (wall_map.tobytes(), point1.tobytes()) in self.map_cache:
            distance_map, action_map = self.map_cache[(wall_map.tobytes(), point1.tobytes())]
            return distance_map[point2[0], point2[1]], action_map[point2[0], point2[1]]
        
        point_distance = PriorityQueue()
        action_map = np.zeros_like(wall_map) * 4 # record the action to take to reach the point
        distance_map = np.ones_like(wall_map) * np.inf
        point_distance.put((0, tuple(point1))) 
        distance_map[point1[0], point1[1]] = 0
        while not point_distance.empty():
            distance, current_point = point_distance.get()
            for action in range(4):
                new_point = self.reverse_move(current_point, action)
                new_y, new_x = new_point
                if wall_map[new_y, new_x] == 0 and distance_map[new_y, new_x] == np.inf:
                    point_distance.put((distance + 1, new_point))
                    distance_map[new_y, new_x] = distance + 1
                    action_map[new_y, new_x] = action
        self.map_cache[(wall_map.tobytes(), point1.tobytes())] = (distance_map, action_map)
        return distance_map[point2[0], point2[1]], action_map[point2[0], point2[1]]
    def distance_to_nearest(self, target_points, start_point, wall_map):
        """
        Use BFS to find the shortest path from start_point to one of the target_points. 
        Same as distance_between_points when target_points is a single point.
        """
        start_point = np.array(start_point)
        target_points = [np.array(point) for point in target_points]
        point_distance = PriorityQueue()
        action_map = np.zeros_like(wall_map) * 4 # record the action to take to reach the point
        distance_map = np.ones_like(wall_map) * np.inf
        for point in target_points:
            point_distance.put((0, tuple(point))) 
            distance_map[point[0], point[1]] = 0
        # check if the distance map is already in the cache
        if (wall_map.tobytes(), np.array(target_points).tobytes()) in self.map_cache:
            distance_map, action_map = self.map_cache[(wall_map.tobytes(), np.array(target_points).tobytes())]
            return distance_map[start_point[0], start_point[1]], action_map[start_point[0], start_point[1]]  
        
        while not point_distance.empty():
            distance, current_point = point_distance.get()
            for action in range(4):
                new_point = self.reverse_move(current_point, action)
                new_y, new_x = new_point
                if wall_map[new_y, new_x] == 0 and distance_map[new_y, new_x] == np.inf:
                    point_distance.put((distance + 1, new_point))
                    distance_map[new_y, new_x] = distance + 1
                    action_map[new_y, new_x] = action
        self.map_cache[(wall_map.tobytes(), np.array(target_points).tobytes())] = (distance_map, action_map)
        
        return distance_map[start_point[0], start_point[1]], action_map[start_point[0], start_point[1]]
                    
    def move(self, point, action):
        """
        Move the point according to the action
        """
        if action == 0:
            return (point[0] - 1, point[1])
        elif action == 1:
            return (point[0] + 1, point[1])
        elif action == 2:
            return (point[0], point[1] - 1)
        elif action == 3:
            return (point[0], point[1] + 1)
        else:
            return point
        
    def reverse_move(self, point, action):
        """
        Reverse the move according to the action
        """
        if action == 0:
            return (point[0] + 1, point[1])
        elif action == 1:
            return (point[0] - 1, point[1])
        elif action == 2:
            return (point[0], point[1] + 1)
        elif action == 3:
            return (point[0], point[1] - 1)
        else:
            return point
    
    def reset(self):
        self.map_cache = {}
    
class grid_Reacher_agent_good(grid_Reacher_agent):
    
    def get_action(self, State):
        """
        Try to get the two special positions and then approach the goal position.
        Simply move to the remaining special position if one of them is reachable.
        Else, approach the goal position.
        """
        agent_pos = State.agent_pos
        goal_pos = State.goal_pos
        wall_map = State.wall_map
        time = State.time
        terminal = State.terminal
        info = State.info
        special_pos = info["special_pos"]
        special_reached = info["special_reached"]
        special_pos_remaining = []
        for i in range(len(special_pos)):
            if not special_reached[i]:
                special_pos_remaining.append(special_pos[i])
        if len(special_pos_remaining) == 0:
            return self.distance_between_points(goal_pos, agent_pos, wall_map)
        else:
            # we don't want to pass the goal before reaching the special positions
            goal_wall_map = wall_map.copy()
            goal_wall_map = goal_wall_map.at[goal_pos[0], goal_pos[1]].set(True)
            dis_to_special, action_to_special = self.distance_to_nearest(special_pos_remaining, agent_pos, goal_wall_map)
            if dis_to_special < np.inf:
                return dis_to_special, action_to_special
            else:
                return self.distance_between_points(goal_pos, agent_pos, wall_map)
            
class grid_Reacher_agent_bad(grid_Reacher_agent):
    
    def get_action(self, State):
        """
        Try to avoid the two special positions and approach the goal position.
        """
        agent_pos = State.agent_pos
        goal_pos = State.goal_pos
        wall_map = State.wall_map
        time = State.time
        terminal = State.terminal
        info = State.info
        special_pos = info["special_pos"]
        special_reached = info["special_reached"]
        special_pos_remaining = []
        for i in range(len(special_pos)):
            if not special_reached[i]:
                special_pos_remaining.append(special_pos[i])
        special_wall_map = wall_map.copy()
        for pos in special_pos_remaining:
            special_wall_map = special_wall_map.at[pos[0], pos[1]].set(True)
        # print("current pos", agent_pos, "special pos", special_pos, "special_wall_map", special_wall_map)
        return self.distance_between_points(goal_pos, agent_pos, special_wall_map)
    
    
class MDP_takeball_agent:
    """
    A simple rule-based agent for the 9*9 gridworld.env.MDPtakeball environment.
    Simply return the action in the action_mode_maps.
    """
    take_ith_ball = jnp.array([[
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 2, 4, 1, 4, 1, 4, 4],
            [4, 0, 2, 2, 2, 2, 2, 2, 4],
            [4, 0, 2, 2, 2, 2, 2, 2, 4],
            [4, 0, 2, 2, 2, 2, 2, 2, 4],
            [4, 0, 0, 0, 0, 0, 0, 0, 4],
            [4, 0, 0, 0, 0, 0, 0, 0, 4],
            [4, 0, 0, 0, 0, 0, 0, 0, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4]
        ],
                              [
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 3, 4, 2, 4, 1, 4, 4],
            [4, 3, 3, 0, 2, 2, 2, 2, 4],
            [4, 3, 3, 0, 2, 2, 2, 2, 4],
            [4, 3, 3, 0, 2, 2, 2, 2, 4],
            [4, 0, 0, 0, 0, 0, 0, 0, 4],
            [4, 0, 0, 0, 0, 0, 0, 0, 4],
            [4, 0, 0, 0, 0, 0, 0, 0, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4]
        ],
                              [
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 1, 4, 3, 4, 2, 4, 4],
            [4, 3, 3, 3, 3, 0, 2, 2, 4],
            [4, 3, 3, 3, 3, 0, 2, 2, 4],
            [4, 3, 3, 3, 3, 0, 2, 2, 4],
            [4, 0, 0, 0, 0, 0, 0, 0, 4],
            [4, 0, 0, 0, 0, 0, 0, 0, 4],
            [4, 0, 0, 0, 0, 0, 0, 0, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4]
        ],
                              [
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 1, 4, 1, 4, 3, 4, 4],
            [4, 3, 3, 3, 3, 3, 3, 0, 4],
            [4, 3, 3, 3, 3, 3, 3, 0, 4],
            [4, 3, 3, 3, 3, 3, 3, 0, 4],
            [4, 0, 0, 0, 0, 0, 0, 0, 4],
            [4, 0, 0, 0, 0, 0, 0, 0, 4],
            [4, 0, 0, 0, 0, 0, 0, 0, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4]
        ]])
    to_the_goal = jnp.array([
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 1, 1, 1, 1, 1, 1, 1, 4],
            [4, 3, 1, 1, 1, 1, 1, 1, 4],
            [4, 3, 3, 1, 1, 1, 1, 1, 4],
            [4, 3, 3, 3, 1, 1, 1, 1, 4],
            [4, 3, 3, 3, 3, 1, 1, 1, 4],
            [4, 3, 3, 3, 3, 3, 1, 1, 4],
            [4, 3, 3, 3, 3, 3, 3, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4]
        ])
    def __init__(self, env, mode: int=0):
        self.env = env
        self.target_ball_idx = int(mode) 
    
    def get_action(self, State):
        to_the_goal_action = self.to_the_goal[State.agent_pos[0], State.agent_pos[1]]
        target_ball = jnp.where(self.target_ball_idx == State.info["balls_idx"], size=1)[0][0]
        get_the_ball_action = self.take_ith_ball[target_ball, State.agent_pos[0], State.agent_pos[1]]
        
        final_action = jax.lax.cond(
            State.info["ball_got"] == -1,
            lambda _: get_the_ball_action,
            lambda _: to_the_goal_action,
            operand=None
        )
        return final_action
    
class MDP_reacher_agent:
    """
    A simple rule-based agent for the 9*9 gridworld.env.MDPGridworld environment.
    Simply return the action in the action_mode_maps.
    """
    action_mode_maps = {"rightfirst": jnp.array([
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 3, 3, 3, 3, 3, 3, 1, 4],
            [4, 3, 3, 3, 3, 3, 3, 1, 4],
            [4, 3, 3, 3, 3, 3, 3, 1, 4],
            [4, 3, 3, 3, 3, 3, 3, 1, 4],
            [4, 3, 3, 3, 3, 3, 3, 1, 4],
            [4, 3, 3, 3, 3, 3, 3, 1, 4],
            [4, 3, 3, 3, 3, 3, 3, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4]
        ]),
        "downfirst": jnp.array([
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 1, 1, 1, 1, 1, 1, 1, 4],
            [4, 1, 1, 1, 1, 1, 1, 1, 4],
            [4, 1, 1, 1, 1, 1, 1, 1, 4],
            [4, 1, 1, 1, 1, 1, 1, 1, 4],
            [4, 1, 1, 1, 1, 1, 1, 1, 4],
            [4, 1, 1, 1, 1, 1, 1, 1, 4],
            [4, 3, 3, 3, 3, 3, 3, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4]
        ]),
        "balanced": jnp.array([
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 1, 1, 1, 1, 1, 1, 1, 4],
            [4, 3, 1, 1, 1, 1, 1, 1, 4],
            [4, 3, 3, 1, 1, 1, 1, 1, 4],
            [4, 3, 3, 3, 1, 1, 1, 1, 4],
            [4, 3, 3, 3, 3, 1, 1, 1, 4],
            [4, 3, 3, 3, 3, 3, 1, 1, 4],
            [4, 3, 3, 3, 3, 3, 3, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4]
        ]),
        "zigzag1": jnp.array([
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 3, 1, 3, 1, 3, 1, 1, 4],
            [4, 1, 3, 1, 3, 1, 3, 1, 4],
            [4, 3, 1, 3, 1, 3, 1, 1, 4],
            [4, 1, 3, 1, 3, 1, 3, 1, 4],
            [4, 3, 1, 3, 1, 3, 1, 1, 4],
            [4, 1, 3, 1, 3, 1, 3, 1, 4],
            [4, 3, 3, 3, 3, 3, 3, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4]
        ]),
        "zigzag2": jnp.array([
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 1, 3, 1, 3, 1, 3, 1, 4],
            [4, 3, 1, 3, 1, 3, 1, 1, 4],
            [4, 1, 3, 1, 3, 1, 3, 1, 4],
            [4, 3, 1, 3, 1, 3, 1, 1, 4],
            [4, 1, 3, 1, 3, 1, 3, 1, 4],
            [4, 3, 1, 3, 1, 3, 1, 1, 4],
            [4, 3, 3, 3, 3, 3, 3, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4]
        ]),
    }
    
    
    def __init__(self, env, mode: str = "balanced"):
        self.env = env
        self.action_mode_maps = self.action_mode_maps[mode]
    
    def get_action(self, State):
        return self.action_mode_maps[State.agent_pos[0], State.agent_pos[1]]
   