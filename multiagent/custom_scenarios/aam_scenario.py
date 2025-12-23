import numpy as np
from multiagent.scenario import BaseScenario
from multiagent.core import Entity, Agent, Landmark
from multiagent.custom_scenarios.base_aam_scenario import AAMBaseScenario
import argparse
from multiagent.core import EntityDynamicsType, World, Agent, Landmark, Entity, Wall
from multiagent.scenario import BaseScenario
from multiagent.config import UnicycleVehicleConfig, DoubleIntegratorConfig, AirTaxiConfig, RewardWeightConfig, RewardBinaryConfig
from numpy import ndarray as arr
from scipy import sparse


entity_mapping = {'agent': 0, 'landmark': 1, 'obstacle':2, 'wall':3}

def direction_alignment_error(theta, theta_ref):
    """Smallest absolute angle difference in [0, pi]."""
    d = (theta - theta_ref + np.pi) % (2*np.pi) - np.pi
    return abs(d)

def cross_track_error(agent_pos, agent_heading, goal_pos):
    """
    Signed cross-track distance from agent_pos to the ray starting at goal_pos
    and pointing along agent_heading.
    Returns |cross-track| in [0,1] approx by soft-bounding.
    """
    # Ray direction
    ray = np.array([np.cos(agent_heading), np.sin(agent_heading)])
    v = agent_pos - goal_pos
    # cross-track magnitude = |v x ray| (2D cross product magnitude)
    c = abs(v[0]*ray[1] - v[1]*ray[0])
    # Soft-normalize: treat >1 as 1
    return float(np.clip(c, 0, 1))

def get_relative_position_from_reference(p, pref, heading_ref):
    """Rotate (p - pref) into the frame aligned with heading_ref."""
    d = p - pref
    c, s = np.cos(heading_ref), np.sin(heading_ref)
    R = np.array([[c, s], [-s, c]])  # frame where +x is along heading_ref
    return R @ d

def double_integrator_velocity_error_from_magnetic_field_reference(state, goal, radius=0.2):
    """
    LS-MARL uses a heading-aware penalty; for AirTaxi we fallback to distance
    if we don’t have velocity components. This returns a smooth penalty
    that’s ~0 near the goal and grows with distance.
    """
    p = state.p_pos
    g = goal.state.p_pos
    d = np.linalg.norm(p - g)
    if d <= radius:
        return 0.0
    # simple smooth ramp starting after radius
    return d - radius


def get_thetas(poses):
    # compute angle (0,2pi) from horizontal
    thetas = [None] * len(poses)
    for i in range(len(poses)):
        # (y,x)
        thetas[i] = find_angle(poses[i])
    return thetas


def find_angle(pose):
    # compute angle from horizontal
    angle = np.arctan2(pose[1], pose[0])
    if angle < 0:
        angle += 2 * np.pi
    return angle

# @staticmethod
# @jit(nopython=True)
def get_rotated_position_from_relative(relative_position: np.ndarray,
    reference_heading: float) -> np.ndarray:
    # returns relative position from the reference state.
    assert relative_position.shape == (2,), "relative_position should be a 2D array."
    rot_matrix = np.array([[np.cos(reference_heading), np.sin(reference_heading)], [-np.sin(reference_heading), np.cos(reference_heading)]])
    relative_position_rotated = np.dot(rot_matrix, relative_position)
    return relative_position_rotated



class AamScenario(AAMBaseScenario):
    """
    Corridor + fairness + rotation-invariant scenario.
    This is the non-safety version. Safety gets added in SafeAamScenario.
    """
    # def __init__(self, args):
    #     super().__init__(args)

    # ===============================================================
    # 1. WORLD + AGENT INITIALIZATION
    # ===============================================================

    # def reset_world(self, world):
    #     """
    #     Calls the corridor placement + resets velocities + flags.
    #     """
    #     # Place agents + landmarks according to corridor geometry
    #     self.random_scenario(world)

    #     # Reset agent state vectors
    #     for agent in world.agents:
    #         agent.state.c = np.zeros(world.dim_c)
    #         agent.done = False

    #     # Cache initial distances for fairness computations
    #     self._cache_initial_spacing(world)

    # # ===============================================================
    # # 2. SCENARIO LAYOUT (override this in TrainScenario/EvalScenario)
    # # ===============================================================
    # def random_scenario(self, world):
    #     """
    #     Default corridor placement — can be overridden by TrainScenario & EvalScenario.
    #     """
    #     tube_x0 = -0.8 * self.world_size
    #     tube_x1 =  0.8 * self.world_size
    #     tube_ymin = -self.corridor_width / 2
    #     tube_ymax =  self.corridor_width / 2

    #     # Evenly space agents across corridor width
    #     ys = np.linspace(tube_ymin * 0.7, tube_ymax * 0.7, self.num_agents)

    #     for i, agent in enumerate(world.agents):
    #         agent.state.p_pos = np.array([tube_x0, ys[i]])
    #         agent.state.reset_velocity(theta=0)

    #     # Each agent’s single "goal" landmark is at right-end of tube
    #     for i, landmark in enumerate(world.landmarks):
    #         landmark.state.p_pos = np.array([tube_x1, ys[i]])
    #         landmark.state.stop()
    #         landmark.heading = 0
    #         landmark.speed = self.goal_speed_max

    # # ===============================================================
    # # 3. REWARD FUNCTION (NO SAFETY TERMS HERE)
    # # ===============================================================
    # def reward(self, agent: Agent, world):
    #     """
    #     Full corridor reward (NO SAFETY). Includes:
    #     - distance to corresponding goal
    #     - collision penalty (non-safety)
    #     - fairness penalty (spacing)
    #     - staying inside corridor
    #     """
    #     rew = 0.0

    #     agent_index = world.agents.index(agent)
    #     goal = world.landmarks[agent_index]

    #     # -----------------------------------------------------------
    #     # Distance-to-goal reward (encourage movement rightward)
    #     # -----------------------------------------------------------
    #     dist = np.linalg.norm(agent.state.p_pos - goal.state.p_pos)
    #     rew -= dist

    #     # -----------------------------------------------------------
    #     # Collision penalty (non-safety version)
    #     # -----------------------------------------------------------
    #     if self.use_collisions:
    #         for other in world.agents:
    #             if other is agent:
    #                 continue
    #             if self._is_collision(agent, other):
    #                 rew -= self.collision_penalty

    #     # -----------------------------------------------------------
    #     # Corridor deviation penalty (encourage staying inside tube)
    #     # -----------------------------------------------------------
    #     y = agent.state.p_pos[1]
    #     tube_half_w = self.corridor_width / 2
    #     if abs(y) > tube_half_w:
    #         rew -= self.out_of_corridor_penalty * (abs(y) - tube_half_w)

    #     # -----------------------------------------------------------
    #     # Fairness / spacing reward (distance variance)
    #     # -----------------------------------------------------------
    #     if self.use_fairness:
    #         rew += self._fairness_reward(agent, world, agent_index)

    #     return rew

    # # ===============================================================
    # # 4. FAIRNESS (SPACING) REWARD
    # # ===============================================================
    # def _cache_initial_spacing(self, world):
    #     """Cache initial y ordering to track spacing deviation."""
    #     self.initial_positions = np.array([a.state.p_pos.copy() for a in world.agents])
    #     self.sorted_indices = np.argsort(self.initial_positions[:, 1])

    # def _fairness_reward(self, agent, world, agent_index):
    #     """
    #     Penalizes deviation from initial y-spacing formation.
    #     """
    #     current_positions = np.array([a.state.p_pos for a in world.agents])
    #     sorted_now = np.argsort(current_positions[:, 1])

    #     # rank difference penalty (how far from original row)
    #     initial_rank = np.where(self.sorted_indices == agent_index)[0][0]
    #     current_rank  = np.where(sorted_now == agent_index)[0][0]

    #     rank_diff = abs(initial_rank - current_rank)
    #     return -self.fairness_penalty * rank_diff

    # # ===============================================================
    # # 5. DONE FUNCTION
    # # ===============================================================
    # def done(self, agent, world):
    #     """
    #     Episode ends for agent if:
    #     - reaches goal (x beyond tube end)
    #     - leaves allowed world boundary
    #     - (time handled by env, not here)
    #     """
    #     x = agent.state.p_pos[0]
    #     y = agent.state.p_pos[1]

    #     # reach goal
    #     if x > self.goal_x_threshold:
    #         return True

    #     # world boundary check
    #     if abs(x) > self.world_size or abs(y) > self.world_size:
    #         return True

    #     return False

    # # ===============================================================
    # # 6. OBSERVATION (ROTATION INVARIANT)
    # # ===============================================================
    # def observation(self, agent, world):
    #     """
    #     Rotation-invariant observation:
    #     Agent-centric frame aligned with corridor axis (x direction).
    #     Observes relative positions of:
    #         - itself
    #         - other agents
    #         - its goal landmark
    #     """
    #     agent_pos = agent.state.p_pos
    #     agent_idx = world.agents.index(agent)

    #     # relative positions of other agents
    #     rel_agents = []
    #     for other in world.agents:
    #         rel = other.state.p_pos - agent_pos
    #         rel_agents.append(rel)

    #     # relative goal
    #     goal = world.landmarks[agent_idx]
    #     rel_goal = goal.state.p_pos - agent_pos

    #     # stack into obs vector
    #     obs = np.concatenate([
    #         np.array([agent.state.speed, agent.state.theta]),
    #         rel_goal,
    #         np.array(rel_agents).flatten(),
    #     ])

    #     return obs

    # # ===============================================================
    # # 7. GRAPH OBSERVATION (GNNS)
    # # ===============================================================
    # def graph_observation(self, agent, world):
    #     """
    #     Node features for GNN encoder:
    #         [relative_pos_x, relative_pos_y, is_agent, is_goal]
    #     """
    #     agent_pos = agent.state.p_pos
    #     nodes = []

    #     for a in world.agents:
    #         nodes.append([
    #             a.state.p_pos[0] - agent_pos[0],
    #             a.state.p_pos[1] - agent_pos[1],
    #             1.0,  # agent flag
    #             0.0   # landmark flag
    #         ])

    #     for l in world.landmarks:
    #         nodes.append([
    #             l.state.p_pos[0] - agent_pos[0],
    #             l.state.p_pos[1] - agent_pos[1],
    #             0.0,
    #             1.0,  # landmark flag
    #         ])

    #     return np.array(nodes)

    # # ===============================================================
    # # 8. INFO CALLBACK
    # # ===============================================================
    # def info_callback(self, agent, world):
    #     agent_index = world.agents.index(agent)
    #     goal = world.landmarks[agent_index]

    #     dist_to_goal = np.linalg.norm(agent.state.p_pos - goal.state.p_pos)
    #     return {"dist_to_goal": dist_to_goal}

    # # ===============================================================
    # # 9. UTILS
    # # ===============================================================
    # def _is_collision(self, a1, a2):
    #     """Simple circle collision."""
    #     dist = np.linalg.norm(a1.state.p_pos - a2.state.p_pos)
    #     return dist < (a1.size + a2.size)

    # ===============================================================
    # 10. REFACTORING
    # ===============================================================

    def get_aspect_ratio_for_scenario(self) -> float:
        return 1.0

    def make_world(self, args:argparse.Namespace) -> World:
        """
            Parameters in args
            ––––––––––––––––––
            • num_agents: int
                Number of agents in the environment
                NOTE: this is equal to the number of goal positions
            • num_obstacles: int
                Number of num_obstacles obstacles
            • collaborative: bool
                If True then reward for all agents is sum(reward_i)
                If False then reward for each agent is what it gets individually
            • max_speed: Optional[float]
                Maximum speed for agents
                NOTE: Even if this is None, the max speed achieved in discrete 
                action space is 2, so might as well put it as 2 in experiments
                TODO: make list for this and add this in the state
            • collision_rew: float
                The reward to be negated for collisions with other agents and 
                obstacles
            • goal_rew: float
                The reward to be added if agent reaches the goal
            • min_dist_thresh: float
                The minimum distance threshold to classify whether agent has 
                reached the goal or not
            • use_dones: bool
                Whether we want to use the 'done=True' when agent has reached 
                the goal or just return False like the `simple.py` or 
                `simple_spread.py`
            • episode_length: int
                Episode length after which environment is technically reset()
                This determines when `done=True` for done_callback
            • graph_feat_type: str
                The method in which the node/edge features are encoded
                Choices: ['global', 'relative']
                    If 'global': 
                        • node features are global [pos, vel, goal, entity-type]
                        • edge features are relative distances (just magnitude)
                        • 
                    If 'relative':
                        • TODO decide how to encode stuff

            • max_edge_dist: float
                Maximum distance to consider to connect the nodes in the graph
        """
        # pull params from args
        if not hasattr(self, 'world_size'):
            self.world_size = args.world_size
        self.args = args
        self.world_aspect_ratio = self.get_aspect_ratio_for_scenario()
        self.num_agents = args.num_agents
        self.num_scripted_agents = args.num_scripted_agents
        self.num_obstacles = args.num_obstacles
        self.collaborative = args.collaborative
        self.max_speed = args.max_speed
        self.collision_rew = args.collision_rew
        self.formation_rew = args.formation_rew
        self.goal_rew = args.goal_rew
        self.min_reward = RewardWeightConfig.MIN_REWARD
        self.max_reward = RewardWeightConfig.MAX_REWARD
        self.conflict_rew_scaled = RewardWeightConfig.SAFETY_VIOLATION
        self.conflict_value_rew_scaled = RewardWeightConfig.HJ_VALUE
        self.multiple_engagement_rew_scaled = RewardWeightConfig.POTENTIAL_CONFLICT
        self.diff_from_filtered_action_rew_scaled = RewardWeightConfig.DIFF_FROM_FILTERED_ACTION

        # dummy_pos = np.array([1.0, 0.0])
        # get_rotated_position_from_relative(dummy_pos, 0.0)
        self.use_dones = args.use_dones
        self.episode_length = args.episode_length
        # used for curriculum learning (in training)
        self.num_total_episode = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
        if args.render_episodes is not None:
            self.num_total_episode = args.render_episodes

        self.target_radius = 0.5  # fixing the target radius for now
        self.ideal_theta_separation = (
            2 * np.pi
        ) / self.num_agents  # ideal theta difference between two agents

        # fairness args
        self.fair_wt = args.fair_wt
        self.fair_rew = args.fair_rew

        self.formation_type = args.formation_type
        self.steps_in_corridor = np.zeros(self.num_agents)
        # create heatmap matrix to determine the goal agent pairs
        self.goal_reached = np.full(self.num_agents, -1)
        self.wrong_goal_reached = np.zeros(self.num_agents)
        self.goal_matched = np.zeros(self.num_agents)

        self.goal_tracker = np.full(self.num_agents, -1)## 	keeps track of which goal each agent goes to using self.goal_tracker[agent.id] = self.goal_match_index[agent.id]

        self.conformance_percent = np.zeros(self.num_agents)

        self.delta_spacing = []

        self.spacing_violation = np.zeros(self.num_agents)
        if args.dynamics_type == 'unicycle_vehicle':
            self.dynamics_type = EntityDynamicsType.UnicycleVehicleXY
            self.config_class = UnicycleVehicleConfig
            self.min_turn_radius = 0.5 * (UnicycleVehicleConfig.V_MAX + UnicycleVehicleConfig.V_MIN) / UnicycleVehicleConfig.ANGULAR_RATE_MAX

        elif args.dynamics_type == 'double_integrator':
            self.dynamics_type = EntityDynamicsType.DoubleIntegratorXY
            self.config_class = DoubleIntegratorConfig
            self.min_turn_radius = 0.0
        
        elif args.dynamics_type == 'air_taxi':
            self.dynamics_type = EntityDynamicsType.AirTaxiXY
            self.config_class = AirTaxiConfig
            self.min_turn_radius = 0.0
        else:
            raise NotImplementedError
        self.coordination_range = self.config_class.COORDINATION_RANGE
        self.min_dist_thresh = self.config_class.DISTANCE_TO_GOAL_THRESHOLD
        self.separation_distance = self.config_class.COLLISION_DISTANCE

        self.phase_reached = np.zeros(self.num_agents)  ## keeps track of which phase each agent is in when it first enters it

        self.phase_reward_cooldown_steps = self.episode_length                   # steps to cool down 0->1 reward

        # scripted agent dynamics are fixed to double integrator for now (as it was originally)
        scripted_agent_dynamics_type = EntityDynamicsType.DoubleIntegratorXY

        # needed when scenario is used in evaluation.
        self.curriculum_ratio = 1.0
        self.max_edge_dist = self.coordination_range
        ####################
        world = World(dynamics_type=self.dynamics_type, all_args = args)
        # graph related attributes
        world.cache_dists = True # cache distance between all entities
        world.graph_mode = True
        world.graph_feat_type = getattr(args, "graph_feat_type", "relative")
        world.world_length = args.episode_length
        # metrics to keep track of
        world.current_time_step = 0
        # to track time required to reach goal
        world.times_required =  np.full(self.num_agents, -1)
        world.dists_to_goal =  np.full(self.num_agents, -1)
        # set any world properties
        world.dim_c = 2
        self.num_landmarks = args.num_landmarks # no. of goals need not equal to no. of agents
        num_scripted_agents_goals = self.num_scripted_agents
        world.collaborative = args.collaborative

        #############
        ## determine the number of actions from arguments
        world.total_actions = args.total_actions
        #############

        # add agents
        global_id = 0
        world.agents = [Agent(self.dynamics_type) for i in range(self.num_agents)]
        world.scripted_agents = [Agent(scripted_agent_dynamics_type) for _ in range(self.num_scripted_agents)]
        for i, agent in enumerate(world.agents + world.scripted_agents):
            agent.id = i
            agent.name = f'agent {i}'
            agent.collide = True
            agent.silent = True
            agent.global_id = global_id
            global_id += 1
            # NOTE not changing size of agent because of some edge cases; 
            # TODO have to change this later
            # agent.size = 0.15
            agent.max_speed = self.max_speed
        # add landmarks (goals)
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        world.scripted_agents_goals = [Landmark() for i in range(num_scripted_agents_goals)]
        for i, landmark in enumerate(world.landmarks):
            landmark.id = i
            landmark.name = f'landmark {i}'
            landmark.collide = False
            landmark.movable = False
            landmark.global_id = global_id
            global_id += 1
        # add obstacles
        world.obstacles = [Landmark() for i in range(self.num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = f'obstacle {i}'
            obstacle.collide = True
            obstacle.movable = False
            obstacle.global_id = global_id
            global_id += 1
        ## add wall
        # num obstacles per wall is twice the length of the wall
        wall_length = np.random.uniform(0.2, 0.4)
        self.wall_length = wall_length * self.world_size/4

        self.num_walls = args.num_walls
        world.walls = [Wall() for i in range(self.num_walls)]
        for i, wall in enumerate(world.walls):
            wall.id = i
            wall.name = f'wall {i}'
            wall.collide = True
            wall.movable = False
            wall.global_id = global_id
            global_id += 1

            
        self.zeroshift = args.zeroshift
        self.reset_world(world)
        world.world_size = self.world_size
        world.world_aspect_ratio = self.world_aspect_ratio

        if hasattr(self, 'with_background'):
            world.with_background = self.with_background
        else:
            world.with_background = False
        return world
        self.prev_goal_dist = np.full(self.num_agents, np.inf, dtype=np.float32)

    def reset_world(self, world:World, num_current_episode: int = 0) -> None:
        # print("RESET WORLD")
        # metrics to keep track of
        world.current_time_step = 0
        world.simulation_time = 0.0
        # to track time required to reach goal
        world.times_required = np.full(self.num_agents, -1)
        world.dists_to_goal =  np.full(self.num_agents, -1)

        # track distance left to the goal
        world.dist_left_to_goal =  np.full(self.num_agents, -1)
        # number of times agents collide with stuff
        world.num_obstacle_collisions = np.zeros(self.num_agents)
        world.num_goal_collisions = np.zeros(self.num_agents)
        world.num_agent_collisions = np.zeros(self.num_agents)
        world.agent_dist_traveled = np.zeros(self.num_agents)

        self.goal_match_index = np.arange(self.num_agents)
        self.goal_history = np.full(self.num_agents, -1)
        self.goal_reached =  np.full(self.num_agents, -1)

        self.goal_tracker = np.full(self.num_agents, -1)
        self.conformance_percent = np.zeros(self.num_agents)
        self.delta_spacing =[]
        self.spacing_violation = np.zeros(self.num_agents)
        self.steps_in_corridor = np.zeros(self.num_agents)

        self.agent_dist_traveled = np.zeros(self.num_agents)
        self.agent_time_taken = np.zeros(self.num_agents)
        wall_length = np.random.uniform(0.2, 0.8)
        self.wall_length = wall_length * self.world_size/4

        self.phase_reached = np.zeros(self.num_agents)
        self.entry_reward_cooldown = np.zeros(self.num_agents, dtype=np.int32)
        # Store previous longitudinal position (s) for progress reward
        self.prev_proj = np.zeros(self.num_agents, dtype=np.float32)

        #################### set colours ####################
        # set colours for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.85, 0.35, 0.35])
            if i%4 == 0:
                agent.color = np.array([0.85, 0.35, 0.35])
            elif i%4 == 1:
                agent.color = np.array([0.35, 0.85, 0.35])
            elif i%4 == 2:
                agent.color = np.array([0.35, 0.35, 0.85])
            else:
                agent.color = np.array([0.85, 0.85, 0.25])
            # if i == 0:
            # 	agent.color = np.array([0.15, 0.75, 0.65])
            agent.state.p_dist = 0.0
            agent.state.time = 0.0
        # set colours for scripted agents
        for i, agent in enumerate(world.scripted_agents):
            agent.color = np.array([0.15, 0.15, 0.15])
        # set colours for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i%4 == 0:
                landmark.color = np.array([0.85, 0.35, 0.35])
            elif i%4 == 1:
                landmark.color = np.array([0.35, 0.85, 0.35])
            elif i%4 == 2:
                landmark.color =  np.array([0.35, 0.35, 0.85])
            else:
                landmark.color = np.array([0.85, 0.85, 0.25])
            # if i == 0:
            # 	landmark.color = np.array([0.15, 0.75, 0.65])
        # set colours for scripted agents goals
        for i, landmark in enumerate(world.scripted_agents_goals):
            landmark.color = np.array([0.15, 0.95, 0.15])
        # set colours for obstacles
        for i, obstacle in enumerate(world.obstacles):
            obstacle.color = np.array([0.25, 0.25, 0.25])
        # set colours for wall obstacles
        # for i, wall_obstacle in enumerate(world.wall_obstacles):
        # 	wall_obstacle.color = np.array([0.25, 0.25, 0.25])
        #####################################################
        # self.update_curriculum(world, num_current_episode)
        self.random_scenario(world)
        self.initialize_min_time_distance_graph(world)
        self.initialize_landmarks_group_reached_goal(world)

    def update_graph(self, world:World):
        """
            Construct a graph from the cached distances.
            Nodes are entities in the environment
            Edges are constructed by thresholding distances
        """
        dists = world.cached_dist_mag
        # just connect the ones which are within connection 
        # distance and do not connect to itself
        connect = np.array((dists <= self.max_edge_dist) * \
                            (dists > 0)).astype(int)
        sparse_connect = sparse.csr_matrix(connect)
        sparse_connect = sparse_connect.tocoo()
        row, col = sparse_connect.row, sparse_connect.col
        edge_list = np.stack([row, col])
        world.edge_list = edge_list
        if world.graph_feat_type == 'global':
            world.edge_weight = dists[row, col]
        elif world.graph_feat_type == 'relative':
            world.edge_weight = dists[row, col]
    
    def update_curriculum(self, world:World, num_current_episode:int) -> None:

        """ Update the curriculum learning parameters if necessary."""
        # print(f"Current Episode: {num_current_episode}")
        # print(f"Total Episodes: {self.num_total_episode}")
        self.curriculum_ratio = np.clip(num_current_episode / self.num_total_episode, 0.1, 1.0)
        # print(f"Curriculum Ratio: {self.curriculum_ratio}")
        ## update collision penalty
        self.collision_rew = self.args.collision_rew * self.curriculum_ratio
        # print(f"Collision Reward: {self.collision_rew}")
        ## update formation reward
        self.formation_rew = self.args.formation_rew * self.curriculum_ratio

        ## update fairness reward
        self.fair_rew = self.args.fair_rew * self.curriculum_ratio

    def collect_goal_info(self, world):
        goal_pos =  np.zeros((self.num_agents, 2)) # create a zero vector with the size of the number of goal and positions of dim 2
        count = 0
        for goal in world.landmarks:
            goal_pos[count]= goal.state.p_pos
            count +=1
        return goal_pos

    def graph_observation(self, agent:Agent, world:World) -> tuple[arr, arr]:
        """
            FIXME: Take care of the case where edge_list is empty
            Returns: [node features, adjacency matrix]
            • Node features (num_entities, num_node_feats):
                If `global`: 
                    • node features are global [pos, vel, goal, entity-type]
                    • edge features are relative distances (just magnitude)
                    NOTE: for `landmarks` and `obstacles` the `goal` is 
                            the same as its position
                If `relative`:
                    • node features are relative [pos, vel, goal, entity-type] to ego agents
                    • edge features are relative distances (just magnitude)
                    NOTE: for `landmarks` and `obstacles` the `goal` is 
                            the same as its position
            • Adjacency Matrix (num_entities, num_entities)
                NOTE: using the distance matrix, need to do some post-processing
                If `global`:
                    • All close-by entities are connectd together
                If `relative`:
                    • Only entities close to the ego-agent are connected
            
        """

        # node observations
        node_obs = []
        fairness_param = 0.0

        if world.graph_feat_type == 'global':
            for i, entity in enumerate(world.entities):

                node_obs_i = self._get_entity_feat_global(entity, world)
                node_obs.append(node_obs_i)

        elif world.graph_feat_type == 'relative':
            for i, entity in enumerate(world.entities):

                node_obs_i = self._get_entity_feat_relative(agent, entity, world, fairness_param)
                node_obs.append(node_obs_i)

        node_obs = np.array(node_obs)
        adj = world.cached_dist_mag

        disconnected_mask = []
        # for agent entity, disconnect if it is done or not departed
        for entity in world.agents:
            disconnected = entity.status
            # if entity.status:
            # 	print("agent_done", entity.id, disconnected)
            disconnected_mask.append(disconnected)

        # for landmark agent, disconnect if it is reached by the agent.
        for (i_landmark, landmark) in enumerate(world.landmarks):
            # landmark_agent_id = i_landmark % self.num_agents
            # landmark_order = i_landmark // self.num_agents
            # landmark_done = self.reached_goal[landmark_agent_id] > landmark_order
            ## use goal_tracker to remove landmarks that are reached
            landmark_done = np.any(self.goal_tracker == landmark.id)
            # if landmark_done:
                # print("landmark_done",landmark.id)
            disconnected_mask.append(landmark_done)
            # print("landmark_done",landmark.id)
        # print("disconnected_mask",disconnected_mask)
        adj[disconnected_mask, :] = 0   # Mask rows for done agents
        adj[:, disconnected_mask] = 0   # Mask columns for done agents
        return node_obs, adj

    def collect_dist(self, world):
        """
        This function collects the distances of all agents at once to reduce further computations of the reward
        input: world and agent information
        output: mean distance, standard deviation of distance, and positions of agents
        """
        agent_dist = np.array([agent.state.p_dist for agent in world.agents])  # Collect distances
        agent_pos = np.array([agent.state.p_pos for agent in world.agents])  # Collect positions

        mean_dist = np.mean(agent_dist)
        std_dev_dist = np.std(agent_dist)
        
        return mean_dist, std_dev_dist, agent_pos

    def _get_entity_feat_global(self, entity:Entity, world:World) -> arr:
        """
            Returns: ([velocity, position, goal_pos, entity_type])
            in global coords for the given entity
        """
        pos = entity.state.p_pos
        vel = entity.state.p_vel
        if 'agent' in entity.name:
            goal_pos = world.get_entity('landmark', self.goal_match_index[entity.id]).state.p_pos
            entity_type = entity_mapping['agent']
        elif 'landmark' in entity.name:
            goal_pos = pos
            entity_type = entity_mapping['landmark']
        elif 'obstacle' in entity.name:
            goal_pos = pos
            entity_type = entity_mapping['obstacle']
        else:
            raise ValueError(f'{entity.name} not supported')

        return np.hstack([vel, pos, goal_pos, entity_type])

    def _get_entity_feat_relative(self, agent: Agent, entity: Entity, world: World, fairness_param: np.ndarray) -> arr:
        """
        Returns rotation-invariant node features for `entity` relative to `agent`.

        Agents/Landmarks/Obstacles:
            [rel_vel(2), rel_pos(2), rel_goal_pos(2), goal_occupied(1), entity_type(1)]

        Walls:
            [rel_vel(2), rel_pos(2), rel_goal_pos(2), goal_occupied(1),
            goal_history(1), wall_o_corner(2), wall_d_corner(2), entity_type(1)]
        """
        # --- Ego (reference) state ---
        agent_pos = np.asarray(agent.state.p_pos, dtype=np.float32)
        agent_vel = np.asarray(agent.state.p_vel, dtype=np.float32)
        agent_heading = float(getattr(agent.state, "theta", getattr(agent.state, "p_ang", 0.0)))

        # --- Entity relative vectors in WORLD frame ---
        entity_pos_world = np.asarray(entity.state.p_pos, dtype=np.float32)
        entity_vel_world = np.asarray(entity.state.p_vel, dtype=np.float32)
        rel_pos_world = entity_pos_world - agent_pos
        rel_vel_world = entity_vel_world - agent_vel

        # --- Rotate into ego (agent) frame ---
        rel_pos = get_rotated_position_from_relative(rel_pos_world, agent_heading).astype(np.float32)
        rel_vel = get_rotated_position_from_relative(rel_vel_world, agent_heading).astype(np.float32)

        if 'agent' in entity.name:
            # Each agent's goal is the matched landmark pose
            # goal_pos_world = np.asarray(self.landmark_poses[self.goal_match_index[entity.id]], dtype=np.float32)
            ## change the goal to the exit position of the corridor
            goal_pos_world = np.asarray(world.tube_params['exit'], dtype=np.float32)
            rel_goal_pos_world = goal_pos_world - agent_pos
            rel_goal_pos = get_rotated_position_from_relative(rel_goal_pos_world, agent_heading).astype(np.float32)

            entity_type = np.array([entity_mapping['agent']], dtype=np.float32)

            return np.hstack([rel_vel, rel_pos, rel_goal_pos, entity_type]).astype(np.float32)

        elif 'landmark' in entity.name:
            # For landmarks, the "goal" is its own position relative to ego
            rel_goal_pos = rel_pos.copy()
            entity_type = np.array([entity_mapping['landmark']], dtype=np.float32)

            return np.hstack([rel_vel, rel_pos, rel_goal_pos, entity_type]).astype(np.float32)

        elif 'obstacle' in entity.name:
            # Same layout as landmarks
            rel_goal_pos = rel_pos.copy()
            entity_type = np.array([entity_mapping['obstacle']], dtype=np.float32)

            return np.hstack([rel_vel, rel_pos, rel_goal_pos, entity_type]).astype(np.float32)

        elif 'wall' in entity.name:
            # For walls, include rotated corner points as you already did
            rel_goal_pos = rel_pos.copy()
            goal_history = np.array([entity.id if entity.id is not None else 0], dtype=np.float32)
            entity_type = np.array([entity_mapping['wall']], dtype=np.float32)

            # Compute wall corners in WORLD frame, then rotate into ego frame
            # (Assumes your wall encodes a segment via endpoints[] along 'axis_pos' with a width)
            wall_o_corner_world = np.array([entity.endpoints[0], entity.axis_pos + entity.width / 2.0], dtype=np.float32) - agent_pos
            wall_d_corner_world = np.array([entity.endpoints[1], entity.axis_pos - entity.width / 2.0], dtype=np.float32) - agent_pos
            wall_o_corner = get_rotated_position_from_relative(wall_o_corner_world, agent_heading).astype(np.float32)
            wall_d_corner = get_rotated_position_from_relative(wall_d_corner_world, agent_heading).astype(np.float32)

            return np.hstack([
                rel_vel, rel_pos, rel_goal_pos, goal_history, wall_o_corner, wall_d_corner, entity_type
            ]).astype(np.float32)

        else:
            raise ValueError(f'{entity.name} not supported')

    def observation(self, agent: Agent, world: World) -> arr:
        """
        Returns an observation for the agent with rotation invariance:
        - Keep the same feature order/length as before to preserve training scripts.
        - All relative vectors (goal, neighbors, tube entrance/exit) are rotated into
        the agent's heading frame so that the agent's x-axis aligns with its heading.
        """
        # --- Agent state ---
        agent_pos = agent.state.p_pos
        agent_heading = float(getattr(agent.state, "theta", getattr(agent.state, "p_ang", 0.0)))
        agent_speed = float(getattr(agent.state, "speed", np.linalg.norm(agent.state.p_vel)))
        agent_vel = np.asarray(agent.state.p_vel, dtype=np.float32)


        # Rotate own velocity into ego frame
        rot_agent_vel = get_rotated_position_from_relative(agent_vel, agent_heading).astype(np.float32)

        # --- Goal related (single fair goal) ---
        goal_world_pos = self.landmark_poses[self.goal_match_index[agent.id]]
        # print("Agent", agent.id, "goal_world_pos", goal_world_pos)
        rel_goal_vec_world = goal_world_pos - agent_pos
        goal_pos = get_rotated_position_from_relative(rel_goal_vec_world, agent_heading).astype(np.float32)
        # print("Rotated goal_pos", goal_pos)

        # --- Two nearest neighbors (rotated) ---
        neighbor_dists = []
        for other in world.agents:
            if other is agent:
                continue
            # Skip completed agents (they're "ghosts")
            if other.status:
                continue
            rel_pos_world = other.state.p_pos - agent_pos
            dist = float(np.linalg.norm(rel_pos_world))
            neighbor_dists.append((dist, rel_pos_world))

        neighbor_dists.sort(key=lambda x: x[0])
        nearest = [n[1] for n in neighbor_dists[:2]]
        while len(nearest) < 2:
            nearest.append(np.zeros(world.dim_p, dtype=np.float32))

        # Rotate each neighbor vector into ego frame, then flatten to 4 slots
        rotated_neighbors = [
            get_rotated_position_from_relative(np.asarray(vec, dtype=np.float32), agent_heading).astype(np.float32)
            for vec in nearest
        ]

        nearest_neighbors = np.concatenate(rotated_neighbors, axis=0)

        # --- Tube params (rotated entrance/exit vectors + width + phase) ---
        # tube_entrance = np.asarray(world.tube_params['entrance'], dtype=np.float32)
        tube_exit = np.asarray(world.tube_params['exit'], dtype=np.float32)
        rel_to_exit_world = tube_exit - agent_pos
        rot_rel_exit = get_rotated_position_from_relative(rel_to_exit_world, agent_heading).astype(np.float32)

        # tube_exit = np.asarray(world.tube_params['exit'], dtype=np.float32)
        # tube_width = float(world.tube_params['width'])

        # Phase is computed in world coords; keep as scalar to preserve layout
        phase = float(self.get_agent_phase(agent, world))

        s, y, L, half_w = self._tube_coords(world, agent_pos)
        s_norm = np.clip(s / L, -2.0, 2.0)          # allow slight overshoot
        y_norm = np.clip(y / (half_w + 1e-9), -2.0, 2.0)
        dist_in = self._entrance_gate_distance(s, y, half_w) / (L + 1e-9)
        dist_out = self._exit_gate_distance(s, y, L, half_w) / (L + 1e-9)
        # print("Agent", agent.id, "s,y,L,half_w:", s_norm, y_norm, "dist_in:", dist_in, "dist_out:", dist_out)
        # === NEW: Add heading alignment feature ===
        corridor_vec = world.tube_params['e']
        corridor_heading = np.arctan2(corridor_vec[1], corridor_vec[0])
        heading_error = (agent_heading - corridor_heading + np.pi) % (2*np.pi) - np.pi
        heading_alignment = np.array([np.cos(heading_error), np.sin(heading_error)], dtype=np.float32)
        tube_params = np.concatenate([
            np.array([s_norm, y_norm]),  # rot_rel_entrance,
            np.array([dist_out], dtype=np.float32),  # rot_rel_exit, dist_in, 
            heading_alignment,                  
            # np.array([tube_width], dtype=np.float32),
            np.array([phase], dtype=np.float32)
        ], axis=0)
        # print("Agent", agent.id, "tube_params", tube_params, "np.array([agent.state.speed,agent_speed])", np.array([agent.state.speed, agent_speed]))

        # print("Agent", agent.id, "tube coords s,y,L,half_w:", s, y, L, half_w)
        # --- Assemble final obs in the SAME field order as before ---
        # [agent_vel(2), goal_pos(2), nearest_neighbors(4), tube_params(8)] = 16 dims
        return np.concatenate([
            np.array([np.cos(agent.state.theta), np.sin(agent.state.theta), agent.state.speed]), # np.array([np.cos(agent.state.theta), np.sin(agent.state.theta), agent.state.speed]),		#   rot_agent_vel, # self velocity (2 slots)
            rot_rel_exit,                # rotated goal vector
            nearest_neighbors,                  # two rotated neighbor vectors
            tube_params                         # rotated entrance/exit + width + phase
        ], axis=0).astype(np.float32)

    # done condition for each agent
    def done(self, agent:Agent, world:World) -> bool:
        # if we are using dones then return appropriate done
        if self.use_dones:
            if world.current_time_step >= world.world_length:
                return True
            else:
                landmark = world.get_entity('landmark',self.goal_match_index[agent.id])
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - 
                                                landmark.state.p_pos)))
                if dist < self.min_dist_thresh:
                    return True
                else:
                    return False
        # it not using dones then return done 
        # only when episode_length is reached
        else:
            if world.current_time_step >= world.world_length:
                return True
            else:
                return False

    def random_scenario(self, world):
        """
            Randomly place agents and landmarks
        """

        # set agents at random positions not colliding with obstacles
        # Initialize tube parameters
        self.setup_tube_params(world)
        num_agents_added = 0
        agents_added = []
        boundary_thresh = 0.99

        while True:
            if num_agents_added == self.num_agents:
                break

            # Add random jitter if needed
            jitter = 0.3 * np.random.uniform(-self.world_size, self.world_size, world.dim_p)
            angle = world.tube_params['angle']
            # print("jitter", jitter)
            perp_dir = np.array([np.sin(angle), np.cos(angle)])
            # print("Entrance", world.tube_params['entrance'], "perp_dir", perp_dir )
            # print("self.world_size+(num_agents_added) / 5 * perp_dir",( self.world_size+(num_agents_added)) / 5 * perp_dir)
            distance_from_entrance = (self.world_size + num_agents_added) / 3
            # print("distance_from_entrance", distance_from_entrance, "jitter", jitter)
            random_pos = world.tube_params['entrance'] + distance_from_entrance * perp_dir + jitter

            agent_size = world.agents[num_agents_added].size
            obs_collision = self.is_obstacle_collision(random_pos, agent_size, world)
            # goal_collision = self.is_goal_collision(uniform_pos, agent_size, world)

            agent_collision = self.check_agent_collision(random_pos, agent_size, agents_added)
            if not obs_collision and not agent_collision:
                world.agents[num_agents_added].state.p_pos = random_pos
                world.agents[num_agents_added].state.reset_velocity()
                world.agents[num_agents_added].state.c = np.zeros(world.dim_c)
                world.agents[num_agents_added].status = False
                agents_added.append(world.agents[num_agents_added])
                num_agents_added += 1
        # agent_pos = [agent.state.p_pos for agent in world.agents]
        #####################################################

        self.agent_id_updated = np.arange(self.num_agents)
        if self.formation_type == 'line':
            set_landmarks_in_line(self, world, line_angle=0, start_pos=np.array([-self.world_size/2, -self.world_size/2]), end_pos=np.array([self.world_size/2,-self.world_size/2]))
        elif self.formation_type == 'circle':
            set_landmarks_in_circle(self, world, center=np.array([0.0, world.tube_params['exit'][1]+self.world_size/5]), radius=self.world_size/3)
        elif self.formation_type == 'point':
            set_landmarks_in_point(self, world, tube_angle=world.tube_params['angle'], tube_endpoints=world.tube_params['exit'])
        # elif self.formation_type == 'random':
        # 	set_landmarks_random(self, world)
        # else:
        # 	raise NotImplementedError

        # Update landmark poses arrays
        self.landmark_poses = np.array([landmark.state.p_pos for landmark in world.landmarks])
        # print("landmark pose",self.landmark_poses)
        self.landmark_poses_occupied = np.zeros(self.num_agents)
        self.landmark_poses_updated = np.array([landmark.state.p_pos for landmark in world.landmarks])
        self.agent_id_updated = np.arange(self.num_agents)
        #####################################################

        ############ find minimum times to goals ############
        if self.max_speed is not None:
            for agent in world.agents:
                self.min_time(agent, world)
        #####################################################

    def setup_tube_params(self, world):
        """
        Set up tube parameters using modified landmark line logic
        """
        # Initialize tube list
        # world.tube_params = []
        # Calculate tube width based on number of agents
        self.tube_width = max(
            3 * world.agents[0].size * 2.5,  # Width based on agents # =3  TODO: harcoded
            self.world_size * 0.15  # Minimum width
        )

        random_angle = np.random.uniform(-np.pi/2, np.pi/2)
        # random_angle = 0.0
        # print(f"Random Angle: {random_angle*180/np.pi} degrees")
        # Calculate tube length
        tube_length = self.world_size * 0.8  # Use 80% of world size for tube length
        tube_length += np.random.uniform(-self.world_size*0.3, self.world_size*0.1)  # Add some randomness
        
        # Scales: make 1 full tube traversal worth ~goal_rew
        self.progress_gain = self.goal_rew / (tube_length*10)
        # Calculate center point of the world
        world_center = np.array([0, 0])
        
        # Calculate entrance and exit points using rotation
        # Start with vertical positions
        base_entrance = np.array([0, tube_length/4])  # Start above center
        base_exit = np.array([0, -tube_length/4])     # End below center
        
        # Create rotation matrix
        rotation_matrix = np.array([
            [np.cos(random_angle), np.sin(random_angle)],
            [-np.sin(random_angle), np.cos(random_angle)]
        ])
        
        # Apply rotation to entrance and exit points
        entrance = world_center + rotation_matrix @ base_entrance
        # print("rotation_matrix @ base_entrance", rotation_matrix @ base_entrance)
        exit = world_center + rotation_matrix @ base_exit
        # Store tube parameters
        world.tube_params = {
            'entrance': entrance,
            'exit': exit,
            'width': self.tube_width,
            'angle': random_angle,
            'length': tube_length
        }

        # print(f"Tube Entrance: {entrance}, Exit: {exit}, Angle: {random_angle*180/np.pi} degrees")
        # Calculate perpendicular direction for formation lines
        perpendicular_angle = random_angle + np.pi/2
        formation_direction = np.array([
            np.cos(perpendicular_angle),
            np.sin(perpendicular_angle)
        ])
        
        # Calculate line formation parameters
        line_length = self.tube_width * 0.8  # Slightly smaller than tube width
        half_line = (line_length/2) * formation_direction
        
        # Pre-tube formation line (above entrance)
        pre_tube_center = entrance + rotation_matrix @ np.array([0, self.world_size * 0.15])
        self.pre_tube_line = {
            'start_pos': pre_tube_center - half_line,
            'end_pos': pre_tube_center + half_line,
            'angle': perpendicular_angle
        }
        
        # Post-tube target line (below exit)
        post_tube_center = exit - rotation_matrix @ np.array([0, self.world_size * 0.15])
        self.post_tube_line = {
            'start_pos': post_tube_center - half_line,
            'end_pos': post_tube_center + half_line,
            'angle': perpendicular_angle
        }
        
        # Store additional parameters
        world.tube_params.update({
            'pre_tube_line': self.pre_tube_line,
            'post_tube_line': self.post_tube_line,
            'rotation_matrix': rotation_matrix,  # Store for potential future use
            'formation_direction': formation_direction  # Direction for agent lineup
        })

        # Precompute tube frame for fast queries
        L = float(np.linalg.norm(exit - entrance)) + 1e-9
        corridor_vec = (exit - entrance) / L
        n_vec = np.array([-corridor_vec[1], corridor_vec[0]], dtype=np.float32)  # left-hand normal
        world.tube_params.update({
            'e': corridor_vec,
            'n': n_vec,
            'L': L,
            'half_width': float(self.tube_width) * 0.5
        })

        # Full-width entrance gate settings (tunable)
        self.gate_front_ratio = getattr(self, 'gate_front_ratio', 0.08)  # inside tube
        self.gate_back_ratio = getattr(self, 'gate_back_ratio', 0.02)    # just outside entrance
        
        # Full-width exit gate settings (tunable)
        self.exit_back_ratio = getattr(self, 'exit_back_ratio', 0.02)    # inside tube near exit
        self.exit_front_ratio = getattr(self, 'exit_front_ratio', 0.08)  # just outside exit	

    def min_time(self, agent:Agent, world:World) -> float:
        assert agent.max_speed is not None, "Agent needs to have a max_speed."
        assert agent.max_speed > 0, "Agent max_speed should be positive."
        agent_id = agent.id
        # get the goal associated to this agent
        landmark = world.get_entity(entity_type='landmark', id=self.goal_match_index[agent.id])
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - 
                                        landmark.state.p_pos)))
        min_time = dist / agent.max_speed
        agent.goal_min_time = min_time
        return min_time