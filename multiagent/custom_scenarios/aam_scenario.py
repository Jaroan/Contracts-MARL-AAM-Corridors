import numpy as np
from multiagent.scenario import BaseScenario
from multiagent.core import Entity, Agent, Landmark
from multiagent.custom_scenarios.base_aam_scenario import AAMBaseScenario


class AamScenario(AAMBaseScenario):
    """
    Corridor + fairness + rotation-invariant scenario.
    This is the non-safety version. Safety gets added in SafeAamScenario.
    """
    def __init__(self, args):
        super().__init__(args)
    # ===============================================================
    # 1. WORLD + AGENT INITIALIZATION
    # ===============================================================

    def reset_world(self, world):
        """
        Calls the corridor placement + resets velocities + flags.
        """
        # Place agents + landmarks according to corridor geometry
        self.random_scenario(world)

        # Reset agent state vectors
        for agent in world.agents:
            agent.state.c = np.zeros(world.dim_c)
            agent.done = False

        # Cache initial distances for fairness computations
        self._cache_initial_spacing(world)

    # ===============================================================
    # 2. SCENARIO LAYOUT (override this in TrainScenario/EvalScenario)
    # ===============================================================
    def random_scenario(self, world):
        """
        Default corridor placement — can be overridden by TrainScenario & EvalScenario.
        """
        tube_x0 = -0.8 * self.world_size
        tube_x1 =  0.8 * self.world_size
        tube_ymin = -self.corridor_width / 2
        tube_ymax =  self.corridor_width / 2

        # Evenly space agents across corridor width
        ys = np.linspace(tube_ymin * 0.7, tube_ymax * 0.7, self.num_agents)

        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.array([tube_x0, ys[i]])
            agent.state.reset_velocity(theta=0)

        # Each agent’s single "goal" landmark is at right-end of tube
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.array([tube_x1, ys[i]])
            landmark.state.stop()
            landmark.heading = 0
            landmark.speed = self.goal_speed_max

    # ===============================================================
    # 3. REWARD FUNCTION (NO SAFETY TERMS HERE)
    # ===============================================================
    def reward(self, agent: Agent, world):
        """
        Full corridor reward (NO SAFETY). Includes:
        - distance to corresponding goal
        - collision penalty (non-safety)
        - fairness penalty (spacing)
        - staying inside corridor
        """
        rew = 0.0

        agent_index = world.agents.index(agent)
        goal = world.landmarks[agent_index]

        # -----------------------------------------------------------
        # Distance-to-goal reward (encourage movement rightward)
        # -----------------------------------------------------------
        dist = np.linalg.norm(agent.state.p_pos - goal.state.p_pos)
        rew -= dist

        # -----------------------------------------------------------
        # Collision penalty (non-safety version)
        # -----------------------------------------------------------
        if self.use_collisions:
            for other in world.agents:
                if other is agent:
                    continue
                if self._is_collision(agent, other):
                    rew -= self.collision_penalty

        # -----------------------------------------------------------
        # Corridor deviation penalty (encourage staying inside tube)
        # -----------------------------------------------------------
        y = agent.state.p_pos[1]
        tube_half_w = self.corridor_width / 2
        if abs(y) > tube_half_w:
            rew -= self.out_of_corridor_penalty * (abs(y) - tube_half_w)

        # -----------------------------------------------------------
        # Fairness / spacing reward (distance variance)
        # -----------------------------------------------------------
        if self.use_fairness:
            rew += self._fairness_reward(agent, world, agent_index)

        return rew

    # ===============================================================
    # 4. FAIRNESS (SPACING) REWARD
    # ===============================================================
    def _cache_initial_spacing(self, world):
        """Cache initial y ordering to track spacing deviation."""
        self.initial_positions = np.array([a.state.p_pos.copy() for a in world.agents])
        self.sorted_indices = np.argsort(self.initial_positions[:, 1])

    def _fairness_reward(self, agent, world, agent_index):
        """
        Penalizes deviation from initial y-spacing formation.
        """
        current_positions = np.array([a.state.p_pos for a in world.agents])
        sorted_now = np.argsort(current_positions[:, 1])

        # rank difference penalty (how far from original row)
        initial_rank = np.where(self.sorted_indices == agent_index)[0][0]
        current_rank  = np.where(sorted_now == agent_index)[0][0]

        rank_diff = abs(initial_rank - current_rank)
        return -self.fairness_penalty * rank_diff

    # ===============================================================
    # 5. DONE FUNCTION
    # ===============================================================
    def done(self, agent, world):
        """
        Episode ends for agent if:
        - reaches goal (x beyond tube end)
        - leaves allowed world boundary
        - (time handled by env, not here)
        """
        x = agent.state.p_pos[0]
        y = agent.state.p_pos[1]

        # reach goal
        if x > self.goal_x_threshold:
            return True

        # world boundary check
        if abs(x) > self.world_size or abs(y) > self.world_size:
            return True

        return False

    # ===============================================================
    # 6. OBSERVATION (ROTATION INVARIANT)
    # ===============================================================
    def observation(self, agent, world):
        """
        Rotation-invariant observation:
        Agent-centric frame aligned with corridor axis (x direction).
        Observes relative positions of:
            - itself
            - other agents
            - its goal landmark
        """
        agent_pos = agent.state.p_pos
        agent_idx = world.agents.index(agent)

        # relative positions of other agents
        rel_agents = []
        for other in world.agents:
            rel = other.state.p_pos - agent_pos
            rel_agents.append(rel)

        # relative goal
        goal = world.landmarks[agent_idx]
        rel_goal = goal.state.p_pos - agent_pos

        # stack into obs vector
        obs = np.concatenate([
            np.array([agent.state.speed, agent.state.theta]),
            rel_goal,
            np.array(rel_agents).flatten(),
        ])

        return obs

    # ===============================================================
    # 7. GRAPH OBSERVATION (GNNS)
    # ===============================================================
    def graph_observation(self, agent, world):
        """
        Node features for GNN encoder:
            [relative_pos_x, relative_pos_y, is_agent, is_goal]
        """
        agent_pos = agent.state.p_pos
        nodes = []

        for a in world.agents:
            nodes.append([
                a.state.p_pos[0] - agent_pos[0],
                a.state.p_pos[1] - agent_pos[1],
                1.0,  # agent flag
                0.0   # landmark flag
            ])

        for l in world.landmarks:
            nodes.append([
                l.state.p_pos[0] - agent_pos[0],
                l.state.p_pos[1] - agent_pos[1],
                0.0,
                1.0,  # landmark flag
            ])

        return np.array(nodes)

    # ===============================================================
    # 8. INFO CALLBACK
    # ===============================================================
    def info_callback(self, agent, world):
        agent_index = world.agents.index(agent)
        goal = world.landmarks[agent_index]

        dist_to_goal = np.linalg.norm(agent.state.p_pos - goal.state.p_pos)
        return {"dist_to_goal": dist_to_goal}

    # ===============================================================
    # 9. UTILS
    # ===============================================================
    def _is_collision(self, a1, a2):
        """Simple circle collision."""
        dist = np.linalg.norm(a1.state.p_pos - a2.state.p_pos)
        return dist < (a1.size + a2.size)
