import math
import numpy as np

from multiagent.custom_scenarios.aam_scenario import AamScenario
from multiagent.core import Agent
from multiagent.config import RewardWeightConfig, RewardBinaryConfig
from multiagent.core import EntityDynamicsType, World, Agent, Landmark, Entity, Wall

class SafeAamScenario(AamScenario):
    """
    Safety-augmented AAM corridor scenario.

    Extends AamScenario (corridor + fairness) with:
      - safety-violation penalty (proximity / conflict)
      - HJ-value penalty (using precomputed HJ tables via the World)

    Medium safety set:
      - collision penalty       -> already in AamScenario.reward
      - safety violation penalty -> reward_safety_violation
      - HJ-value penalty         -> reward_hj_value

    No:x    
      - barrier shaping
      - TTR shaping
      - diff_from_filtered_action penalty
    """

    # def __init__(self, args):
    #     super().__init__(args)

    #     # Global reward clipping bounds
    #     self.min_reward = RewardWeightConfig.MIN_REWARD
    #     self.max_reward = RewardWeightConfig.MAX_REWARD

    #     # Scales for safety components
    #     self.conflict_rew_scaled = RewardWeightConfig.SAFETY_VIOLATION
    #     self.conflict_value_rew_scaled = RewardWeightConfig.HJ_VALUE

    #     # Optional extras (not used in "medium" regime, but left for extensibility)
    #     self.multiple_engagement_rew_scaled = getattr(
    #         RewardWeightConfig, "POTENTIAL_CONFLICT", 0.0
    #     )
    #     self.diff_from_filtered_action_rew_scaled = getattr(
    #         RewardWeightConfig, "DIFF_FROM_FILTERED_ACTION", 0.0
    #     )

    # # ------------------------------------------------------------------
    # # 1. REWARD: base corridor/fairness + safety augmentation
    # # ------------------------------------------------------------------
    # def reward(self, agent: Agent, world) -> float:
    #     """
    #     Start with AamScenario reward (corridor + fairness + collisions),
    #     then add medium safety terms (safety violation + HJ value).
    #     """
    #     # Base reward from corridor + fairness + collisions
    #     rew = super().reward(agent, world)

    #     # SAFETY_VIOLATION (proximity / conflict)
    #     if getattr(RewardBinaryConfig, "SAFETY_VIOLATION", False):
    #         rew += self.reward_safety_violation(agent, world)

    #     # HJ_VALUE (distance to unsafe set from HJ analysis)
    #     if getattr(RewardBinaryConfig, "HJ_VALUE", False):
    #         rew += self.reward_hj_value(agent, world)

    #     # (Optional extras: only active if user enables the flags)
    #     if getattr(RewardBinaryConfig, "POTENTIAL_CONFLICT", False):
    #         rew += self.reward_multiple_engagement(agent, world)

    #     if getattr(RewardBinaryConfig, "DIFF_FROM_FILTERED_ACTION", False):
    #         rew += self.reward_diff_from_filtered_action(agent)

    #     # Final global clip
    #     return float(np.clip(rew, self.min_reward, self.max_reward))

    # # ------------------------------------------------------------------
    # # 2. SAFETY VIOLATION (geometric conflict)
    # # ------------------------------------------------------------------
    # def reward_safety_violation(self, agent: Agent, world) -> float:
    #     """
    #     Penalize when another agent is in geometric conflict
    #     (within minimum separation distance).
    #     """
    #     rew = 0.0
    #     for other in world.agents:
    #         if other is agent or other.done:
    #             continue

    #         d = np.linalg.norm(agent.state.p_pos - other.state.p_pos)
    #         if d <= self.separation_distance:
    #             # Single hit per violating neighbor
    #             rew += self.conflict_rew_scaled
    #     return rew

    # # ------------------------------------------------------------------
    # # 3. POTENTIAL CONFLICT (optional, not needed for "medium")
    # # ------------------------------------------------------------------
    # def reward_multiple_engagement(self, agent: Agent, world) -> float:
    #     """
    #     Optional: penalize when >1 agent is within a larger engagement range
    #     and closing. Left for extensibility; can be disabled via config.
    #     """
    #     engagement_dist = getattr(
    #         self, "engagement_distance", 1.5 * self.separation_distance
    #     )
    #     engagement_count = 0
    #     engagement_pen = 0.0

    #     for other in world.agents:
    #         if other is agent or other.done:
    #             continue

    #         rel = other.state.p_pos - agent.state.p_pos
    #         d = np.linalg.norm(rel)
    #         if d > engagement_dist:
    #             continue

    #         # closeness factor in [0,1]
    #         closeness = 1.0 - float(
    #             np.clip((d - self.separation_distance) /
    #                     (engagement_dist - self.separation_distance + 1e-6),
    #                     0.0, 1.0)
    #         )

    #         # closure rate along line-of-sight
    #         dir_vec = rel / (d + 1e-8)
    #         rel_vel = other.state.p_vel - agent.state.p_vel
    #         closing_rate = float(np.inner(dir_vec, rel_vel))
    #         closing_rate = abs(min(0.0, closing_rate))  # only count approaching

    #         engagement_pen += closeness * closing_rate
    #         engagement_count += 1

    #     if engagement_count > 1:
    #         return self.multiple_engagement_rew_scaled * engagement_pen
    #     return 0.0

    # # ------------------------------------------------------------------
    # # 4. DIFF FROM FILTERED ACTION (optional, not needed for "medium")
    # # ------------------------------------------------------------------
    # def reward_diff_from_filtered_action(self, agent: Agent) -> float:
    #     """
    #     Optional: penalize deviation from a safety-filtered action.
    #     Assumes agent.action_diff is set by a higher-level safety filter.
    #     """
    #     if agent.done:
    #         return 0.0
    #     action_diff = float(getattr(agent, "action_diff", 0.0))
    #     return self.diff_from_filtered_action_rew_scaled * action_diff

    # # ------------------------------------------------------------------
    # # 5. HJ-VALUE PENALTY
    # # ------------------------------------------------------------------
    # def reward_hj_value(self, agent: Agent, world, eps_hj: float = 0.4) -> float:
    #     """
    #     Penalize low HJ-value between pairs of agents.

    #     Requires:
    #       - world.get_hj_value_between_two_agents(agent_i, agent_j)
    #     If this method is missing, returns 0 and safety is purely geometric.
    #     """
    #     if not hasattr(world, "get_hj_value_between_two_agents"):
    #         return 0.0

    #     rew = 0.0
    #     for other in world.agents:
    #         if other is agent or other.done:
    #             continue

    #         # get_hj_value_between_two_agents is assumed to return
    #         # a scalar HJ value v; smaller v means more unsafe.
    #         v = float(world.get_hj_value_between_two_agents(agent, other))

    #         # penalize when v < eps_hj
    #         conflict_val_pen = abs(min(v - eps_hj, 0.0))
    #         rew += self.conflict_value_rew_scaled * conflict_val_pen

    #     return rew

    # ------------------------------------------------------------------
    # 6. REFACTORING
    # ------------------------------------------------------------------

    def is_obstacle_collision(self, pos, entity_size:float, world:World) -> bool:
        # pos is entity position "entity.state.p_pos"
        collision = False
        for obstacle in world.obstacles:
            delta_pos = obstacle.state.p_pos - pos
            dist = np.linalg.norm(delta_pos)
            dist_min = 2.0*(obstacle.size + entity_size)

            if dist < dist_min:
                collision = True
                break	
        
        # check collision with walls
        for wall in world.walls:
            if wall.orient == 'H':
                # Horizontal wall, check for collision along the y-axis
                if (wall.axis_pos - 1.5*entity_size ) <= pos[1] <= (wall.axis_pos + 1.5*entity_size ):
                    if (wall.endpoints[0] - 1.5*entity_size ) <= pos[0] <= (wall.endpoints[1] + 1.5*entity_size ):
                        collision = True
                        break
            elif wall.orient == 'V':
                # Vertical wall, check for collision along the x-axis
                if (wall.axis_pos - 1.5*entity_size ) <= pos[0] <= (wall.axis_pos + 1.5*entity_size ):
                    if (wall.endpoints[0] - 1.5*entity_size ) <= pos[1] <= (wall.endpoints[1] + 1.5*entity_size ):
                        collision = True
                        break
        return collision
    # check collision of entity with obstacles and walls

    # check collision of agent with other agents
    def check_agent_collision(self, pos, agent_size, agent_added) -> bool:
        collision = False
        if len(agent_added):
            for agent in agent_added:
                delta_pos = agent.state.p_pos - pos
                dist = np.linalg.norm(delta_pos)
                if dist < self.separation_distance:
                    collision = True
                    break
        return collision

    # check collision of agent with another agent
    def is_collision(self, agent1:Agent, agent2:Agent) -> bool:
        if agent1.status or agent2.status:
            # print("agent status",agent1.status,agent2.status)
            return False
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.linalg.norm(delta_pos)
        dist_min = self.separation_distance
        return True if dist < dist_min else False

    def is_landmark_collision(self, pos, size:float, landmark_list:list) -> bool:
        collision = False
        for landmark in landmark_list:
            delta_pos = landmark.state.p_pos - pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = 1.2*(size + landmark.size)
            if dist < dist_min:
                collision = True
                break
        return collision

    
    def is_confliction(self, a, b):
        """True if agents are within the minimum separation distance."""
        d = np.linalg.norm(a.state.p_pos - b.state.p_pos)
        return d <= self.separation_distance

    def is_in_engagement(self, a, b):
        """True if agents are within a larger 'engagement' range."""
        engagement_distance = getattr(self, "engagement_distance", 1.5 * self.separation_distance)
        d = np.linalg.norm(a.state.p_pos - b.state.p_pos)
        return d <= engagement_distance


