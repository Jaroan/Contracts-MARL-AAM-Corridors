# multiagent/custom_scenarios/aam_eval.py

from multiagent.custom_scenarios.safe_aam_scenario import SafeAamScenario
import numpy as np
from multiagent.core import EntityDynamicsType, World, Agent, Landmark, Entity, Wall
from numpy import ndarray as arr
from multiagent.custom_scenarios.utils import *


class Scenario(SafeAamScenario):
    """
    EVALUATION SCENARIO (Deterministic)
    ------------------------------------
    No randomness allowed.
    Uses a fixed corridor, fixed initial positions, and fixed landmarks.

    This ensures:
      - reproducible evaluation
      - stable rendering
      - directly comparable results for different checkpoints
    """

    # def __init__(self, args):
    #     super().__init__(args)

    # ------------------------------------------------------------
    # MAIN WORLD SETUP
    # ------------------------------------------------------------
    # def make_world(self, args):
    #     """
    #     Build the world using SafeAamScenario, then apply FIXED settings.
    #     """
    #     world = super().make_world(args)

    #     self.fix_corridor(world)
    #     self.fix_start_positions(world)
    #     self.fix_landmarks(world)

    #     return world

    # # ------------------------------------------------------------
    # # FIXED CORRIDOR
    # # ------------------------------------------------------------
    # def fix_corridor(self, world):
    #     """
    #     No randomness: use canonical corridor parameters.
    #     Angle = 0, width = default.
    #     """
    #     self.world_rotation_angle = 0.0
    #     # Do not change corridor_width — use default from AamScenario

    # # ------------------------------------------------------------
    # # FIXED INITIAL POSITIONS
    # # ------------------------------------------------------------
    # def fix_start_positions(self, world):
    #     """
    #     Canonical left-to-right corridor start layout.
    #     """
    #     num = self.num_agents
    #     ys = np.linspace(-0.4 * self.world_size, 0.4 * self.world_size, num)
    #     x0 = -0.9 * self.world_size

    #     for i, agent in enumerate(world.agents):
    #         agent.state.p_pos = np.array([x0, ys[i]])
    #         agent.state.reset_velocity(theta=0)
    #         agent.done = False

    # # ------------------------------------------------------------
    # # FIXED LANDMARK POSITIONS
    # # ------------------------------------------------------------
    # def fix_landmarks(self, world):
    #     """
    #     Canonical 2-waypoint corridor:
    #         1. midpoint
    #         2. end of corridor
    #     """

    #     # AamScenario already creates correct number of landmarks per agent
    #     # Here we place them in fixed positions.
    #     num = self.num_agents
    #     L = self.num_landmarks

    #     # midpoint (0, 0)
    #     # terminus (world_size/2, 0)
    #     y_spacing = 0.0  # simple corridor

    #     xs = np.array([0.0, 0.9 * self.world_size])
    #     ys = np.array([y_spacing, y_spacing])

    #     # one set for all agents (RI handles individuality)
    #     landmark_positions = [np.array([xs[k], ys[k]]) for k in range(L)]

    #     for lm, pos in zip(world.landmarks, landmark_positions):
    #         lm.state.p_pos = pos
    #         lm.state.stop()

    # ------------------------------------------------------------
    # REFACTORING
    # ------------------------------------------------------------

    def reward(self, agent: Agent, world: World) -> float:
        rew = 0.0
        current_phase = self.get_agent_phase(agent, world)
        # print("Agent", agent.id, "phase", current_phase, "previous_phase", agent.previous_phase, "phase_reached", self.phase_reached[agent.id])
        # print("Goalrew",self.goal_rew, "Collisionrew",self.collision_rew)
        # Common rewards across all phases
        # Collision penalties
        # if agent.collide:
        # 	for a in world.agents:
        # 		if a.id == agent.id:
        # 			continue
        # 		if self.is_collision(a, agent):
        # 			rew -= self.collision_rew
        # 			# print(f"!!!Agent {agent.id} collided with agent {a.id} penalty",self.collision_rew*4 )
        # 			# print(" self.separation_distance", self.separation_distance)
        # 			# input("Collision")
            
        # 	if self.is_obstacle_collision(pos=agent.state.p_pos,
        # 								entity_size=agent.size, 
        # 								world=world):
        # 		rew -= self.collision_rew
        # 		# print(f"Agent {agent.id} collided with obstacle")

        # Calculate tube length
        tube_direction = world.tube_params['exit'] - world.tube_params['entrance']
        tube_length = np.linalg.norm(tube_direction)
        agent_pos = agent.state.p_pos
        agent_heading = agent.state.theta
        heading_vec = np.array([np.cos(agent_heading), np.sin(agent_heading)])
        s, y, L, half_w = self._tube_coords(world, agent_pos)
        front_agents = []
        back_agents = []
        # print("self._in_entrance_gate(s, y, L, half_w)",self._in_entrance_gate(s, y, L, half_w))
        for other in world.agents:
            if other is agent:
                continue
            rel_vec = other.state.p_pos - agent_pos
            proj = np.dot(rel_vec, heading_vec)
            if proj > 0:
                front_agents.append((proj, other))
            else:
                back_agents.append((proj, other))

        # Get closest in front and back
        front_agent = min(front_agents, key=lambda x: x[0])[1] if front_agents else None
        back_agent = max(back_agents, key=lambda x: x[0])[1] if back_agents else None

        # print("front_agent", front_agent.id if front_agent else "None")
        # print("back_agent", back_agent.id if back_agent else "None")
        
        # Calculate desired spacing based on tube length and number of agents
        desired_spacing = self.separation_distance  # 3 is the number of agents in the tube TODO: harcoded
        
        # Track agent's previous phase if not already stored
        if not hasattr(agent, 'previous_phase'):
            agent.previous_phase = 0
        
        # ##Modified phase transition handling
        if current_phase == 2 and current_phase > agent.previous_phase + 1:
            # Only penalize clear phase skips (e.g., 0 to 2)
            # print("self.goal_rew",self.goal_rew)
            rew -= self.goal_rew  #*3  # Reduced penalty
            # print(f"Agent {agent.id} penalized for skipping from phase {agent.previous_phase} to {current_phase} rew", rew)
        tube_direction_vector = tube_direction / tube_length
        entrance_to_agent = agent.state.p_pos - world.tube_params['entrance']
        proj = np.dot(entrance_to_agent, tube_direction_vector)
        # entrance_dist = np.linalg.norm(entrance_to_agent - proj * tube_direction)

        if current_phase == agent.previous_phase+1 and self.phase_reached[agent.id] == current_phase-1:
            # Reward proper phase progression
            # if current_phase == 1 and 0 <= proj < 0.1 * tube_length and entrance_dist <  0.2 * tube_length and (self.entry_reward_cooldown[agent.id] == 0):
            if current_phase == 1 and self._in_entrance_gate(s, y, L, half_w) and (self.entry_reward_cooldown[agent.id] == 0):
                # print("Agent pos, entrance, exit", agent.state.p_pos, world.tube_params['entrance'], world.tube_params['exit'])
                # print("distance check", proj, entrance_dist)

                # Reward if agent moves into tube after exiting
                rew += self.goal_rew  # *3  # Positive reward for proper transition
                self.entry_reward_cooldown[agent.id] = self.phase_reward_cooldown_steps  # Cooldown period to prevent repeated rewards
                self.phase_reached[agent.id] = 1  # Mark Phase 1 completed
                # print(f"Agent {agent.id} properly progressed from phase {agent.previous_phase} to {current_phase} rew", rew)
            elif current_phase == 2:
                # Rewards if agent moves out of tube
                # print("Agent", agent.id, " in post-tube phase. Status:",  agent.status)
                rew += self.goal_rew  # *3
                self.phase_reached[agent.id] = 2  # Mark Phase 2 completed
                # print("Agent",agent.id,"reached fair goal")
                if agent.status is False:
                    agent.status = True
                    agent.state.reset_velocity()
                    rew += self.goal_rew * 5
                # print(f"Agent {agent.id} properly progressed from phase {agent.previous_phase} to {current_phase} rew", rew)
                # Update the global phase tracker if any agent progresses

        # Phase-specific rewards
        # print("Agent",agent.id,"current_phase",current_phase,"prev phase_reached",self.phase_reached)
        if current_phase == 0:  # Pre-tube phase
            # Reward for getting closer to tube entrance
            s, y, L, half_w = self._tube_coords(world, agent.state.p_pos)
            # print("Agent", agent.id, " Phase 0 s,y,L,half_w:", s, y, L, half_w)
            dist_to_entrance_edge = self._entrance_gate_distance(s, y, half_w)
            rew -= dist_to_entrance_edge

            # === NEW: Heading alignment reward ===
            # Desired heading: align with corridor direction
            corridor_vec = world.tube_params['e']  # unit vector along corridor
            corridor_heading = np.arctan2(corridor_vec[1], corridor_vec[0])
            # print("corridor_heading (deg):", corridor_heading*180/np.pi, "angle", world.tube_params['angle']*180/np.pi)
            agent_heading = agent.state.theta
            heading_error = abs((agent_heading - corridor_heading + np.pi) % (2*np.pi) - np.pi)

            # Only enforce alignment when close to entrance (within 0.5*world_size)
            if dist_to_entrance_edge < self.world_size * 0.1:
                # Penalize heading misalignment (max penalty at 180°, none at 0°)
                rew -= heading_error * self.formation_rew * 0.5
            # === Penalize lateral approach (agents must approach along corridor axis) ===
            # If agent is laterally offset from entrance but trying to enter: penalize
            # if abs(y) > half_w * 0.8 and s < 0 and s > -self.world_size * 0.1:
            # 	# Agent is beside entrance but close: penalize
            # 	# print("s", s, "-self.world_size * 0.1", -self.world_size * 0.1, "y", y, "half_w*0.7", half_w*0.7)
            # 	rew -= abs(y) * self.collision_rew * 0.05
                # print(f"Agent {agent.id} penalized for lateral approach y={y:.2f} rew", rew)
                # input("Lateral approach penalty applied")
        elif current_phase == 1:  # In-tube phase
            spacing_error = 0
            max_spacing_error = 0

            if front_agent:
                # print("np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos)", np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos), "desired_spacing", desired_spacing)
                diff = np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos) - desired_spacing
                # print("diff", diff)
                spacing_error += np.abs(diff) if diff < 0 else 0
                max_spacing_error = max(max_spacing_error, np.abs(diff))
            if back_agent:
                # print("np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos)", np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos), "desired_spacing", desired_spacing)
                diff = np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos) - desired_spacing
                # print("diff", diff)
                max_spacing_error = max(max_spacing_error, np.abs(diff))
                spacing_error += np.abs(diff) if diff < 0 else 0
            if spacing_error > 0:
                # print("Phase 1 spacing_error",spacing_error)
                self.spacing_violation[agent.id] += 1
            rew -= spacing_error * self.formation_rew  # Higher weight for maintaining formation in tube
            dist_to_exit_edge = self._exit_gate_distance(s, y, L, half_w)
            rew -= dist_to_exit_edge
            self.steps_in_corridor[agent.id] += 1

            corridor_vec = world.tube_params['e']  # unit vector along corridor
            corridor_heading = np.arctan2(corridor_vec[1], corridor_vec[0])
            # print("corridor_heading (deg):", corridor_heading*180/np.pi, "angle", world.tube_params['angle']*180/np.pi)
            agent_heading = agent.state.theta
            heading_error = abs((agent_heading - corridor_heading + np.pi) % (2*np.pi) - np.pi)
            rew -= heading_error * self.formation_rew * 0.1




        # print("Agent.status",agent.status)
        if self.phase_reached[agent.id] == 1 and current_phase == 0:
            # print("Agent",agent.id,"left corridor")
            self.conformance_percent[agent.id] += 1
            # print("conformance_percent",self.conformance_percent[agent.id])

        if current_phase > self.phase_reached[agent.id]:
            # print(f"Agent {agent.id} reached phase {current_phase}")
            self.phase_reached[agent.id] = current_phase  # Update max phase reached globally
        ## penalize for moving from higher phase to lower phase
        if current_phase < agent.previous_phase:
            # print(f"Agent {agent.id} tried to move back to phase {current_phase} from {agent.previous_phase}")
            rew -= self.collision_rew  #*4
            # print(f"Agent {agent.id} tried to move back to phase {current_phase} from {agent.previous_phase} rew", rew)
        if current_phase < self.phase_reached[agent.id]:
            rew -= self.collision_rew
            # print(f"Agent {agent.id} tried to move back to phase {current_phase} from {self.phase_reached[agent.id]} rew", rew)
        # Store current phase for next step
        agent.previous_phase = current_phase
        if self._in_tube_rect(s, y, L, half_w) and not current_phase == 1:
            rew -= self.collision_rew  #*2
            # print(f"Agent {agent.id} is in tube but not in phase 1 rew", rew)

        # If agent is past exit plane but never entered corridor: continuous penalty
        if s > L and self.phase_reached[agent.id] < 1:
            rew -= self.goal_rew
            # print(f"Agent {agent.id} skipped corridor (s={s:.2f} > L={L:.2f}): penalty {self.goal_rew}")
        
        # print(f"Agent {agent.id} total reward: ", rew)
        # input("Reward calculation complete for agent {}".format(agent.id))
        # input("Press Enter to continue...")

        return np.clip(rew, -4*self.collision_rew, self.goal_rew*5)

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

    # --- Shared geometry helpers (reduce redundancy) ---
    def _tube_frame(self, world: World):
        tp = world.tube_params
        return tp['entrance'], tp['e'], tp['n'], float(tp['L']), float(tp['half_width'])

    def _tube_coords(self, world: World, pos: np.ndarray):
        """Return (s, y, L, half_w): longitudinal s from entrance and signed lateral y."""
        entrance, e, n, L, half_w = self._tube_frame(world)
        r = np.asarray(pos, dtype=np.float32) - entrance  # r: 2D vector from the tube entrance to the queried position, in world coordinates. r = pos − entrance.
        s = float(np.dot(r, e))  # s is the along-tube coordinate from the entrance plane (s=0 at the entrance, s>0 inside the tube, s<0 before the entrance).
        y = float(np.dot(r, n))  # y is the signed lateral offset from the tube centerline (y=0 on centerline, |y| increases outward).
        return s, y, L, half_w

    def _in_tube_rect(self, s: float, y: float, L: float, half_w: float, eps: float = 0.05) -> bool:
        return (-eps <= s <= L + eps) and (abs(y) <= half_w + eps)
    
    def _in_entrance_gate(self, s: float, y: float, L: float, half_w: float, eps: float = 0.05) -> bool:
        """Full-width gate spanning the entrance edge: s in [-gate_back, +gate_front], |y|<=half_w."""
        gate_front = float(self.gate_front_ratio) * L
        gate_back = float(self.gate_back_ratio) * L
        return (-gate_back - eps <= s <= gate_front + eps) and (abs(y) <= half_w + eps)
    
    # Distance to the full-width entrance edge (rectangle at s=0 with |y|<=half_w)
    def _entrance_gate_distance(self, s: float, y: float, half_w: float) -> float:
        # nearest point on entrance edge: (s'=0, y' clamped to [-half_w, half_w])
        clamped_y = float(np.clip(y, -half_w, half_w))
        ds = abs(float(s))   # before entrance plane → distance along axis
        dy = float(y) - clamped_y       # lateral overflow outside corridor width
        return float(np.hypot(ds, dy))
    
    # Full-width exit gate: s in [L - exit_back, L + exit_front], |y| <= half_w
    def _in_exit_gate(self, s: float, y: float, L: float, half_w: float, eps: float = 0.05) -> bool:
        exit_back = float(self.exit_back_ratio) * L
        exit_front = float(self.exit_front_ratio) * L
        return (L - exit_back - eps <= s <= L + exit_front + eps) and (abs(y) <= half_w + eps)

    # Distance to the exit edge segment (plane s=L, clamped laterally)
    def _exit_gate_distance(self, s: float, y: float, L: float, half_w: float, penalize_backward: bool = False) -> float:
        """
        If penalize_backward is False, only measure remaining distance inside the tube (L - s)+.
        If True, use |s - L| (useful if already beyond exit but you want symmetric distance).
        """
        clamped_y = float(np.clip(y, -half_w, half_w))
        ds = max(0.0, float(L - s)) if not penalize_backward else abs(float(s - L))
        dy = float(y) - clamped_y
        return float(np.hypot(ds, dy))

    # --- Optimized queries using the frame ---
    def is_in_tube(self, world: World, pos: np.ndarray) -> bool:
        s, y, L, half_w = self._tube_coords(world, pos)
        return self._in_tube_rect(s, y, L, half_w)
    
    def get_agent_phase(self, agent: Agent, world: World):
        pos = agent.state.p_pos
        # in_tube = self.is_in_tube(world, pos)
        s, y, L, half_w = self._tube_coords(world, pos)
        in_tube = self._in_tube_rect(s, y, L, half_w)

        passed_tube = (s > L)
        valid_entrance = self._in_entrance_gate(s, y, L, half_w)
        valid_exit = self._in_exit_gate(s, y, L, half_w)
        # if agent.id == 2:
        # 	print("Agent", agent.id, "position", pos, "in_tube:", in_tube)
        # 	print(f"valid exit: {valid_exit}, passed_tube: {passed_tube}, s: {float(s):.3f}, y: {float(y):.3f}, L: {float(L):.3f}, half_w: {float(half_w):.3f}")
        # Decrement cooldown here once per call
        if self.entry_reward_cooldown[agent.id] > 0:
            # print("Decrementing cooldown for agent", agent.id, "from", self.entry_reward_cooldown[agent.id])
            self.entry_reward_cooldown[agent.id] -= 1

        # Only check valid_entrance when transitioning from phase 0 to 1
        if not in_tube and not passed_tube:
            if not hasattr(agent, 'previous_phase'):
                # print("Agent", agent.id, "is in pre-tube phase")
                agent.previous_phase = 0
            # print("Agent {} is in pre-tube phase 000".format(agent.id))
            return 0  # Pre-tube phase
        elif in_tube:
            if agent.previous_phase == 0:
                if valid_entrance:
                    # print("Agent", agent.id, "entered tube correctly 1111")
                    # agent.previous_phase = 1
                    return 1  # Entered correctly
                else:
                    return 0  # Reset if entered incorrectly
            else:
                # print("Agent", agent.id, "is in tube phase 1111")
                # agent.previous_phase = 1

                if agent.previous_phase == 1 and valid_exit:
                    # if passed_tube and valid_exit:
                    # print("Agent", agent.id, "exited tube correctly YOOO")
                        # agent.previous_phase = 2
                    # agent.previous_phase = 2
                    return 2
                return 1  # Already in tube, stay in phase 1
        if passed_tube:
            if self.phase_reached[agent.id] >= 1:
                if agent.previous_phase == 1 and valid_exit:
                    # if passed_tube and valid_exit:
                    # print("Agent", agent.id, "exited tube correctly 2222")
                        # agent.previous_phase = 2
                    # agent.previous_phase = 2
                    return 2
                elif agent.previous_phase == 2:
                    # print("Agent", agent.id, "is in post-tube phase 2222")
                    return 2
                # print("Agent", agent.id, "didn't correctly exit tube  0000")
                # agent.previous_phase = 0
                return 0  # Post-tube phase
        # Default: Phase 0
        # agent.previous_phase = 0
        return 0

    def initialize_min_time_distance_graph(self, world):
        for agent in world.agents:
            self.min_time(agent, world)
        world.calculate_distances()
        self.update_graph(world)

    def initialize_landmarks_group_reached_goal(self, world):
        # create a self.landmarks_group that has different sets of landmarks for each agent
        self.landmarks_group = np.array(np.split(np.array(world.landmarks), self.num_agents)).T
        # print("landmarks_group",self.landmarks_group)
        # create a flag that tells if the agent has reached the goal in a particular set of landmarks
        self.reached_goal = np.zeros(self.num_agents)

    def get_agent_reached_goal(self, agent:Agent) -> int:
        return self.reached_goal[agent.id]

    def info_callback(self, agent:Agent, world:World) -> tuple:
        # TODO modify this 

        world.dists = np.array([np.linalg.norm(agent.state.p_pos - l.state.p_pos) for l in world.landmarks])
        
        nearest_landmark = np.argmin(world.dists)
        dist_to_goal = world.dists[nearest_landmark]
        if dist_to_goal < self.min_dist_thresh and (nearest_landmark != self.goal_reached[agent.id] and self.goal_reached[agent.id] != -1):
            # print("AGENT", agent.id, "reached NEW goal",world.dist_left_to_goal[agent.id])
            self.goal_reached[agent.id] = nearest_landmark
            world.dist_left_to_goal[agent.id] = dist_to_goal
        # only update times_required for the first time it reaches the goal
        if dist_to_goal < self.min_dist_thresh and (world.times_required[agent.id] == -1):
            # print("agent", agent.id, "reached goal",world.dist_left_to_goal[agent.id])
            world.times_required[agent.id] = world.current_time_step * world.dt
            world.dists_to_goal[agent.id] = agent.state.p_dist
            world.dist_left_to_goal[agent.id] = dist_to_goal
            self.goal_reached[agent.id] = nearest_landmark
            # print("dist to goal",world.dists_to_goal[agent.id],world.dists_to_goal)
        if world.times_required[agent.id] == -1:
            # print("agent", agent.id, "not reached goal yet",world.dist_left_to_goal[agent.id])
            world.dists_to_goal[agent.id] = agent.state.p_dist
            world.dist_left_to_goal[agent.id] = dist_to_goal
        if dist_to_goal > self.min_dist_thresh and world.times_required[agent.id] != -1:
            # print("AGENT", agent.id, "left goal",dist_to_goal,world.dist_left_to_goal[agent.id])
            world.dists_to_goal[agent.id] = agent.state.p_dist
            world.times_required[agent.id] = world.current_time_step * world.dt
            world.dist_left_to_goal[agent.id] = dist_to_goal
        if dist_to_goal < self.min_dist_thresh and (nearest_landmark == self.goal_reached[agent.id]):
            # print("AGENT", agent.id, "on SAME goal",world.dist_left_to_goal[agent.id])
            ## TODO: How to update time as well?
            world.dist_left_to_goal[agent.id] = dist_to_goal
            self.goal_reached[agent.id] = nearest_landmark

            # print("dist is",world.dists_to_goal[agent.id])

        if agent.collide:
            if self.is_obstacle_collision(agent.state.p_pos, agent.size, world):
                world.num_obstacle_collisions[agent.id] += 1
            for a in world.agents:
                if a is agent:
                    self.agent_dist_traveled[a.id] = a.state.p_dist
                    self.agent_time_taken[a.id] = a.state.time
                    continue
                if self.is_collision(agent, a):
                    world.num_agent_collisions[agent.id] += 1


        world.dist_traveled_mean = np.mean(world.dists_to_goal)
        world.dist_traveled_stddev = np.std(world.dists_to_goal)
        # create the mean and stddev for time taken
        world.time_taken_mean = np.mean(world.times_required)
        world.time_taken_stddev = np.std(world.times_required)

        # print("CONFORMANCE",self.conformance_percent)
        # print("self.conformance_percent[agent.id]/self.args.episode_length,",self.conformance_percent[agent.id]/self.args.episode_length)
        # print("SPACING VIOLATION",self.spacing_violation)
        # print("DELTA SPACING",self.delta_spacing)
        # convert delta spacing to an array
        self.delta_spacing_sum = np.array(self.delta_spacing)
        #sum all the values in the list
        self.delta_spacing_sum = np.sum(self.delta_spacing, axis=0)
        # print("self.delta_spacing_sum",self.delta_spacing_sum)
        # print("self.spacing_violation",np.sum(self.spacing_violation))
        # print(self.delta_spacing[agent.id]/(self.spacing_violation[agent.id] if self.spacing_violation[agent.id] != 0 else 1))
        agent_info = {
            'Dist_to_goal': np.float32(world.dist_left_to_goal[agent.id]),  # make this into float using numpy float32
            'Time_req_to_goal': world.times_required[agent.id],
            # NOTE: total agent collisions is half since we are double counting. # EDIT corrected this.
            'Num_agent_collisions': world.num_agent_collisions[agent.id], 
            'Num_obst_collisions': world.num_obstacle_collisions[agent.id],
            'Distance_mean': world.dist_traveled_mean, 
            'Distance_variance': world.dist_traveled_stddev,
            'Mean_by_variance': world.dist_traveled_mean/(world.dist_traveled_stddev+0.0001),
            'Dists_traveled': world.dists_to_goal[agent.id],
            'Time_taken': world.times_required[agent.id],
            # Time mean and stddev
            'Time_mean': world.time_taken_mean,
            'Time_stddev': world.time_taken_stddev,
            'Time_mean_by_stddev': world.time_taken_mean/(world.time_taken_stddev+0.0001),
            'Conformance': self.conformance_percent[agent.id]/self.args.episode_length,
            'Delta_spacing': self.delta_spacing_sum /(np.sum(self.spacing_violation) if np.sum(self.spacing_violation) != 0 else 1),
            'Spacing_violations': self.spacing_violation[agent.id]/(self.steps_in_corridor[agent.id] if self.steps_in_corridor[agent.id] != 0 else 1),
            'Phase_reached': self.phase_reached[agent.id],


        }
        if self.max_speed is not None:
            agent_info['Min_time_to_goal'] = agent.goal_min_time
        return agent_info

    # ======== Layered-Safe-MARL style helpers ========

    def get_agent_current_goal(self, agent, world):
        """Return a lightweight goal object with position, heading, and speed."""
        gid = self.goal_match_index[agent.id]
        goal = type("TmpGoal", (), {})()
        goal.state = type("TmpState", (), {})()
        goal.state.p_pos = self.landmark_poses[gid]
        vec = goal.state.p_pos - agent.state.p_pos
        goal.heading = float(np.arctan2(vec[1], vec[0]))
        # For AirTaxi, midrange cruise speed
        try:
            from multiagent.config import AirTaxiConfig
            goal.speed = 0.5 * (AirTaxiConfig.V_MIN + AirTaxiConfig.V_MAX)
        except Exception:
            goal.speed = float(agent.state.speed)
        return goal

    def agent_distance_to_current_goal(self, agent, world):
        g = self.get_agent_current_goal(agent, world)
        return float(np.linalg.norm(agent.state.p_pos - g.state.p_pos))

    def evaluate_agent_goal_reached(self, agent, world):
        return self.agent_distance_to_current_goal(agent, world) < self.min_dist_thresh

    def get_effective_curriculum_ratio_sloped(self, start=0.0, end=1.0):
        return getattr(self, "curriculum_ratio", 1.0) * (end - start) + start

    def get_effective_curriculum_ratio_stair(self):
        return 1.0


    # check collision of entity with obstacles and walls
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


    # get min time required to reach to goal without obstacles
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

    def reward(self, agent: Agent, world: World) -> float:
        rew = 0.0
        current_phase = self.get_agent_phase(agent, world)
        # print("Agent", agent.id, "phase", current_phase, "previous_phase", agent.previous_phase, "phase_reached", self.phase_reached[agent.id])
        # print("Goalrew",self.goal_rew, "Collisionrew",self.collision_rew)
        # Common rewards across all phases
        # Collision penalties
        # if agent.collide:
        # 	for a in world.agents:
        # 		if a.id == agent.id:
        # 			continue
        # 		if self.is_collision(a, agent):
        # 			rew -= self.collision_rew
        # 			# print(f"!!!Agent {agent.id} collided with agent {a.id} penalty",self.collision_rew*4 )
        # 			# print(" self.separation_distance", self.separation_distance)
        # 			# input("Collision")
            
        # 	if self.is_obstacle_collision(pos=agent.state.p_pos,
        # 								entity_size=agent.size, 
        # 								world=world):
        # 		rew -= self.collision_rew
        # 		# print(f"Agent {agent.id} collided with obstacle")

        # Calculate tube length
        tube_direction = world.tube_params['exit'] - world.tube_params['entrance']
        tube_length = np.linalg.norm(tube_direction)
        agent_pos = agent.state.p_pos
        agent_heading = agent.state.theta
        heading_vec = np.array([np.cos(agent_heading), np.sin(agent_heading)])
        s, y, L, half_w = self._tube_coords(world, agent_pos)
        front_agents = []
        back_agents = []
        # print("self._in_entrance_gate(s, y, L, half_w)",self._in_entrance_gate(s, y, L, half_w))
        for other in world.agents:
            if other is agent:
                continue
            rel_vec = other.state.p_pos - agent_pos
            proj = np.dot(rel_vec, heading_vec)
            if proj > 0:
                front_agents.append((proj, other))
            else:
                back_agents.append((proj, other))

        # Get closest in front and back
        front_agent = min(front_agents, key=lambda x: x[0])[1] if front_agents else None
        back_agent = max(back_agents, key=lambda x: x[0])[1] if back_agents else None

        # print("front_agent", front_agent.id if front_agent else "None")
        # print("back_agent", back_agent.id if back_agent else "None")
        
        # Calculate desired spacing based on tube length and number of agents
        desired_spacing = self.separation_distance  # 3 is the number of agents in the tube TODO: harcoded
        
        # Track agent's previous phase if not already stored
        if not hasattr(agent, 'previous_phase'):
            agent.previous_phase = 0
        
        # ##Modified phase transition handling
        if current_phase == 2 and current_phase > agent.previous_phase + 1:
            # Only penalize clear phase skips (e.g., 0 to 2)
            # print("self.goal_rew",self.goal_rew)
            rew -= self.goal_rew  #*3  # Reduced penalty
            # print(f"Agent {agent.id} penalized for skipping from phase {agent.previous_phase} to {current_phase} rew", rew)
        tube_direction_vector = tube_direction / tube_length
        entrance_to_agent = agent.state.p_pos - world.tube_params['entrance']
        proj = np.dot(entrance_to_agent, tube_direction_vector)
        # entrance_dist = np.linalg.norm(entrance_to_agent - proj * tube_direction)

        if current_phase == agent.previous_phase+1 and self.phase_reached[agent.id] == current_phase-1:
            # Reward proper phase progression
            # if current_phase == 1 and 0 <= proj < 0.1 * tube_length and entrance_dist <  0.2 * tube_length and (self.entry_reward_cooldown[agent.id] == 0):
            if current_phase == 1 and self._in_entrance_gate(s, y, L, half_w) and (self.entry_reward_cooldown[agent.id] == 0):
                # print("Agent pos, entrance, exit", agent.state.p_pos, world.tube_params['entrance'], world.tube_params['exit'])
                # print("distance check", proj, entrance_dist)

                # Reward if agent moves into tube after exiting
                rew += self.goal_rew  # *3  # Positive reward for proper transition
                self.entry_reward_cooldown[agent.id] = self.phase_reward_cooldown_steps  # Cooldown period to prevent repeated rewards
                self.phase_reached[agent.id] = 1  # Mark Phase 1 completed
                # print(f"Agent {agent.id} properly progressed from phase {agent.previous_phase} to {current_phase} rew", rew)
            elif current_phase == 2:
                # Rewards if agent moves out of tube
                # print("Agent", agent.id, " in post-tube phase. Status:",  agent.status)
                rew += self.goal_rew  # *3
                self.phase_reached[agent.id] = 2  # Mark Phase 2 completed
                # print("Agent",agent.id,"reached fair goal")
                if agent.status is False:
                    agent.status = True
                    agent.state.reset_velocity()
                    rew += self.goal_rew * 5
                # print(f"Agent {agent.id} properly progressed from phase {agent.previous_phase} to {current_phase} rew", rew)
                # Update the global phase tracker if any agent progresses

        # Phase-specific rewards
        # print("Agent",agent.id,"current_phase",current_phase,"prev phase_reached",self.phase_reached)
        if current_phase == 0:  # Pre-tube phase
            # Reward for getting closer to tube entrance
            s, y, L, half_w = self._tube_coords(world, agent.state.p_pos)
            # print("Agent", agent.id, " Phase 0 s,y,L,half_w:", s, y, L, half_w)
            dist_to_entrance_edge = self._entrance_gate_distance(s, y, half_w)
            rew -= dist_to_entrance_edge

            # === NEW: Heading alignment reward ===
            # Desired heading: align with corridor direction
            corridor_vec = world.tube_params['e']  # unit vector along corridor
            corridor_heading = np.arctan2(corridor_vec[1], corridor_vec[0])
            # print("corridor_heading (deg):", corridor_heading*180/np.pi, "angle", world.tube_params['angle']*180/np.pi)
            agent_heading = agent.state.theta
            heading_error = abs((agent_heading - corridor_heading + np.pi) % (2*np.pi) - np.pi)

            # Only enforce alignment when close to entrance (within 0.5*world_size)
            if dist_to_entrance_edge < self.world_size * 0.1:
                # Penalize heading misalignment (max penalty at 180°, none at 0°)
                rew -= heading_error * self.formation_rew * 0.5
            # === Penalize lateral approach (agents must approach along corridor axis) ===
            # If agent is laterally offset from entrance but trying to enter: penalize
            # if abs(y) > half_w * 0.8 and s < 0 and s > -self.world_size * 0.1:
            # 	# Agent is beside entrance but close: penalize
            # 	# print("s", s, "-self.world_size * 0.1", -self.world_size * 0.1, "y", y, "half_w*0.7", half_w*0.7)
            # 	rew -= abs(y) * self.collision_rew * 0.05
                # print(f"Agent {agent.id} penalized for lateral approach y={y:.2f} rew", rew)
                # input("Lateral approach penalty applied")
        elif current_phase == 1:  # In-tube phase
            spacing_error = 0
            max_spacing_error = 0

            if front_agent:
                # print("np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos)", np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos), "desired_spacing", desired_spacing)
                diff = np.linalg.norm(front_agent.state.p_pos - agent.state.p_pos) - desired_spacing
                # print("diff", diff)
                spacing_error += np.abs(diff) if diff < 0 else 0
                max_spacing_error = max(max_spacing_error, np.abs(diff))
            if back_agent:
                # print("np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos)", np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos), "desired_spacing", desired_spacing)
                diff = np.linalg.norm(back_agent.state.p_pos - agent.state.p_pos) - desired_spacing
                # print("diff", diff)
                max_spacing_error = max(max_spacing_error, np.abs(diff))
                spacing_error += np.abs(diff) if diff < 0 else 0
            if spacing_error > 0:
                # print("Phase 1 spacing_error",spacing_error)
                self.spacing_violation[agent.id] += 1
            rew -= spacing_error * self.formation_rew  # Higher weight for maintaining formation in tube
            dist_to_exit_edge = self._exit_gate_distance(s, y, L, half_w)
            rew -= dist_to_exit_edge
            self.steps_in_corridor[agent.id] += 1

            corridor_vec = world.tube_params['e']  # unit vector along corridor
            corridor_heading = np.arctan2(corridor_vec[1], corridor_vec[0])
            # print("corridor_heading (deg):", corridor_heading*180/np.pi, "angle", world.tube_params['angle']*180/np.pi)
            agent_heading = agent.state.theta
            heading_error = abs((agent_heading - corridor_heading + np.pi) % (2*np.pi) - np.pi)
            rew -= heading_error * self.formation_rew * 0.1




        # print("Agent.status",agent.status)
        if self.phase_reached[agent.id] == 1 and current_phase == 0:
            # print("Agent",agent.id,"left corridor")
            self.conformance_percent[agent.id] += 1
            # print("conformance_percent",self.conformance_percent[agent.id])

        if current_phase > self.phase_reached[agent.id]:
            # print(f"Agent {agent.id} reached phase {current_phase}")
            self.phase_reached[agent.id] = current_phase  # Update max phase reached globally
        ## penalize for moving from higher phase to lower phase
        if current_phase < agent.previous_phase:
            # print(f"Agent {agent.id} tried to move back to phase {current_phase} from {agent.previous_phase}")
            rew -= self.collision_rew  #*4
            # print(f"Agent {agent.id} tried to move back to phase {current_phase} from {agent.previous_phase} rew", rew)
        if current_phase < self.phase_reached[agent.id]:
            rew -= self.collision_rew
            # print(f"Agent {agent.id} tried to move back to phase {current_phase} from {self.phase_reached[agent.id]} rew", rew)
        # Store current phase for next step
        agent.previous_phase = current_phase
        if self._in_tube_rect(s, y, L, half_w) and not current_phase == 1:
            rew -= self.collision_rew  #*2
            # print(f"Agent {agent.id} is in tube but not in phase 1 rew", rew)

        # If agent is past exit plane but never entered corridor: continuous penalty
        if s > L and self.phase_reached[agent.id] < 1:
            rew -= self.goal_rew
            # print(f"Agent {agent.id} skipped corridor (s={s:.2f} > L={L:.2f}): penalty {self.goal_rew}")
        
        # print(f"Agent {agent.id} total reward: ", rew)
        # input("Reward calculation complete for agent {}".format(agent.id))
        # input("Press Enter to continue...")

        return np.clip(rew, -4*self.collision_rew, self.goal_rew*5)

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

    def get_id(self, agent:Agent) -> arr:
        return np.array([agent.global_id])
    
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
