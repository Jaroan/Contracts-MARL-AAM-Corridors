# multiagent/custom_scenarios/aam_eval.py

from multiagent.custom_scenarios.safe_aam_scenario import SafeAamScenario
import numpy as np


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

    def __init__(self, args):
        super().__init__()

    # ------------------------------------------------------------
    # MAIN WORLD SETUP
    # ------------------------------------------------------------
    def make_world(self, args):
        """
        Build the world using SafeAamScenario, then apply FIXED settings.
        """
        world = super().make_world(args)

        self.fix_corridor(world)
        self.fix_start_positions(world)
        self.fix_landmarks(world)

        return world

    # ------------------------------------------------------------
    # FIXED CORRIDOR
    # ------------------------------------------------------------
    def fix_corridor(self, world):
        """
        No randomness: use canonical corridor parameters.
        Angle = 0, width = default.
        """
        self.world_rotation_angle = 0.0
        # Do not change corridor_width — use default from AamScenario

    # ------------------------------------------------------------
    # FIXED INITIAL POSITIONS
    # ------------------------------------------------------------
    def fix_start_positions(self, world):
        """
        Canonical left-to-right corridor start layout.
        """
        num = self.num_agents
        ys = np.linspace(-0.4 * self.world_size, 0.4 * self.world_size, num)
        x0 = -0.9 * self.world_size

        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.array([x0, ys[i]])
            agent.state.reset_velocity(theta=0)
            agent.done = False

    # ------------------------------------------------------------
    # FIXED LANDMARK POSITIONS
    # ------------------------------------------------------------
    def fix_landmarks(self, world):
        """
        Canonical 2-waypoint corridor:
            1. midpoint
            2. end of corridor
        """

        # AamScenario already creates correct number of landmarks per agent
        # Here we place them in fixed positions.
        num = self.num_agents
        L = self.num_landmarks

        # midpoint (0, 0)
        # terminus (world_size/2, 0)
        y_spacing = 0.0  # simple corridor

        xs = np.array([0.0, 0.9 * self.world_size])
        ys = np.array([y_spacing, y_spacing])

        # one set for all agents (RI handles individuality)
        landmark_positions = [np.array([xs[k], ys[k]]) for k in range(L)]

        for lm, pos in zip(world.landmarks, landmark_positions):
            lm.state.p_pos = pos
            lm.state.stop()
