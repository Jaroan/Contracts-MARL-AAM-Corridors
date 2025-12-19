# multiagent/custom_scenarios/aam_train.py

import numpy as np
from multiagent.custom_scenarios.safe_aam_scenario import SafeAamScenario


class Scenario(SafeAamScenario):
    """
    TRAINING SCENARIO (Randomized)
    --------------------------------
    This wraps SafeAamScenario and adds:

      - random rotation of the corridor
      - random corridor width scaling
      - random waypoint jitter
      - random initial agent positions
      - random landmark perturbations

    Everything else (fairness reward, safety reward, RI obs)
    comes from AamScenario + SafeAamScenario.
    """

    def __init__(self, args):
        super().__init__(args)

        # You can tune these if needed (Jasmine may tweak these):
        self.max_rotation_deg = 30.0          # random corridor rotation
        self.corridor_width_scale_range = (0.8, 1.2)
        self.landmark_jitter = 0.15
        self.start_jitter = 0.25

    # ------------------------------------------------------------
    # MAIN WORLD SETUP
    # ------------------------------------------------------------
    def make_world(self, args):
        """
        Build the world using SafeAamScenario, then add training randomness.
        """
        world = super().make_world(args)

        # Apply training-time randomness
        self.randomize_corridor(world)
        self.randomize_start_positions(world)
        self.randomize_landmarks(world)

        return world

    # ------------------------------------------------------------
    # RANDOMIZE CORRIDOR GEOMETRY
    # ------------------------------------------------------------
    def randomize_corridor(self, world):
        """
        Random rotation and scaling of the corridor tube.

        Assumes your existing AamScenario already uses:
            self.world_rotation_angle
            self.corridor_width
        and that update_graph/observations apply rotation-invariant frame.
        """

        # 1) Random rotation angle
        max_angle_rad = np.deg2rad(self.max_rotation_deg)
        self.world_rotation_angle = np.random.uniform(-max_angle_rad, max_angle_rad)

        # 2) Random corridor width scale
        w_min, w_max = self.corridor_width_scale_range
        self.corridor_width *= np.random.uniform(w_min, w_max)

    # ------------------------------------------------------------
    # RANDOMIZE START POSITIONS
    # ------------------------------------------------------------
    def randomize_start_positions(self, world):
        """
        Add jitter to initial agent positions.
        AamScenario already places agents; here we just perturb them.
        """
        for agent in world.agents:
            agent.state.p_pos += self.start_jitter * (np.random.rand(2) * 2 - 1)

    # ------------------------------------------------------------
    # RANDOMIZE LANDMARKS
    # ------------------------------------------------------------
    def randomize_landmarks(self, world):
        """
        Add small jitter to landmark positions.
        """
        for lm in world.landmarks:
            lm.state.p_pos += self.landmark_jitter * (np.random.rand(2) * 2 - 1)
