"""
BaseScenario for AAM-style multi-agent tasks.
This class provides ONLY the generic, scenario-independent structure:
- make_world()
- reset_world()
- entity creation
- abstract API hooks

All corridor logic, fairness logic, and safety logic MUST be implemented
in subclasses (AamScenario and SafeAamScenario), not here.

This cleanly matches Layered-Safe-MARL’s structure.
"""

import numpy as np
from multiagent.scenario import BaseScenario
from multiagent.core import World, Agent, Landmark, Wall, Entity, EntityDynamicsType
from multiagent.config import AirTaxiConfig, RewardWeightConfig, RewardBinaryConfig


class AAMBaseScenario(BaseScenario):
    def __init__(self, args):
        super().__init__()

        # Store args for later use
        self.args = args

        # -----------------------------
        # WORLD / SCENARIO PARAMETERS
        # -----------------------------
        self.world_size       = getattr(args, "world_size", 2.0)
        self.num_agents       = getattr(args, "num_agents", 3)
        self.num_landmarks    = getattr(args, "num_landmarks", 3)
        self.num_obstacles    = getattr(args, "num_obstacles", 0)

        # corridor geometry
        self.corridor_width   = getattr(args, "corridor_width", 1.0)
        self.goal_x_threshold = getattr(args, "goal_x_threshold", 0.75 * self.world_size)

        # dynamics defaults
        self.goal_speed_max   = getattr(args, "max_speed", 1.0)
        self.goal_speed_min   = getattr(args, "min_speed", 0.1)

        # collision
        self.use_collisions     = getattr(args, "use_collisions", True)
        self.collision_penalty  = getattr(args, "collision_rew", -5.0)

        # fairness
        self.use_fairness       = getattr(args, "increase_fairness", False)
        self.fairness_penalty   = getattr(args, "fair_rew", -1.0)

        # safety settings will be used by SafeAamScenario
        self.separation_distance = getattr(args, "min_dist_thresh", 0.3)

        # graph obs flags
        self.graph_feat_type = getattr(args, "graph_feat_type", "relative")


    def make_world(self, args):
        """Construct the world & create agents/landmarks/obstacles/walls."""
        self.args = args
        world = World(
            dynamics_type=self._resolve_dynamics_type(args),
            all_args=args
        )

        # === Create agents ===
        world.agents = [Agent(world.dynamics_type) for _ in range(args.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.id = i

        # === Create landmarks ===
        world.landmarks = [Landmark() for _ in range(args.num_landmarks)]
        for i, lm in enumerate(world.landmarks):
            lm.name = f"landmark_{i}"
            lm.collide = False
            lm.movable = False
            lm.id = i

        # === Create obstacles ===
        world.obstacles = [Landmark() for _ in range(args.num_obstacles)]
        for i, obs in enumerate(world.obstacles):
            obs.name = f"obstacle_{i}"
            obs.collide = True
            obs.movable = False
            obs.id = i

        # === Create walls ===
        world.walls = [Wall() for _ in range(args.num_walls)]
        for i, wall in enumerate(world.walls):
            wall.name = f"wall_{i}"
            wall.collide = True
            wall.movable = False
            wall.id = i

        # === Initialize world ===
        world.dim_c = 2
        world.world_size = args.world_size
        world.world_aspect_ratio = 1.0

        # Subclass will place agents + landmarks
        self.reset_world(world)
        return world

    # ===============================================================
    # GRAPH UPDATE HOOK (for GraphMPEEnv)
    # ===============================================================
    def update_graph(self, world):
        """
        Default graph update for GraphMPE:

        For now, we just build a fully-connected, undirected graph over
        all entities (agents + landmarks). If the underlying graph env
        expects something more structured, we can refine this later.

        We assume the GraphMPE env will either:
          - read world.graph_edges / world.graph_adjacency, or
          - ignore the return value (no-op graph).

        This keeps training from crashing due to a missing attribute.
        """
        # Try to be minimally compatible without assuming too much
        entities = list(world.agents) + list(world.landmarks)
        n = len(entities)

        # Fully-connected (no self loops) adjacency
        adj = np.ones((n, n), dtype=np.float32) - np.eye(n, dtype=np.float32)

        # Store on world so the env can grab it if it wants
        world.graph_adjacency = adj
        return adj


    # -----------------------------------------------------------

    def _resolve_dynamics_type(self, args):
        """Resolve dynamics type from args."""
        if args.dynamics_type == "unicycle_vehicle":
            return EntityDynamicsType.UnicycleVehicleXY
        elif args.dynamics_type == "double_integrator":
            return EntityDynamicsType.DoubleIntegratorXY
        elif args.dynamics_type == "air_taxi":
            return EntityDynamicsType.AirTaxiXY
        else:
            raise ValueError(f"Unknown dynamics type {args.dynamics_type}")

    # -----------------------------------------------------------

    def reset_world(self, world):
        """
        Default method: zero velocities, no specific positions.
        Subclasses MUST override to implement:
            - corridor geometry
            - fairness init
            - safety filtering init
            - placement of agents/landmarks
        """
        for agent in world.agents:
            agent.state.p_pos = np.zeros(2)
            agent.state.p_vel = np.zeros(2)
            agent.state.c = np.zeros(world.dim_c)
            agent.status = False

        for lm in world.landmarks:
            lm.state.p_pos = np.zeros(2)

        # For safety scenarios, SafeAamScenario.reset_world() extends this.

        # Scenario-specific placement:
        self.random_scenario(world)

    # -----------------------------------------------------------

    def random_scenario(self, world):
        """Abstract hook – subclasses MUST implement this."""
        raise NotImplementedError

    # -----------------------------------------------------------

    def reward(self, agent, world):
        """Base class defines NO reward. Must be overridden."""
        raise NotImplementedError

    def observation(self, agent, world):
        """Base class defines NO observation. Must be overridden."""
        raise NotImplementedError

    def graph_observation(self, agent, world):
        """Optional; overridden if using GNN features."""
        raise NotImplementedError

    def info_callback(self, agent, world):
        """Optional; overridden in fairness/safety scenarios."""
        return {}
