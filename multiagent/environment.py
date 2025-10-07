import gym
from gym import spaces
import numpy as np
import math
import random
from typing import Callable, List, Tuple, Dict, Union, Optional
from multiagent.core import EntityDynamicsType, World, Agent, is_list_of_lists
from multiagent.multi_discrete import MultiDiscrete
from multiagent.config import DoubleIntegratorConfig, UnicycleVehicleConfig

# update bounds to center around agent
cam_range = 2

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentBaseEnv(gym.Env):
	"""
		Base environment for all multi-agent environments
	"""
	metadata = {
		'render.modes' : ['human', 'rgb_array']
	}

	def __init__(self, world:World, reset_callback:Callable=None, 
					reward_callback:Callable=None,
					observation_callback:Callable=None, 
					info_callback:Callable=None,
					done_callback:Callable=None,
					agent_reached_goal_callback:Callable=None,
					shared_viewer:bool=True, 
					discrete_action:bool=True,
					scenario_name:str='navigation',
					dynamics_type:str='unicycle_vehicle') -> None:
		self.world = world
		self.world_length = self.world.world_length
		self.current_step = 0
		self.agents = self.world.policy_agents
		# set required vectorized gym env property
		self.n = len(world.policy_agents)
		self.num_agents = len(world.policy_agents)  # for compatibility with offpolicy baseline envs
		# scenario callbacks
		self.reset_callback = reset_callback
		self.reward_callback = reward_callback
		self.observation_callback = observation_callback
		self.info_callback = info_callback
		self.done_callback = done_callback
		self.agent_reached_goal_callback = agent_reached_goal_callback
		# print("done_callback: ", done_callback)
		self.scenario_name = scenario_name

		self.world_size = self.world.world_size

		# environment parameters
		# self.discrete_action_space = True
		self.discrete_action_space = discrete_action

		if dynamics_type == 'unicycle_vehicle':
			self.dynamics_type = EntityDynamicsType.UnicycleVehicleXY
			self.num_accel_options = UnicycleVehicleConfig.MOTION_PRIM_ACCEL_OPTIONS
			self.num_angle_rate_options = UnicycleVehicleConfig.MOTION_PRIM_ANGRATE_OPTIONS
			self.num_discrete_action = self.num_accel_options * self.num_angle_rate_options
		elif dynamics_type == 'double_integrator':
			self.dynamics_type = EntityDynamicsType.DoubleIntegratorXY

			# self.num_accel_y_options = DoubleIntegratorConfig.ACCELY_OPTIONS
			# self.num_accel_x_options = DoubleIntegratorConfig.ACCELX_OPTIONS
			# self.num_discrete_action = self.num_accel_x_options * self.num_accel_y_options
			self.num_discrete_action = self.world.total_actions
		else:
			raise NotImplementedError
		# if true, action is a number 0...N, 
		# otherwise action is a one-hot N-dimensional vector
		self.discrete_action_input = False
		# if true, even the action is continuous, 
		# action will be performed discretely
		self.force_discrete_action = world.discrete_action if hasattr(world, 
												'discrete_action') else False
		# if true, every agent has the same reward
		self.shared_reward = world.collaborative if hasattr(world, 
													'collaborative') else False
		self.time = 0

		# configure spaces
		self.action_space = []
		self.observation_space = []
		self.share_observation_space = []   # adding this for compatibility with MAPPO code
		share_obs_dim = 0
		if is_list_of_lists(world.agents):
			for team in world.agents:
				for agent in team:
					total_action_space = []

					# physical action space
					if self.discrete_action_space:
						u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
					else:
						u_action_space = spaces.Box(low=-agent.u_range, 
													high=+agent.u_range, 
													shape=(world.dim_p,), 
													dtype=np.float32)
					if agent.movable:
						total_action_space.append(u_action_space)

					# communication action space
					if self.discrete_action_space:
						c_action_space = spaces.Discrete(world.dim_c)
					else:
						c_action_space = spaces.Box(low=0.0, 
													high=1.0, 
													shape=(world.dim_c,), 
													dtype=np.float32)

					if not agent.silent:
						total_action_space.append(c_action_space)
					# total action space
					if len(total_action_space) > 1:
						# all action spaces are discrete, 
						# so simplify to MultiDiscrete action space
						if all([isinstance(act_space, spaces.Discrete) 
								for act_space in total_action_space]):
							act_space = MultiDiscrete([[0, act_space.n - 1] 
												for act_space in total_action_space])
						else:
							act_space = spaces.Tuple(total_action_space)
						self.action_space.append(act_space)
					else:
						self.action_space.append(total_action_space[0])

					# observation space
					# for original MPE Envs like simple_spread, simple_reference, etc.
					if 'simple' in self.scenario_name:
						obs_dim = len(observation_callback(agent=agent, world=self.world))
					else:
						obs_dim = len(observation_callback(agent=agent, world=self.world))
					share_obs_dim += obs_dim
					self.observation_space.append(spaces.Box(low=-np.inf, 
															high=+np.inf, 
															shape=(obs_dim,), 
															dtype=np.float32))

					agent.action.c = np.zeros(self.world.dim_c)
		else:
			for agent in world.agents:
				total_action_space = []
				# physical action space
				if self.discrete_action_space:
					u_action_space = spaces.Discrete(self.num_discrete_action)
				else:
					u_action_space = spaces.Box(low=-agent.u_range, 
												high=+agent.u_range, 
												shape=(world.dim_p,), 
												dtype=np.float32)
				if agent.movable:
					total_action_space.append(u_action_space)

				# communication action space
				if self.discrete_action_space:
					c_action_space = spaces.Discrete(world.dim_c)
				else:
					c_action_space = spaces.Box(low=0.0, 
												high=1.0, 
												shape=(world.dim_c,), 
												dtype=np.float32)

				if not agent.silent:
					total_action_space.append(c_action_space)
				# total action space
				if len(total_action_space) > 1:
					# all action spaces are discrete, 
					# so simplify to MultiDiscrete action space
					if all([isinstance(act_space, spaces.Discrete) 
							for act_space in total_action_space]):
						act_space = MultiDiscrete([[0, act_space.n - 1] 
											for act_space in total_action_space])
					else:
						act_space = spaces.Tuple(total_action_space)
					self.action_space.append(act_space)
				else:
					self.action_space.append(total_action_space[0])

				# observation space
				# for original MPE Envs like simple_spread, simple_reference, etc.
				if 'simple' in self.scenario_name:
					obs_dim = len(observation_callback(agent=agent, world=self.world))
				else:
					obs_dim = len(observation_callback(agent=agent, world=self.world))
				share_obs_dim += obs_dim
				self.observation_space.append(spaces.Box(low=-np.inf, 
														high=+np.inf, 
														shape=(obs_dim,), 
														dtype=np.float32))

				agent.action.c = np.zeros(self.world.dim_c)			
		
		self.share_observation_space = [spaces.Box(low=-np.inf, 
													high=+np.inf, 
													shape=(share_obs_dim,), 
													dtype=np.float32) 
													for _ in range(self.n)]
		

		# rendering
		self.shared_viewer = shared_viewer
		if self.shared_viewer:
			self.viewers = [None]
		else:
			self.viewers = [None] * self.n
		self._reset_render()

	def seed(self, seed=None):
		if seed is None:
			np.random.seed(1)
		else:
			np.random.seed(seed)
	
	def step(self, action_n:List):
		raise NotImplementedError

	def reset(self):
		raise NotImplementedError

	# get info used for benchmarking
	def _get_info(self, agent:Agent) -> Dict:
		if self.info_callback is None:
			return {}
		return self.info_callback(agent, self.world)

	# get observation for a particular agent
	def _get_obs(self, agent:Agent) -> np.ndarray:
		if self.observation_callback is None:
			return np.zeros(0)
		# for original MPE Envs like simple_spread, simple_reference, etc.
		if 'simple' in self.scenario_name:
			return self.observation_callback(agent=agent, world=self.world)
		else:
			return self.observation_callback(agent=agent, world=self.world)

	# get shared observation for the environment
	def _get_shared_obs(self) -> np.ndarray:
		if self.shared_obs_callback is None:
			return None
		return self.shared_obs_callback(self.world)
		
	# # get dones for a particular agent
	# # unused right now -- agents are allowed to go beyond the viewing screen
	# def _get_done(self, agent:Agent) -> bool:
	# 	# print("self.done_callback: ", self.done_callback(agent, self.world))
	# 	if self.done_callback is None:
	# 		if self.current_step >= self.world_length:
	# 			return True
	# 		else:
	# 			return False
	# 	return self.done_callback(agent, self.world)

	def _get_done(self, agent, count = None):
		# print("get done count ",count )
		if agent.status ==True:
			return True
		if self.current_step >= self.world_length:
			return True
		else:
			return False


	# get reward for a particular agent
	def _get_reward(self, agent:Agent) -> float:
		if self.reward_callback is None:
			return 0.0
		return self.reward_callback(agent, self.world)

	def decode_action_index(self, action_index):
		""" action_index: numpy array of the index of the action in the action space
		"""
		if self.dynamics_type == EntityDynamicsType.UnicycleVehicleXY:
			max_angular_rate = UnicycleVehicleConfig.ANGULAR_RATE_MAX
			max_accel = UnicycleVehicleConfig.ACCEL_MAX
			min_accel = UnicycleVehicleConfig.ACCEL_MIN
			accel_options = np.linspace(min_accel, max_accel, self.num_accel_options)
			angle_rate_options = np.linspace(-max_angular_rate, max_angular_rate, self.num_angle_rate_options)
			angle_rate_index = action_index // self.num_accel_options
			accel_index = action_index % self.num_accel_options
			# Initialize the output array with the correct shape
			u = np.zeros((*action_index.shape, self.world.dim_p))

			# Use advanced indexing to fill in the values
			u[..., 0] = angle_rate_options[angle_rate_index]
			u[..., 1] = accel_options[accel_index]
		elif self.dynamics_type == EntityDynamicsType.DoubleIntegratorXY:

			#############################################
			## previous setup

			print("action_index: ", action_index)
			############################################
			## new setup
			# max_accel_x = DoubleIntegratorConfig.ACCELX_MAX
			# min_accel_x = DoubleIntegratorConfig.ACCELX_MIN
			# max_accel_y = DoubleIntegratorConfig.ACCELY_MAX
			# min_accel_y = DoubleIntegratorConfig.ACCELY_MIN
			# accel_x_options = np.linspace(min_accel_x, max_accel_x, self.num_accel_x_options)
			# accel_y_options = np.linspace(min_accel_y, max_accel_y, self.num_accel_y_options)
			# accel_x_index = action_index // self.num_accel_y_options #check
			# accel_y_index = action_index % self.num_accel_y_options #check

			# u = np.zeros((*action_index.shape, self.world.dim_p))

			# u[..., 0] = accel_x_options[accel_x_index]
			# u[..., 1] = accel_y_options[accel_y_index]
			#################################################
		else:
			raise NotImplementedError
		return u

	# set env action for a particular agent
	def _set_action(self, action, agent:Agent, action_space, 
					time:Optional=None) -> None:
		agent.action.u = np.zeros(self.world.dim_p)
		agent.action.c = np.zeros(self.world.dim_c)
		# process action
		if isinstance(action_space, MultiDiscrete):
			act = []
			size = action_space.high - action_space.low + 1
			index = 0
			for s in size:
				act.append(action[index:(index+s)])
				index += s
			action = act
			# print("multi discrete action space", action)
		else:
			if not isinstance(action, list):
				action = [action]

		# actions: [None, ←, →, ↓, ↑, comm1, comm2]
		if agent.movable:
			# physical action
			# print(f'discrete_action_input: {self.discrete_action_input}, force_discrete_action: {self.force_discrete_action}, discrete_action_space: {self.discrete_action_space}')
			action_description = ""

			if self.discrete_action_input:
				agent.action.u = np.zeros(self.world.dim_p)
				if self.world.total_actions ==5:
					# process discrete action
					if action[0] == 1: agent.action.u[0] = -1.0
					if action[0] == 2: agent.action.u[0] = +1.0
					if action[0] == 3: agent.action.u[1] = -1.0
					if action[0] == 4: agent.action.u[1] = +1.0
				elif self.world.total_actions ==9:
					# update action space for 9 actions using the 8 cardinal directions
					if action[0] == 1: agent.action.u[0] = -1.0	# left
					if action[0] == 2: agent.action.u[0] = -0.71; agent.action.u[1] = -0.71	# left-down
					if action[0] == 3: agent.action.u[1] = -1.0	# down
					if action[0] == 4: agent.action.u[0] = +0.71; agent.action.u[1] = -0.71	# right-down

					if action[0] == 5: agent.action.u[0] = +1.0	# right
					if action[0] == 6: agent.action.u[0] = +0.71; agent.action.u[1] = +0.71	# right-up
					if action[0] == 7: agent.action.u[1] = +1.0	# up
					if action[0] == 8: agent.action.u[0] = -0.71; agent.action.u[1] = +0.71	# left-up


			else:
				action_map = np.array([
									[0.0, 0.0],       # Action 0: No movement
									[-1.0, 0.0],      # Action 1: Left
									[-0.71, -0.71],   # Action 2: Left-Down
									[0.0, -1.0],      # Action 3: Down
									[0.71, -0.71],    # Action 4: Right-Down
									[1.0, 0.0],       # Action 5: Right
									[0.71, 0.71],     # Action 6: Right-Up
									[0.0, 1.0],       # Action 7: Up
									[-0.71, 0.71]     # Action 8: Left-Up
								])
				if self.force_discrete_action:
					d = np.argmax(action[0])
					action[0][:] = 0.0
					action[0][d] = 1.0
				if self.discrete_action_space:

					if self.dynamics_type == EntityDynamicsType.DoubleIntegratorXY:

						############################################
						## previous setup

						if len(action[0]) == 5:
							agent.action.u[0] += action[0][1] - action[0][2]
							agent.action.u[1] += action[0][3] - action[0][4]
						elif len(action[0]) == 9:
							active_action = np.argmax(action[0])  # find the index of the selected action
							agent.action.u = action_map[active_action]

						############################################
						##  new setup
						# accel_x_max = DoubleIntegratorConfig.ACCELX_MAX
						# accel_y_max = DoubleIntegratorConfig.ACCELY_MAX
						# accel_x_options = np.linspace(-accel_x_max, accel_x_max, self.num_accel_x_options)
						# accel_y_options = np.linspace(-accel_y_max, accel_y_max, self.num_accel_y_options)
						# action_index = np.argmax(action[0])
						# accel_x_index = int(action_index // self.num_accel_y_options)
						# accel_y_index = int(action_index - accel_x_index * self.num_accel_y_options)
						# agent.action.u[0] = accel_x_options[accel_x_index]
						# agent.action.u[1] = accel_y_options[accel_y_index]
						# action_description = f"accel x: {agent.action.u[0]}, y: {agent.action.u[1]}"
						#################################################
					elif self.dynamics_type == EntityDynamicsType.UnicycleVehicleXY:
						agent.action.u = np.zeros(self.world.dim_p)
						max_angular_rate = UnicycleVehicleConfig.ANGULAR_RATE_MAX
						max_accel = UnicycleVehicleConfig.ACCEL_MAX
						min_accel = UnicycleVehicleConfig.ACCEL_MIN
						accel_options = np.linspace(min_accel, max_accel, self.num_accel_options)
						angle_rate_options = np.linspace(-max_angular_rate, max_angular_rate, self.num_angle_rate_options)
						action_index = np.argmax(action[0])
						angle_rate_index = int(action_index // self.num_accel_options)
						accel_index = int(action_index - angle_rate_index * self.num_accel_options)
						agent.action.u[0] = angle_rate_options[angle_rate_index]
						agent.action.u[1] = accel_options[accel_index]
						action_description = f"turn: {agent.action.u[0]}, accel: {agent.action.u[1]}"
					else:
						raise NotImplementedError
					# if len(action[0]) == 5:
					# 	agent.action.u[0] += action[0][1] - action[0][2]
					# 	agent.action.u[1] += action[0][3] - action[0][4]
					# elif len(action[0]) == 9:
					# 	active_action = np.argmax(action[0])  # find the index of the selected action
					# 	agent.action.u = action_map[active_action]
				else:
					agent.action.u = action[0]
			sensitivity = 5.0
			if agent.accel is not None:
				sensitivity = agent.accel
			agent.action.u *= sensitivity
			# NOTE: refer offpolicy/envs/mpe/environment.py -> MultiAgentEnv._set_action() for non-silent agent
			action = action[1:]
		if not agent.silent:
			# communication action
			if self.discrete_action_input:
				agent.action.c = np.zeros(self.world.dim_c)
				agent.action.c[action[0]] = 1.0
			else:
				agent.action.c = action[0]
			action = action[1:]
		# make sure we used all elements of action
		assert len(action) == 0

	# reset rendering assets
	def _reset_render(self) -> None:
		self.render_geoms = None
		self.render_geoms_xform = None

	# render environment
	def render(self, mode:str='human', close:bool=False) -> List:
		if close:
			# close any existic renderers
			for i, viewer in enumerate(self.viewers):
				if viewer is not None:
					viewer.close()
				self.viewers[i] = None
			return []

		if mode == 'human':
			alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
			message = ''
			if is_list_of_lists(self.world.agents):
				for team in self.world.agents:
					for agent in team:
						comm = []
						for other in team:
							if other is agent: continue
							if np.all(other.state.c == 0):
								word = '_'
							else:
								word = alphabet[np.argmax(other.state.c)]
							message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
			else:
				for agent in self.world.agents:
					comm = []
					for other in self.world.agents:
						if other is agent: continue
						if np.all(other.state.c == 0):
							word = '_'
						else:
							word = alphabet[np.argmax(other.state.c)]
						message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
			# print(message)

		for i in range(len(self.viewers)):
			# create viewers (if necessary)
			if self.viewers[i] is None:
				# import rendering only if we need it 
				# (and don't import for headless machines)
				# from gym.envs.classic_control import rendering
				from multiagent import rendering
				self.viewers[i] = rendering.Viewer(700,700)

		# create rendering geometry
		if self.render_geoms is None:
			# import rendering only if we need it 
			# (and don't import for headless machines)
			# from gym.envs.classic_control import rendering
			from multiagent import rendering
			self.render_geoms = []
			self.render_geoms_xform = []

			self.comm_geoms = []
			# self.agent_separation_geoms = []
			# self.agent_separation_xforms = []
			for entity in self.world.entities:

				entity_comm_geoms = []

				if 'agent' in entity.name:
					# aircraft_img = rendering.Image('multiagent/aircraft3.png', entity.size*3, entity.size*3)
					# xform = rendering.Transform()
					# theta = entity.state.theta
					# entity.initial_theta = theta


					# # # Adjust anchor point to the center
					# # offset_x = entity.size*1.5
					# # offset_y = entity.size*1.5
					# # anchor = rendering.Transform()
					# # anchor.set_translation(offset_x, offset_y)
					# # aircraft_img.add_attr(anchor)

					# aircraft_img.add_attr(xform)
					# # Assign aircraft_img to geom to maintain consistency
					# geom = aircraft_img


					width = entity.size*2
					height = entity.size
					theta = entity.state.theta
					entity.initial_theta = theta
					# print("theta",theta)
					# Define the vertices relative to the center (0, 0)
					# vertices = np.array([
					# 	[-width/2, -height/2],
					# 	[width/2, -height/2],
					# 	[width/2, height/2],
					# 	[-width/2, height/2]
					# ])

					# # Create rotation matrix
					# rotation_matrix = np.array([
					# 	[np.cos(theta), -np.sin(theta)],
					# 	[np.sin(theta), np.cos(theta)]
					# ])

					# # Apply rotation to all vertices at once
					# rotated_vertices = np.dot(vertices, rotation_matrix.T)
					base_size = entity.size

					# # Define vertices for aircraft shape relative to center (0,0)
					# vertices = np.array([
					# 	[entity.size*2, 0],          # nose
					# 	[entity.size, entity.size],    # right wing tip
					# 	[0, entity.size/2],          # right wing inner
					# 	[-entity.size/2, entity.size/2], # tail right
					# 	[-entity.size, 0],           # tail center
					# 	[-entity.size/2, -entity.size/2], # tail left
					# 	[0, -entity.size/2],         # left wing inner
					# 	[entity.size, -entity.size],   # left wing tip
					# ])

					vertices = np.array([
						[2 * entity.size, 0],       # Nose
						[0.8 * entity.size, entity.size],    # Right wingtip
						[0.4 * entity.size, 0.4 * entity.size],  # Right wing inner
						[-0.5 * entity.size, 0.6 * entity.size], # Tail right
						[-entity.size, 0],          # Tail center
						[-0.5 * entity.size, -0.6 * entity.size], # Tail left
						[0.4 * entity.size, -0.4 * entity.size],  # Left wing inner
						[0.8 * entity.size, -entity.size],   # Left wingtip
					])


					# Create rotation matrix (if you want to keep initial rotation)
					rotation_matrix = np.array([
						[np.cos(theta), -np.sin(theta)],
						[np.sin(theta), np.cos(theta)]
					])

					# Apply rotation to vertices
					rotated_vertices = np.dot(vertices, rotation_matrix.T)
					geom = rendering.make_polygon(rotated_vertices)


					# geom = rendering.make_circle(entity.size)
					xform = rendering.Transform()
					geom.set_color(*entity.color, alpha=0.8)

					if not entity.silent:
						dim_c = self.world.dim_c
						# make circles to represent communication
						for ci in range(dim_c):
							comm = rendering.make_circle(entity.size / dim_c)
							comm.set_color(1, 1, 1)
							comm.add_attr(xform)
							offset = rendering.Transform()
							comm_size = (entity.size / dim_c)
							offset.set_translation(ci * comm_size * 2 -
													entity.size + comm_size, 0)
							comm.add_attr(offset)
							entity_comm_geoms.append(comm)
					# geom_separation = rendering.make_circle(entity.size*2)
					# geom_separation.set_color(0, 0, 0, alpha=0.1)
					# xform_separation = rendering.Transform()
					# geom_separation.add_attr(xform_separation)

				else:
					geom = rendering.make_circle(entity.size)
					xform = rendering.Transform()
					geom.set_color(*entity.color, alpha=0.1)
					if entity.channel is not None:
						dim_c = self.world.dim_c
						# make circles to represent communication
						for ci in range(dim_c):
							comm = rendering.make_circle(entity.size / dim_c)
							comm.set_color(1, 1, 1)
							comm.add_attr(xform)
							offset = rendering.Transform()
							comm_size = (entity.size / dim_c)
							offset.set_translation(ci * comm_size * 2 -
													entity.size + comm_size, 0)
							comm.add_attr(offset)
							entity_comm_geoms.append(comm)
				geom.add_attr(xform)
				self.render_geoms.append(geom)
				self.render_geoms_xform.append(xform)
				self.comm_geoms.append(entity_comm_geoms)
				# self.agent_separation_geoms.append(geom_separation)
				# self.agent_separation_xforms.append(xform_separation)

			for wall in self.world.walls:
				corners = ((wall.axis_pos - 0.5 * wall.width, wall.endpoints[0]),
						   (wall.axis_pos - 0.5 *
							wall.width, wall.endpoints[1]),
						   (wall.axis_pos + 0.5 *
							wall.width, wall.endpoints[1]),
						   (wall.axis_pos + 0.5 * wall.width, wall.endpoints[0]))
				if wall.orient == 'H':
					corners = tuple(c[::-1] for c in corners)
				geom = rendering.make_polygon(corners)
				if wall.hard:
					geom.set_color(*wall.color, alpha=0.1)
				else:
					geom.set_color(*wall.color, alpha=0.5)
				self.render_geoms.append(geom)


			# # # Add tube rendering
			# if hasattr(self.world, 'tube_params'):
			# 	# Calculate tube corners
			# 	entrance = self.world.tube_params['entrance']
			# 	exit = self.world.tube_params['exit']
			# 	width = self.world.tube_params['width']
				
			# 	# Create corners for tube rectangle
			# 	tube_corners = (
			# 		(entrance[0] - width/2, entrance[1]),  # top left
			# 		(entrance[0] + width/2, entrance[1]),  # top right
			# 		(exit[0] + width/2, exit[1]),         # bottom right
			# 		(exit[0] - width/2, exit[1])          # bottom left
			# 	)
				
			# 	# Create tube polygon
			# 	tube_geom = rendering.make_polygon(tube_corners)
			# 	tube_geom.set_color(0.8, 0.8, 0.8, alpha=0.2)  # Light gray, semi-transparent
			# 	self.render_geoms.append(tube_geom)
				
			# 	# Create tube outline
			# 	tube_outline = rendering.make_polygon(tube_corners, filled=False)
			# 	tube_outline.set_color(0.6, 0.6, 0.6, alpha=0.5)  # Darker gray for outline
			# 	self.render_geoms.append(tube_outline)

			# # Add tube rendering
			# if hasattr(self.world, 'tube_params'):
			# 	for tube in self.world.tube_params:
			# 		# Get tube parameters
			# 		entrance = tube['entrance']
			# 		exit = tube['exit']
			# 		width = tube['width']
			# 		angle = tube['angle']

			# 		# Create corners for tube rectangle
			# 		tube_corners = (
			# 			(entrance[0] - width / 2, entrance[1]),  # top left
			# 			(entrance[0] + width / 2, entrance[1]),  # top right
			# 			(exit[0] + width / 2, exit[1]),          # bottom right
			# 			(exit[0] - width / 2, exit[1])           # bottom left
			# 		)
			# 		print("tube_corners: ", tube_corners)
			# 		# Create tube polygon (semi-transparent)
			# 		tube_geom = rendering.make_polygon(tube_corners)
			# 		tube_geom.set_color(0.8, 0.8, 0.8, alpha=0.2)  # Light gray
			# 		self.render_geoms.append(tube_geom)

			# 		# Create tube outline (visible edges)
			# 		tube_outline = rendering.make_polygon(tube_corners, filled=False)
			# 		tube_outline.set_color(0.6, 0.6, 0.6, alpha=0.5)  # Darker gray
			# 		self.render_geoms.append(tube_outline)

			# Add tube rendering
			if hasattr(self.world, 'tube_params'):
				# for tube in self.world.tube_params:
					# print("tube: ", tube)
					tube= self.world.tube_params
					# Get tube parameters
					entrance = tube['entrance']
					exit = tube['exit']
					width = tube['width']
					angle = tube['angle']  # Rotation angle (in radians)
					# print("entrance: ", entrance, "exit: ", exit, "width: ", width, "angle: ", angle)
					# Compute the center of the tube
					center_x = (entrance[0] + exit[0]) / 2
					center_y = (entrance[1] + exit[1]) / 2
					center = np.array([center_x, center_y])
					# print("center: ", center)
					# Define unrotated tube corners relative to center
					half_length = np.linalg.norm(entrance - exit) / 2  # Half the tube length
					half_width = width / 2

					# Corner points before rotation (relative to center)
					unrotated_corners = np.array([
						[-half_width, half_length],   # Top-left
						[half_width, half_length],    # Top-right
						[half_width, -half_length],   # Bottom-right
						[-half_width, -half_length]   # Bottom-left
					])

					# print("unrotated_corners: ", unrotated_corners)
					# Create rotation matrix
					rotation_matrix = np.array([
						[np.cos(angle), np.sin(angle)],
						[-np.sin(angle), np.cos(angle)]
					])

					# Rotate corners
					rotated_corners = np.dot(unrotated_corners, rotation_matrix.T) + center

					# Extract rotated corners
					tube_corners = [tuple(corner) for corner in rotated_corners]
					# print("tube_corners: ", tube_corners)
					# Create tube polygon (semi-transparent)
					tube_geom = rendering.make_polygon(tube_corners)
					tube_geom.set_color(0.8, 0.8, 0.8, alpha=0.2)  # Light gray
					self.render_geoms.append(tube_geom)

					# Create tube outline (visible edges)
					tube_outline = rendering.make_polygon(tube_corners, filled=False)
					tube_outline.set_color(0.6, 0.6, 0.6, alpha=0.5)  # Darker gray
					self.render_geoms.append(tube_outline)
			# add geoms to viewer
			# for viewer in self.viewers:
			#     viewer.geoms = []
			#     for geom in self.render_geoms:
			#         viewer.add_geom(geom)

			for viewer in self.viewers:
				viewer.geoms = []
				for geom in self.render_geoms:
					viewer.add_geom(geom)
				for entity_comm_geoms in self.comm_geoms:
					for geom in entity_comm_geoms:
						viewer.add_geom(geom)
				# for geom in self.agent_separation_geoms:
				# 	if geom:
				# 		viewer.add_geom(geom)

		results = []
		for i in range(len(self.viewers)):
			from multiagent import rendering

			if self.shared_viewer:
				pos = np.zeros(self.world.dim_p)
			else:
				pos = self.agents[i].state.p_pos
			self.viewers[i].set_bounds(pos[0]-self.world.world_size,
										pos[0]+self.world.world_size,
										pos[1]-self.world.world_size,
										pos[1]+self.world.world_size)
			# update geometry positions
			for e, entity in enumerate(self.world.entities):
				self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
				# self.agent_separation_xforms[e].set_translation(*entity.state.p_pos)

				# print("entity.state.theta: ", entity.state.theta, "Degrees: ", math.degrees(entity.state.theta))
				# input("enter..")

				if 'agent' in entity.name:
					# delta_theta = entity.state.theta - entity.initial_theta
					# self.render_geoms_xform[e].set_rotation(entity.state.theta/2 )
					# Get the change in orientation
					delta_theta = entity.state.theta - entity.initial_theta
					# print("theta",entity.state.theta)
					# print("delta_theta",delta_theta)
					# input("Press Enter to continue...")
					self.render_geoms_xform[e].set_rotation(delta_theta)
					self.render_geoms[e].set_color(*entity.color, alpha=0.8)
					# self.agent_separation_geoms[e].set_color(0, 0, 0, alpha=0.1)
					if not entity.silent:
						for ci in range(self.world.dim_c):
							color = 1 - entity.state.c[ci]
							self.comm_geoms[e][ci].set_color(
								color, color, color)
				else:
					self.render_geoms[e].set_color(*entity.color, alpha=0.3)
					if entity.channel is not None:
						for ci in range(self.world.dim_c):
							color = 1 - entity.channel[ci]
							self.comm_geoms[e][ci].set_color(
								color, color, color)

			# render the graph connections
			if hasattr(self.world, 'graph_mode'):
				if self.world.graph_mode:
					edge_list = self.world.edge_list.T
					assert edge_list is not None, ("Edge list should not be None")
					for entity1 in self.world.entities:
						for entity2 in self.world.entities:
							e1_id, e2_id = entity1.global_id, entity2.global_id
							if e1_id == e2_id:
								continue
							# if edge exists draw a line
							if [e1_id, e2_id] in edge_list.tolist():
								src = entity1.state.p_pos
								dest = entity2.state.p_pos
								### commenting out edge line drawings
								# self.viewers[i].draw_line(start=src, end=dest)

			# render to display or array
			results.append(self.viewers[i].render(
						return_rgb_array = mode=='rgb_array'))

		return results

	# create receptor field locations in local coordinate frame
	def _make_receptor_locations(self, agent:Agent) -> List:
		receptor_type = 'polar'
		range_min = 0.05 * 2.0
		range_max = 1.00
		dx = []
		# circular receptive field
		if receptor_type == 'polar':
			for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
				for distance in np.linspace(range_min, range_max, 3):
					dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
			# add origin
			dx.append(np.array([0.0, 0.0]))
		# grid receptive field
		if receptor_type == 'grid':
			for x in np.linspace(-range_max, +range_max, 5):
				for y in np.linspace(-range_max, +range_max, 5):
					dx.append(np.array([x,y]))
		return dx

	@staticmethod
	def is_collision(agent1:Agent, agent2:Agent, dist_min: Optional[float] = None) -> bool:
		delta_pos = agent1.state.p_pos - agent2.state.p_pos
		dist = np.linalg.norm(delta_pos)
		if dist_min is None:
			dist_min = 1.05*(agent1.size + agent2.size)
		return True if dist < dist_min else False

class MultiAgentGraphEnv(MultiAgentBaseEnv):
	metadata = {
		'render.modes' : ['human', 'rgb_array']
	}
	"""
		Parameters:
		–––––––––––
		world: World
			World for the environment. Refer `multiagent/core.py`
		reset_callback: Callable
			Reset function for the environment. Refer `reset()` in 
			`multiagent/navigation_graph.py`
		reward_callback: Callable
			Reward function for the environment. Refer `reward()` in 
			`multiagent/navigation_graph.py`
		observation_callback: Callable
			Observation function for the environment. Refer `observation()` 
			in `multiagent/navigation_graph.py`
		graph_observation_callback: Callable
			Observation function for graph_related stuff in the environment. 
			Refer `graph_observation()` in `multiagent/navigation_graph.py`
		id_callback: Callable
			A function to get the id of the agent in graph
			Refer `get_id()` in `multiagent/navigation_graph.py`
		info_callback: Callable
			Reset function for the environment. Refer `info_callback()` in 
			`multiagent/navigation_graph.py`
		done_callback: Callable
			Reset function for the environment. Refer `done()` in 
			`multiagent/navigation_graph.py`
		update_graph: Callable
			A function to update the graph structure in the environment
			Refer `update_graph()` in `multiagent/navigation_graph.py`
		shared_viewer: bool
			If we want a shared viewer for rendering the environment or 
			individual windows for each agent as the ego
		discrete_action: bool
			If the action space is discrete or not
		scenario_name: str
			Name of the scenario to be loaded. Refer `multiagent/custom_scenarios.py`
	"""
	def __init__(self, world:World, reset_callback:Callable=None, 
					reward_callback:Callable=None,
					observation_callback:Callable=None, 
					graph_observation_callback:Callable=None,
					id_callback:Callable=None,
					info_callback:Callable=None,
					done_callback:Callable=None,
					agent_reached_goal_callback:Callable=None,
					update_graph:Callable=None,
					shared_viewer:bool=True, 
					discrete_action:bool=True,
					scenario_name:str='navigation',
					dynamics_type:str='double_integrator') -> None:
		super(MultiAgentGraphEnv, self).__init__(world, reset_callback, 
											reward_callback,observation_callback, 
											info_callback,done_callback, agent_reached_goal_callback,
											shared_viewer, discrete_action,
											scenario_name, dynamics_type)
		self.update_graph = update_graph
		self.graph_observation_callback = graph_observation_callback
		self.id_callback = id_callback

		# variabls to save episode data
		self.dt = self.world.dt
		self.episode_length = self.world.world_length
		self.coordination_range = self.world.coordination_range
		# This are the values saved in info. (static per each episode)
		# travel metric
		self.prev_episode_travel_time_mean = self.world.world_length
		self.prev_episode_travel_distance_mean = 0.0
		self.prev_episode_done_percentage = 0.0 # percentage between 0 and 1
		self.prev_episode_num_reached_goal_mean = 0.0

		# travel metric
		self.episode_agent_travel_length_list = None
		self.episode_agent_travel_distance_list = None
		self.episode_agent_done_list = None


		self.set_graph_obs_space()

	def set_graph_obs_space(self):
		self.node_observation_space = []
		self.adj_observation_space = []
		self.edge_observation_space = []
		self.agent_id_observation_space = []
		self.share_agent_id_observation_space = []
		num_agents = len(self.agents)
		for agent in self.agents:
			node_obs, adj = self.graph_observation_callback(agent, self.world)
			node_obs_dim = node_obs.shape
			adj_dim = adj.shape
			edge_dim = 1      # NOTE hardcoding edge dimension
			agent_id_dim = 1  # NOTE hardcoding agent id dimension
			self.node_observation_space.append(spaces.Box(low=-np.inf,
														high=+np.inf,
														shape=node_obs_dim,
														dtype=np.float32))
			self.adj_observation_space.append(spaces.Box(low=-np.inf,
														high=+np.inf,
														shape=adj_dim,
														dtype=np.float32))
			self.edge_observation_space.append(spaces.Box(low=-np.inf,
														high=+np.inf,
														shape=(edge_dim,),
														dtype=np.float32))
			self.agent_id_observation_space.append(spaces.Box(low=-np.inf,
														high=+np.inf,
														shape=(agent_id_dim,),
														dtype=np.float32))
			self.share_agent_id_observation_space.append(spaces.Box(low=-np.inf,
														high=+np.inf,
														shape=(num_agents*agent_id_dim,),
														dtype=np.float32))
	

	def step(self, action_n:List) -> Tuple[List, List, List, List, List, List, List]:
		if self.update_graph is not None:
			self.update_graph(self.world)
		self.current_step += 1
		obs_n, reward_n, done_n, info_n = [], [], [], []
		node_obs_n, adj_n, agent_id_n = [], [], []
		cooperate_n, defect_n = [], []
		self.world.current_time_step += 1
		self.agents = self.world.policy_agents
		# set action for each agent
		for i, agent in enumerate(self.agents):
			self._set_action(action_n[i], agent, self.action_space[i])
		# advance world state
		self.world.step()
		# record observation for each agent
		for (i, agent) in enumerate(self.agents):
			obs_n.append(self._get_obs(agent))
			agent_id_n.append(self._get_id(agent))

			reward = self._get_reward(agent)
			reward_n.append(reward)

			node_obs, adj = self._get_graph_obs(agent)
			node_obs_n.append(node_obs)
			
			adj_n.append(adj)

			done_n.append(self._get_done(agent))

			info = {'individual_reward': reward}
			env_info = self._get_info(agent)
			info.update(env_info)   # nothing fancy here, just appending dict to dict
			info_n.append(info)

		# all agents get total reward in cooperative case
		reward = np.sum(reward_n)

		if self.shared_reward:
			reward_n = [[reward]] * self.n  # NOTE this line is similar to PPOEnv
		else:
			reward_n = reward_n

		return obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n


	def reset(self, num_current_episode: Optional[int] = 0) -> Tuple[List, List, List, List]:
		self.current_step = 0
		# reset world
		self.reset_callback(self.world, num_current_episode)
		# reset renderer
		self._reset_render()
		# record observations for each agent
		obs_n, node_obs_n, adj_n, agent_id_n = [], [], [], []
		self.agents = self.world.policy_agents
		for agent in self.agents:
			obs_n.append(self._get_obs(agent))
			agent_id_n.append(self._get_id(agent))
			node_obs, adj = self._get_graph_obs(agent)
			node_obs_n.append(node_obs)
			adj_n.append(adj)
		return obs_n, agent_id_n, node_obs_n, adj_n
	
	def _get_graph_obs(self, agent:Agent):
		if self.graph_observation_callback is None:
			return None, None, None
		return self.graph_observation_callback(agent, self.world)
	
	def _get_id(self, agent:Agent):
		if self.id_callback is None:
			return None
		return self.id_callback(agent)


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
	metadata = {
		'runtime.vectorized': True,
		'render.modes' : ['human', 'rgb_array']
	}

	def __init__(self, env_batch):
		self.env_batch = env_batch

	@property
	def n(self):
		return np.sum([env.n for env in self.env_batch])

	@property
	def action_space(self):
		return self.env_batch[0].action_space

	@property
	def observation_space(self):
		return self.env_batch[0].observation_space

	def step(self, action_n, time):
		obs_n = []
		shared_obs_n = []
		reward_n = []
		done_n = []
		info_n = {'n': []}
		i = 0
		for env in self.env_batch:
			obs, shared_obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
			i += env.n
			obs_n += obs
			shared_obs_n += shared_obs
			# reward = [r / len(self.env_batch) for r in reward]
			reward_n += reward
			done_n += done
		return obs_n, shared_obs_n, reward_n, done_n, info_n

	def reset(self):
		obs_n = []
		shared_obs_n = []
		for env in self.env_batch:
			obs, shared_obs = env.reset()
			obs_n += obs
			shared_obs_n += shared_obs
		return obs_n, shared_obs

	# render environment
	def render(self, mode='human', close=True):
		results_n = []
		for env in self.env_batch:
			results_n += env.render(mode, close)
		return results_n
