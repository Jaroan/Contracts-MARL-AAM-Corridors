import datetime
import time
import numpy as np
from numpy import ndarray as arr
from typing import Tuple
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio
import csv
# import matplotlib.pyplot as plt
def _t2n(x):
	return x.detach().cpu().numpy()

class GMPERunner(Runner):
	"""
		Runner class to perform training, evaluation and data 
		collection for the MPEs. See parent class for details
	"""
	dt = 0.1
	def __init__(self, config):
		super(GMPERunner, self).__init__(config)
		self.num_total_episode = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

	def run(self):
		self.warmup()   

		start = time.time()
		episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
		print("episodes", episodes)

		# This is where the episodes are actually run.
		for episode in range(episodes):
			## increase fairness rewards mid episode
			if self.all_args.increase_fairness:
				if episode ==episodes/2:
					print("2 args check", self.all_args.fair_rew)
					self.all_args.fair_rew = 10
				if episode == episodes/2 +3:
					print("later args check", self.all_args.fair_rew)

			if self.use_linear_lr_decay:
				self.trainer.policy.lr_decay(episode, episodes)
			flag = 0
			# print("episode", episode)
			self.active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1),
									dtype=np.int32)
			# self.active_agents = np.ones((self.n_rollout_threads, self.num_agents, self.episode_length+1),
			# 						dtype=np.int32)


			for step in range(self.episode_length):
				# print("\nstep", step)
				# Sample actions
				if step == 0:
					# active_agents = None
					# active_mask = None
					# finished = None
					values, actions, action_log_probs, rnn_states, \
						rnn_states_critic, actions_env = self.collect(step)
					obs, agent_id, node_obs, adj, rewards, \
						dones, infos = self.envs.step(actions_env, episode)
					finished,finished_list = self.get_finished(dones)
					self.active_masks[dones] = np.zeros((((dones).astype(int)).sum(), 1), dtype=np.float32)

					dones_env = np.all(dones, axis=1)
					self.active_masks[dones_env] = np.ones((((dones_env).astype(int)).sum(), 
											self.num_agents, 1), dtype=np.float32)

					available_actions = np.ones((self.n_rollout_threads, self.num_agents, self.envs.action_space[0].n), dtype=np.float32)

					# # For no-Collab uncomment the line
					rewards = rewards[:,:, np.newaxis]
					data = (obs, agent_id, node_obs, adj, agent_id, rewards, 
							dones, infos, values, actions, action_log_probs, 
							rnn_states, rnn_states_critic, available_actions)

					# insert data into buffer
					self.insert(data)
				else:

					ret = []
					for e in range(self.n_rollout_threads):
						for a in range(self.all_args.num_agents):
							if self.active_masks[e, a,0]:
								ret.append((e, a, self.active_masks[e, a,0]))
					self.active_agents = ret

					values, actions, action_log_probs, rnn_states, \
						rnn_states_critic, actions_env, available_actions = self.collect_with_mask(step,self.active_agents,self.active_masks,finished)
    
					obs, agent_id, node_obs, adj, rewards, \
						dones, infos = self.envs.step(actions_env, episode)

					# Calculate the number of elements to update
					num_elements = np.sum(dones.astype(int))

					# Create an array of zeros with the correct shape
					zeros_array = np.zeros((num_elements, 1), dtype=np.float32)

					# Reshape dones to match the shape of self.active_masks
					reshaped_dones = dones[..., np.newaxis]

					if not reshaped_dones.all():
						self.active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1),
									dtype=np.int32)
					self.active_masks[reshaped_dones] = 0

					finished,finished_list = self.get_finished(dones)

					dones_env = np.all(dones, axis=1)
					self.active_masks[dones_env] = np.ones((((dones_env).astype(int)).sum(), 
											self.num_agents, 1), dtype=np.float32)

					# # For no-Collab uncomment the line
					rewards = rewards[:,:, np.newaxis]
					data = (obs, agent_id, node_obs, adj, agent_id, rewards, 
							dones, infos, values, actions, action_log_probs, 
							rnn_states, rnn_states_critic, available_actions)
					# insert data into buffer
					self.insert(data, active_agents=self.active_agents,active_masks=self.active_masks,finished=finished)

			# compute return and update network
			self.compute()
			train_infos = self.train()
			
			# post process
			total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
			
			# save model
			if (episode % self.save_interval == 0 or episode == episodes - 1):
				self.save()

			# log information
			if episode % self.log_interval == 0:
				end = time.time()
				env_infos = self.process_infos(infos)
				avg_ep_rew = np.mean(self.buffer.rewards) * self.episode_length
				train_infos["average_episode_rewards"] = avg_ep_rew
				print(f"Average episode rewards is {avg_ep_rew:.3f} \t"
					f"Total timesteps: {total_num_steps} \t "
					f"Percentage complete {total_num_steps / self.num_env_steps * 100:.3f}")
				for agent_id in range(self.num_agents):
					str1 = f'agent{agent_id}/distance_mean'
					str2 = f'agent{agent_id}/distance_variance'
					str3 = f'agent{agent_id}/mean_variance'
					str4 = f'agent{agent_id}/dist_to_goal'
					str5 = f'agent{agent_id}/individual_rewards'
					str6 = f'agent{agent_id}/num_agent_collisions'
					print(str1, str2, str3)
					print(f"mean: {env_infos[str1][0]:.3f} \t"
						f"Var: {env_infos[str2][0]:.3f} \t "
						f"Mean/Var: {env_infos[str3][0]:.3f} \t "
						f"Dist2goal: {env_infos[str4][0]:.3f} \t "
						f"Rew: {env_infos[str5][0]:.3f} \t "
						f"Col: {env_infos[str6][0]:.3f} \t ")
				self.log_train(train_infos, total_num_steps)
				self.log_env(env_infos, total_num_steps)


			# eval
			if episode % self.eval_interval == 0 and self.use_eval:
				self.eval(total_num_steps)



	def warmup(self):
		# reset env
		obs, agent_id, node_obs, adj = self.envs.reset()

		# replay buffer
		if self.use_centralized_V:
			# (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
			share_obs = obs.reshape(self.n_rollout_threads, -1)
			# (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
			share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, 
																	axis=1)
			# (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
			share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
			# (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
			share_agent_id = np.expand_dims(share_agent_id, 
											1).repeat(self.num_agents, axis=1)
		else:
			share_obs = obs
			share_agent_id = agent_id

		self.buffer.share_obs[0] = share_obs.copy()
		self.buffer.obs[0] = obs.copy()
		self.buffer.node_obs[0] = node_obs.copy()
		self.buffer.adj[0] = adj.copy()
		self.buffer.agent_id[0] = agent_id.copy()
		self.buffer.share_agent_id[0] = share_agent_id.copy()


	def get_finished(self,dones):
		finished=[]
		f=[]
		bools=[]
		current_env=0
		for env in range(len(dones)) :
			if current_env != env:
				current_env = env
				finished.append(f)
				f=[]
			for agent in range(len(dones[env])):
				if dones[env][agent]==True:
					f.append(agent)
					bools.append(True)
					
				else:
					bools.append(False)
		finished.append(f)
		return finished,bools


	@torch.no_grad()
	def collect_with_mask(self, step:int,active_agents,active_masks,finished) -> Tuple[arr, arr, arr, arr, arr, arr, arr]:
		self.trainer.prep_rollout()
		flag = False
        ### look and see who is active and force the avilable actions to be limited 
		all_actions=[]
		avail_actions_list=[]
		envs_aa=[]
		for env in range(self.n_rollout_threads): #len(active_mask)):
			avail_actions_list=[]
			for a in range(self.all_args.num_agents):
				# if active_mask[env][a]==True:
				
					if a in finished[env]:
						available_actions = np.zeros((self.envs.action_space[0].n))
						available_actions[int(self.envs.action_space[0].n/2)] = 1 ## TODO: Find a better representation for stop action
						flag= True
					else:
						available_actions=np.ones((self.envs.action_space[0].n))
						
					avail_actions_list.append(available_actions)
					
			envs_aa.append(avail_actions_list)
					
		aa= np.asarray(envs_aa)

		value, action, action_log_prob, rnn_states, rnn_states_critic \
			= self.trainer.policy.get_actions(
						np.concatenate(self.buffer.share_obs[step]),
						np.concatenate(self.buffer.obs[step]),
						np.concatenate(self.buffer.node_obs[step]),
						np.concatenate(self.buffer.adj[step]),
						np.concatenate(self.buffer.agent_id[step]),
						np.concatenate(self.buffer.share_agent_id[step]),
						np.concatenate(self.buffer.rnn_states[step]),
						np.concatenate(self.buffer.rnn_states_critic[step]),
						np.concatenate(self.buffer.masks[step]),
						available_actions = np.reshape(aa, (self.all_args.num_agents*self.n_rollout_threads, self.envs.action_space[0].n), order='C'))
		values = np.array(np.split(_t2n(value), self.n_rollout_threads))
		actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
		action_log_probs = np.array(np.split(_t2n(action_log_prob), 
											self.n_rollout_threads))
		rnn_states = np.array(np.split(_t2n(rnn_states), 
								self.n_rollout_threads))
		rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), 
											self.n_rollout_threads))
		# rearrange action
		if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
			for i in range(self.envs.action_space[0].shape):
				uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 
															1)[actions[:, :, i]]
				if i == 0:
					actions_env = uc_actions_env
				else:
					actions_env = np.concatenate((actions_env, 
												uc_actions_env), axis=2)
		elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
			actions_env = np.squeeze(np.eye(
									self.envs.action_space[0].n)[actions], 2)
		else:
			raise NotImplementedError

		if flag:
			avail_actions = aa
		else:
			avail_actions_list=[]

			for env in range(len(active_masks)):
				aa=[]
				for a in range(len(active_masks[env])):
					available_actions=np.ones((self.envs.action_space[0].n))
					aa.append(available_actions)
				avail_actions_list.append(aa)

			
			avail_actions= np.asarray(avail_actions_list)

		return (values, actions, action_log_probs, rnn_states, 
				rnn_states_critic, actions_env, avail_actions)

	@torch.no_grad()
	def collect(self, step:int) -> Tuple[arr, arr, arr, arr, arr, arr]:
		self.trainer.prep_rollout()
		value, action, action_log_prob, rnn_states, rnn_states_critic \
			= self.trainer.policy.get_actions(
							np.concatenate(self.buffer.share_obs[step]),
							np.concatenate(self.buffer.obs[step]),
							np.concatenate(self.buffer.node_obs[step]),
							np.concatenate(self.buffer.adj[step]),
							np.concatenate(self.buffer.agent_id[step]),
							np.concatenate(self.buffer.share_agent_id[step]),
							np.concatenate(self.buffer.rnn_states[step]),
							np.concatenate(self.buffer.rnn_states_critic[step]),
							np.concatenate(self.buffer.masks[step]))
		# [self.envs, agents, dim]
		values = np.array(np.split(_t2n(value), self.n_rollout_threads))
		actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
		action_log_probs = np.array(np.split(_t2n(action_log_prob), 
											self.n_rollout_threads))
		rnn_states = np.array(np.split(_t2n(rnn_states), 
								self.n_rollout_threads))
		rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), 
											self.n_rollout_threads))
		# rearrange action
		if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
			for i in range(self.envs.action_space[0].shape):
				uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 
															1)[actions[:, :, i]]
				if i == 0:
					actions_env = uc_actions_env
				else:
					actions_env = np.concatenate((actions_env, 
												uc_actions_env), axis=2)
		elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
			actions_env = np.squeeze(np.eye(
									self.envs.action_space[0].n)[actions], 2)
		else:
			raise NotImplementedError

		return (values, actions, action_log_probs, rnn_states, 
				rnn_states_critic, actions_env)

	def insert(self, data,active_agents=None,active_masks = None, finished=None):
		obs, agent_id, node_obs, adj, agent_id, rewards, dones, \
			infos, values, actions, action_log_probs, \
			rnn_states, rnn_states_critic, available_actions = data

		dones_env = np.all(dones, axis=1)
		rnn_states[dones] = np.zeros(((dones).sum(),
												self.recurrent_N, 
												self.hidden_size), 
												dtype=np.float32)
		rnn_states_critic[dones] = np.zeros(((dones).sum(),
										*self.buffer.rnn_states_critic.shape[3:]), 
										dtype=np.float32)
		masks = np.ones((self.n_rollout_threads, 
						self.num_agents, 1), 
						dtype=np.float32)

		masks[dones] = np.zeros(((dones).sum(), 1), 
										dtype=np.float32)
		active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1),
							   dtype=np.float32)
		active_masks[dones] = np.zeros((((dones).astype(int)).sum(), 1), dtype=np.float32)
		active_masks[dones_env] = np.ones((((dones_env).astype(int)).sum(), 
								self.num_agents, 1), dtype=np.float32)


		# if centralized critic, then shared_obs is concatenation of obs from all agents
		if self.use_centralized_V:
			# TODO stack agent_id as well for agent specific information
			# (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
			share_obs = obs.reshape(self.n_rollout_threads, -1)
			# (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
			share_obs = np.expand_dims(share_obs, 
										1).repeat(self.num_agents, axis=1)
			# (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
			share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
			# (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
			share_agent_id = np.expand_dims(share_agent_id, 
											1).repeat(self.num_agents, axis=1)
		else:
			share_obs = obs
			share_agent_id = agent_id
		self.buffer.insert(share_obs, obs, node_obs, adj, agent_id, share_agent_id, 
						rnn_states, rnn_states_critic, actions, action_log_probs, 
						values, rewards, masks,active_masks=active_masks,available_actions=available_actions)

	@torch.no_grad()
	def compute(self):
		"""Calculate returns for the collected data."""
		self.trainer.prep_rollout()
		next_values = self.trainer.policy.get_values(
							np.concatenate(self.buffer.share_obs[-1]),
							np.concatenate(self.buffer.node_obs[-1]),
							np.concatenate(self.buffer.adj[-1]),
							np.concatenate(self.buffer.share_agent_id[-1]),
							np.concatenate(self.buffer.rnn_states_critic[-1]),
							np.concatenate(self.buffer.masks[-1]))
		next_values = np.array(np.split(_t2n(next_values), 
								self.n_rollout_threads))
		self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

	@torch.no_grad()
	def eval(self, total_num_steps:int):
		eval_episode_rewards = []
		eval_obs, eval_agent_id, eval_node_obs, eval_adj = self.eval_envs.reset()

		eval_rnn_states = np.zeros((self.n_eval_rollout_threads, 
									*self.buffer.rnn_states.shape[2:]), 
									dtype=np.float32)
		eval_masks = np.ones((self.n_eval_rollout_threads, 
								self.num_agents, 1), 
								dtype=np.float32)

		for eval_step in range(self.episode_length):
			self.trainer.prep_rollout()
			eval_action, eval_rnn_states = self.trainer.policy.act(
												np.concatenate(eval_obs),
												np.concatenate(eval_node_obs),
												np.concatenate(eval_adj),
												np.concatenate(eval_agent_id),
												np.concatenate(eval_rnn_states),
												np.concatenate(eval_masks),
												deterministic=True)
			eval_actions = np.array(np.split(_t2n(eval_action), 
											self.n_eval_rollout_threads))
			eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), 
											self.n_eval_rollout_threads))
			
			if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
				for i in range(self.eval_envs.action_space[0].shape):
					eval_uc_actions_env = np.eye(
								self.eval_envs.action_space[0].high[i] + 
														1)[eval_actions[:, :, i]]
					if i == 0:
						eval_actions_env = eval_uc_actions_env
					else:
						eval_actions_env = np.concatenate((eval_actions_env, 
															eval_uc_actions_env), 
															axis=2)
			elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
				eval_actions_env = np.squeeze(np.eye(
							self.eval_envs.action_space[0].n)[eval_actions], 2)
			else:
				raise NotImplementedError

			# Obser reward and next obs
			eval_obs, eval_agent_id, eval_node_obs, eval_adj, eval_rewards, \
				eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
			eval_episode_rewards.append(eval_rewards)
			eval_dones_env = np.all(eval_dones, axis=1)

			eval_rnn_states[eval_dones_env] = np.zeros((
													(eval_dones_env == True).sum(), 
													self.recurrent_N, 
													self.hidden_size), 
													dtype=np.float32)
			eval_masks = np.ones((self.n_eval_rollout_threads, 
								self.num_agents, 1), 
								dtype=np.float32)
			eval_masks[eval_dones_env] = np.zeros((
												(eval_dones_env == True).sum(), 1), 
												dtype=np.float32)

		eval_episode_rewards = np.array(eval_episode_rewards)
		eval_env_infos = {}
		eval_env_infos['eval_average_episode_rewards'] = np.sum(
												np.array(eval_episode_rewards), 
												axis=0)
		eval_average_episode_rewards = np.mean(
									eval_env_infos['eval_average_episode_rewards'])
		print("eval average episode rewards of agent: " + 
											str(eval_average_episode_rewards))
		self.log_env(eval_env_infos, total_num_steps)

	def save_images(self, img_list, filename):

		for i, img in enumerate(img_list):
			imageio.imwrite(filename + str(i) + '.png', img)
		# overlay all at alpha=0.5 for the white background
		# img = np.sum(np.array(img_list), axis=0)

	@torch.no_grad()
	def render(self, get_metrics:bool=False):
		"""
			Visualize the env.
			get_metrics: bool (default=False)
				if True, just return the metrics of the env and don't render.
		"""
		envs = self.envs
		self.reset_number = 0
		all_frames = []
		total_dists_traveled, total_time_taken = [], []
		rewards_arr, success_rates_arr, num_collisions_arr, frac_episode_arr, fairness_param = [], [], [], [],[]
		dist_mean_arr, time_mean_arr = [],[]
		stddev_param = []
		dists_trav_list = np.zeros((self.num_agents))
		time_taken_list = np.zeros((self.num_agents))
		formation_success = []
		time_fairness, time_stddev_param, time_mean = [], [], []
		conformance_percentage_list = []
		delta_space_list = []
		spacing_violations_list = []
		print("num_episodes: ", self.all_args.render_episodes)
		# num_total_episode = int(self.all_args.num_env_steps) // self.all_args.episode_length // self.all_args.n_rollout_threads

		obs, agent_id, node_obs, adj = envs.reset(self.all_args.render_episodes)
		with open(str(self.all_args.model_dir)+'/'+str(self.all_args.model_name)+'_'+str(self.all_args.formation_type)+'_results_collect_oct.csv', 'a', newline="") as f_traj:
			# create the csv writer
			writer = csv.writer(f_traj)


		for episode in range(self.all_args.render_episodes):

			if not get_metrics:
				if self.all_args.save_gifs:
					image = envs.render('rgb_array')[0][0]
					all_frames.append(image)
				else:
					envs.render('human')

			rnn_states = np.zeros((self.n_rollout_threads, 
									self.num_agents, 
									self.recurrent_N, 
									self.hidden_size), 
									dtype=np.float32)
			masks = np.ones((self.n_rollout_threads, 
							self.num_agents, 1), 
							dtype=np.float32)
			available_actions = np.ones((self.num_agents, self.envs.action_space[0].n), 
										dtype=np.float32)
			episode_rewards = []
			
			for step in range(self.episode_length):
				# print("\nstep",step)
				calc_start = time.time()

				zero_masks = masks[0] == 0
				if 	not zero_masks.all():
					available_actions = np.ones((self.num_agents, self.envs.action_space[0].n), 
										dtype=np.float32)
				# Broadcast the boolean mask to match the shape of available_actions
				broadcasted_zero_masks = np.broadcast_to(zero_masks, available_actions.shape)
				# TODO: This is a hack to make only the stop action available when the agent is done
				stop_mask = np.zeros(self.envs.action_space[0].n)
				stop_mask[int(self.envs.action_space[0].n/2)] = 1
				available_actions[zero_masks[:,0]] = stop_mask
				self.trainer.prep_rollout()
				action, rnn_states = self.trainer.policy.act(
													np.concatenate(obs),
													np.concatenate(node_obs),
													np.concatenate(adj),
													np.concatenate(agent_id),
													np.concatenate(rnn_states),
													np.concatenate(masks),
													available_actions = available_actions,
													deterministic=True)
				actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
				rnn_states = np.array(np.split(_t2n(rnn_states), 
									self.n_rollout_threads))

				if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
					for i in range(envs.action_space[0].shape):
						uc_actions_env = np.eye(
								envs.action_space[0].high[i]+1)[actions[:, :, i]]
						if i == 0:
							actions_env = uc_actions_env
						else:
							actions_env = np.concatenate((actions_env, 
														uc_actions_env), 
														axis=2)
				elif envs.action_space[0].__class__.__name__ == 'Discrete':
					actions_env = np.squeeze(np.eye(
											envs.action_space[0].n)[actions], 2)
				else:
					raise NotImplementedError

				# Obser reward and next obs
				obs, agent_id, node_obs, adj, \
					rewards,dones, infos, reset_count = envs.step(actions_env)

				episode_rewards.append(rewards)


				dones_env = np.all(dones)
				rnn_states[dones == True] = np.zeros(((dones == True).sum(), 
													self.recurrent_N, 
													self.hidden_size), 
													dtype=np.float32)
				masks = np.ones((self.n_rollout_threads, 
								self.num_agents, 1), 
								dtype=np.float32)
				
				masks[dones == True] = np.zeros(((dones == True).sum(), 1), 
												dtype=np.float32)
				dones_env = np.all(dones, axis=1)
				masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
				if not get_metrics:
					if self.all_args.save_gifs:
						image = envs.render('rgb_array')[0][0]
						all_frames.append(image)
						calc_end = time.time()
						elapsed = calc_end - calc_start
						if elapsed < self.all_args.ifi:
							time.sleep(self.all_args.ifi - elapsed)
					else:
						envs.render('human')


				self.reset_number += reset_count
				if reset_count > 0:
					break
				if self.reset_number == self.all_args.render_episodes :
					break

				# input("Step: {}, Press Enter to continue...".format(step))
				# print("\n\n")
			env_infos = self.process_infos(infos)
			# print('_'*2)
			num_collisions = self.get_collisions(env_infos)
			frac, success,time_taken = self.get_fraction_episodes(env_infos)
			if np.any(frac==1):
				frac_max = 1.0
			else:
				frac_max = np.max(frac)
			rewards_arr.append(np.mean(np.sum(np.array(episode_rewards), axis=0)))
			frac_episode_arr.append(frac_max)
			success_rates_arr.append(success)
			num_collisions_arr.append(num_collisions)
			fairness_metric = self.get_fairness_metric(env_infos)
			stddev_metric = self.get_dist_std(env_infos)

			fairness_param.append(fairness_metric[-1])


			stddev_param.append(1.0/(stddev_metric[-1]+0.0001))

			dist_mean = self.get_dist_mean(env_infos)
			dist_mean_arr.append(dist_mean[-1])
			time_mean = self.get_time_mean(env_infos)
			time_mean_arr.append(time_mean[-1])
	
			dists_traveled = self.get_dists_traveled(env_infos)
			dists_trav_list +=dists_traveled

			time_taken_list +=time_taken

			total_dists_traveled.append(np.sum(dists_traveled))
			total_time_taken.append(np.sum(time_taken))

			time_fairness_metric = self.get_time_fairness(env_infos)
			time_stddev_metric = self.get_time_std(env_infos)
			time_fairness.append(time_fairness_metric[-1])
			time_stddev_param.append(1.0/(time_stddev_metric[-1]+0.0001))


			conformance_percentage = self.get_conformation_percentages(env_infos)
			# print("conformance_percentage", conformance_percentage)
			conformance_percentage_list.append(np.mean(conformance_percentage))
			# print("conformance_percentage_list", conformance_percentage_list)

			delta_space = self.get_delta_spacing(env_infos)
			# print("delta_space", delta_space)
			delta_space_list.append(np.mean(delta_space))

			spacing_violations = self.get_spacing_violations(env_infos)
			spacing_violations_list.append(np.mean(spacing_violations))
			# write a row to the csv file
			csv_data1 = [self.num_obstacles, 
						self.num_agents,
						self.all_args.world_size,
						self.episode_length,
						self.all_args.render_episodes,
						self.reset_number, 
						step, 
						fairness_metric[-1],
						np.mean(success),
						frac_max,
						total_dists_traveled[-1],
						total_time_taken[-1],
					]
			# with open(str(self.all_args.model_dir)+'/'+str(self.all_args.model_name)+'_'+str(self.all_args.formation_type)+'_fairness.csv', 'a', newline="") as f1:
			# 	# create the csv writer
			# 	writer = csv.writer(f1)
			# 	writer.writerow(csv_data1)
			# RESET HERE
			obs, agent_id, node_obs, adj = envs.reset(self.all_args.render_episodes)
		# Calculate the statistics for the box whisker plot
		fair_minimum = np.min(fairness_param)
		fair_0_1_quantile = np.percentile(fairness_param, 10)
		fair_median = np.median(fairness_param)
		fair_0_9_quantile = np.percentile(fairness_param, 90)
		fair_maximum = np.max(fairness_param)
		fair_mean = np.mean(fairness_param)

		## calculate the statistics for the box whisker plot for stddev_param
		stddev_minimum = np.min(stddev_param)
		stddev_0_1_quantile = np.percentile(stddev_param, 10)
		stddev_median = np.median(stddev_param)
		stddev_0_9_quantile = np.percentile(stddev_param, 90)
		stddev_maximum = np.max(stddev_param)
		stddev_mean = np.mean(stddev_param)


		time_fair_minimum = np.min(time_fairness)
		time_fair_0_1_quantile = np.percentile(time_fairness, 10)
		time_fair_median = np.median(time_fairness)
		time_fair_0_9_quantile = np.percentile(time_fairness, 90)
		time_fair_maximum = np.max(time_fairness)
		time_fair_mean = np.mean(time_fairness)

		## calculate the statistics for the box whisker plot for time_stddev_param
		time_stddev_minimum = np.min(time_stddev_param)
		time_stddev_0_1_quantile = np.percentile(time_stddev_param, 10)
		time_stddev_median = np.median(time_stddev_param)
		time_stddev_0_9_quantile = np.percentile(time_stddev_param, 90)
		time_stddev_maximum = np.max(time_stddev_param)
		time_stddev_mean = np.mean(time_stddev_param)


		# calculate statistics for dist_mean_arr
		dist_mean_minimum = np.min(dist_mean_arr)
		dist_mean_0_1_quantile = np.percentile(dist_mean_arr, 10)
		dist_mean_median = np.median(dist_mean_arr)
		dist_mean_0_9_quantile = np.percentile(dist_mean_arr, 90)
		dist_mean_maximum = np.max(dist_mean_arr)
		dist_mean_mean = np.mean(dist_mean_arr)


		# calculate statistics for time_mean_arr
		time_mean_minimum = np.min(time_mean_arr)
		time_mean_0_1_quantile = np.percentile(time_mean_arr, 10)
		time_mean_median = np.median(time_mean_arr)
		time_mean_0_9_quantile = np.percentile(time_mean_arr, 90)
		time_mean_maximum = np.max(time_mean_arr)
		time_mean_mean = np.mean(time_mean_arr)

		# Convert boolean array to integers
		# success_rates_arr = success_rates_arr.astype(int)
		# success_rates_arr = [int(value) for value in success_rates_arr]
		# calculate statistics for success rates
		success_rates_minimum = np.min(success_rates_arr)
		success_rates_0_1_quantile = np.percentile(success_rates_arr, 10)
		success_rates_median = np.median(success_rates_arr)
		success_rates_0_9_quantile = np.percentile(success_rates_arr, 90)
		success_rates_maximum = np.max(success_rates_arr)
		success_rates_mean = np.mean(success_rates_arr)

		total_dists_traveled_median = np.median(total_dists_traveled)
		total_dists_traveled_mean = np.mean(total_dists_traveled)
		total_dists_traveled_0_1_quantile = np.percentile(total_dists_traveled, 10)
		total_dists_traveled_0_9_quantile = np.percentile(total_dists_traveled, 90)
		total_dists_traveled_min = np.min(total_dists_traveled)
		total_dists_traveled_max = np.max(total_dists_traveled)

		total_time_taken_median = np.median(total_time_taken)
		total_time_taken_mean = np.mean(total_time_taken)
		total_time_taken_0_1_quantile = np.percentile(total_time_taken, 10)
		total_time_taken_0_9_quantile = np.percentile(total_time_taken, 90)
		total_time_taken_min = np.min(total_time_taken)
		total_time_taken_max = np.max(total_time_taken)		

		conformance_percentage_minimum = np.min(conformance_percentage_list)
		conformance_percentage_0_1_quantile = np.percentile(conformance_percentage_list, 10)
		conformance_percentage_median = np.median(conformance_percentage_list)
		conformance_percentage_0_9_quantile = np.percentile(conformance_percentage_list, 90)
		conformance_percentage_maximum = np.max(conformance_percentage_list)
		conformance_percentage_mean = np.mean(conformance_percentage_list)
		conformance_percentage_std = np.std(conformance_percentage_list)
		
		delta_space_minimum = np.min(delta_space_list)
		delta_space_0_1_quantile = np.percentile(delta_space_list, 10)
		delta_space_median = np.median(delta_space_list)
		delta_space_0_9_quantile = np.percentile(delta_space_list, 90)
		delta_space_maximum = np.max(delta_space_list)
		delta_space_mean = np.mean(delta_space_list)
		delta_space_std = np.std(delta_space_list)


		spacing_violations_minimum = np.min(spacing_violations_list)
		spacing_violations_0_1_quantile = np.percentile(spacing_violations_list, 10)
		spacing_violations_median = np.median(spacing_violations_list)
		spacing_violations_0_9_quantile = np.percentile(spacing_violations_list, 90)
		spacing_violations_maximum = np.max(spacing_violations_list)
		spacing_violations_mean = np.mean(spacing_violations_list)
		spacing_violations_std = np.std(spacing_violations_list)

		np.set_printoptions(linewidth=400)
		print("Rewards", np.mean(rewards_arr))
		print("Frac of episode", np.mean(frac_episode_arr))

		# report the success rates statistics
		print("Success rates mean", success_rates_mean)
		print("Success rates median",success_rates_median)
		print("Success rates minimum", success_rates_minimum)
		print("Success rates 0.1 quantile", success_rates_0_1_quantile)
		print("Success rates 0.9 quantile", success_rates_0_9_quantile)
		print("Success rates maximum", success_rates_maximum)

		print("Num collisions", np.mean(num_collisions_arr))
		print("Fairness Median", fair_median)
		print("Fairness Mean", fair_mean)
		# Print the values
		print("Fair Minimum:", fair_minimum)
		print("Fair 0_1 Quantile:", fair_0_1_quantile)
		print("Fair Median:", fair_median)
		print("Fair 0.9 Quantile:", fair_0_9_quantile)
		print("Fair Maximum:", fair_maximum)

		print("Stddev Minimum:", stddev_minimum)
		print("Stddev 0_1 Quantile:", stddev_0_1_quantile)
		print("Stddev Median:", stddev_median)
		print("Stddev 0.9 Quantile:", stddev_0_9_quantile)
		print("Stddev Maximum:", stddev_maximum)
		print("Stddev Mean:", stddev_mean)
		print("Dists traveled", dists_trav_list)
		print("Time taken", time_taken_list)


		# print time fairness and time stddev
		print("Time Fair Minimum:", time_fair_minimum)
		print("Time Fair 0.1 Quantile:", time_fair_0_1_quantile)
		print("Time Fair Median:", time_fair_median)
		print("Time Fair 0.9 Quantile:", time_fair_0_9_quantile)
		print("Time Fair Maximum:", time_fair_maximum)
		print("Time Fair Mean:", time_fair_mean)

		print("Time Stddev Minimum:", time_stddev_minimum)
		print("Time Stddev 0.1 Quartile:", time_stddev_0_1_quantile)
		print("Time Stddev Median:", time_stddev_median)
		print("Time Stddev 0.9 Quartile:", time_stddev_0_9_quantile)
		print("Time Stddev Maximum:", time_stddev_maximum)
		print("Time Stddev Mean:", time_stddev_mean)

		# print dist mean and time mean
		print("Dist Mean Minimum:", dist_mean_minimum)
		print("Dist Mean 0.1 Quartile:", dist_mean_0_1_quantile)
		print("Dist Mean Median:", dist_mean_median)
		print("Dist Mean 0.9 Quartile:", dist_mean_0_9_quantile)
		print("Dist Mean Maximum:", dist_mean_maximum)
		print("Dist Mean Mean:", dist_mean_mean)

		print("Time Mean Minimum:", time_mean_minimum)
		print("Time Mean 0.1 Quartile:", time_mean_0_1_quantile)
		print("Time Mean Median:", time_mean_median)
		print("Time Mean 0.9 Quartile:", time_mean_0_9_quantile)
		print("Time Mean Maximum:", time_mean_maximum)
		print("Time Mean Mean:", time_mean_mean)

		print("Total Dists Traveled Median:", total_dists_traveled_median)
		print("Total Time Taken Median:", total_time_taken_median)

		print("Conformance_percentage Mean:", conformance_percentage_mean)
		print("Conformance_percentage median:", conformance_percentage_median)

		print("Delta_space Mean:", delta_space_mean)
		print("Delta_space median:", delta_space_median)


		print("Spacing violations Mean:", spacing_violations_mean)
		print("Spacing violations median:", spacing_violations_median)

		rewards_mean = np.mean(rewards_arr)

		# csv_data = [
		# 	self.num_obstacles, 
		# 	self.num_agents,
		# 	self.all_args.world_size,
		# 	self.episode_length,
		# 	self.all_args.render_episodes,
		# 	fair_mean,  # Add mean value for fairness_param
		# 	fair_minimum,  # Add minimum value for fairness_param
		# 	fair_0_1_quantile,  # Add first quartile value for fairness_param
		# 	fair_median,  # Add median value for fairness_param
		# 	fair_0_9_quantile,  # Add third quartile value for fairness_param
		# 	fair_maximum,  # Add maximum value for fairness_param
		# 	stddev_mean,  # Add mean value for stddev_param
		# 	stddev_minimum,  # Add minimum value for stddev_param
		# 	stddev_0_1_quantile,  # Add first quartile value for stddev_param
		# 	stddev_median,  # Add median value for stddev_param
		# 	stddev_0_9_quantile,  # Add third quartile value for stddev_param
		# 	stddev_maximum,  # Add maximum value for stddev_param
		# 	np.mean(frac_episode_arr),
		# 	success_rates_minimum,
		# 	success_rates_0_1_quantile,
		# 	success_rates_median ,
		# 	success_rates_0_9_quantile ,
		# 	success_rates_maximum,
		# 	success_rates_mean,
		# 	np.mean(num_collisions_arr),
		# 	rewards_mean,
		# 	rewards_mean / self.num_agents,
		# 	rewards_mean / (self.num_agents * self.episode_length),
		# 	dists_trav_list,
		# 	time_taken_list,
		# 	time_fair_mean,  # Add mean value for time_fairness
		# 	time_fair_minimum,  # Add minimum value for time_fairness
		# 	time_fair_0_1_quantile,  # Add first quartile value for time_fairness
		# 	time_fair_median,  # Add median value for time_fairness
		# 	time_fair_0_9_quantile,  # Add third quartile value for time_fairness
		# 	time_fair_maximum,  # Add maximum value for time_fairness
		# 	time_stddev_mean,  # Add mean value for time_stddev_param
		# 	time_stddev_minimum,  # Add minimum value for time_stddev_param
		# 	time_stddev_0_1_quantile,  # Add first quartile value for time_stddev_param
		# 	time_stddev_median,  # Add median value for time_stddev_param
		# 	time_stddev_0_9_quantile,  # Add third quartile value for time_stddev_param
		# 	time_stddev_maximum,  # Add maximum value for time_stddev_param
		# 	dist_mean_mean,  # Add mean value for dist_mean_arr
		# 	dist_mean_minimum,  # Add minimum value for dist_mean_arr
		# 	dist_mean_0_1_quantile,  # Add first quartile value for dist_mean_arr
		# 	dist_mean_median,  # Add median value for dist_mean_arr
		# 	dist_mean_0_9_quantile,  # Add third quartile value for dist_mean_arr
		# 	dist_mean_maximum,  # Add maximum value for dist_mean_arr
		# 	time_mean_mean,  # Add mean value for time_mean_arr
		# 	time_mean_minimum,  # Add minimum value for time_mean_arr
		# 	time_mean_0_1_quantile,  # Add first quartile value for time_mean_arr
		# 	time_mean_median,  # Add median value for time_mean_arr
		# 	time_mean_0_9_quantile,  # Add third quartile value for time_mean_arr
		# 	time_mean_maximum,  # Add maximum value for time_mean_arr
		# 	total_dists_traveled_median,  # Add median value for total_dists_traveled
		# 	total_dists_traveled_mean,
		# 	total_dists_traveled_0_1_quantile,
		# 	total_dists_traveled_0_9_quantile,
		# 	total_dists_traveled_min,
		# 	total_dists_traveled_max,
		# 	total_time_taken_median,  # Add median value for total_time_taken
		# 	total_time_taken_mean,
		# 	total_time_taken_0_1_quantile,
		# 	total_time_taken_0_9_quantile,
		# 	total_time_taken_min,
		# 	total_time_taken_max			
		# ]

		csv_data = [
			self.num_obstacles, 
			self.num_agents,
			self.all_args.world_size,
			self.episode_length,
			self.all_args.render_episodes,
			np.mean(frac_episode_arr),
			success_rates_minimum,
			success_rates_0_1_quantile,
			success_rates_median ,
			success_rates_0_9_quantile ,
			success_rates_maximum,
			success_rates_mean,
			np.mean(num_collisions_arr),
			rewards_mean,
			rewards_mean / self.num_agents,
			rewards_mean / (self.num_agents * self.episode_length),
			dists_trav_list,
			time_taken_list,
			conformance_percentage_mean,
			conformance_percentage_median,
			conformance_percentage_std,
			delta_space_mean,
			delta_space_median,
			delta_space_std,
			time_fair_mean,  # Add mean value for time_fairness
			time_fair_minimum,  # Add minimum value for time_fairness
			time_fair_0_1_quantile,  # Add first quartile value for time_fairness
			time_fair_median,  # Add median value for time_fairness
			time_fair_0_9_quantile,  # Add third quartile value for time_fairness
			time_fair_maximum,  # Add maximum value for time_fairness
			time_stddev_mean,  # Add mean value for time_stddev_param
			time_stddev_minimum,  # Add minimum value for time_stddev_param
			time_stddev_0_1_quantile,  # Add first quartile value for time_stddev_param
			time_stddev_median,  # Add median value for time_stddev_param
			time_stddev_0_9_quantile,  # Add third quartile value for time_stddev_param
			time_stddev_maximum,  # Add maximum value for time_stddev_param
			dist_mean_mean,  # Add mean value for dist_mean_arr
			dist_mean_minimum,  # Add minimum value for dist_mean_arr
			dist_mean_0_1_quantile,  # Add first quartile value for dist_mean_arr
			dist_mean_median,  # Add median value for dist_mean_arr
			dist_mean_0_9_quantile,  # Add third quartile value for dist_mean_arr
			dist_mean_maximum,  # Add maximum value for dist_mean_arr
			time_mean_mean,  # Add mean value for time_mean_arr
			time_mean_minimum,  # Add minimum value for time_mean_arr
			time_mean_0_1_quantile,  # Add first quartile value for time_mean_arr
			time_mean_median,  # Add median value for time_mean_arr
			time_mean_0_9_quantile,  # Add third quartile value for time_mean_arr
			time_mean_maximum,  # Add maximum value for time_mean_arr
			total_dists_traveled_median,  # Add median value for total_dists_traveled
			total_dists_traveled_mean,
			total_dists_traveled_0_1_quantile,
			total_dists_traveled_0_9_quantile,
			total_dists_traveled_min,
			total_dists_traveled_max,
			total_time_taken_median,  # Add median value for total_time_taken
			total_time_taken_mean,
			total_time_taken_0_1_quantile,
			total_time_taken_0_9_quantile,
			total_time_taken_min,
			total_time_taken_max,
			fair_mean,  # Add mean value for fairness_param
			fair_minimum,  # Add minimum value for fairness_param
			fair_0_1_quantile,  # Add first quartile value for fairness_param
			fair_median,  # Add median value for fairness_param
			fair_0_9_quantile,  # Add third quartile value for fairness_param
			fair_maximum,  # Add maximum value for fairness_param
			spacing_violations_mean,
			spacing_violations_median,
			spacing_violations_std
		]

		# open the file in the write mode
		with open(str(self.all_args.model_dir)+'/'+str(self.all_args.model_name)+'_'+str(self.all_args.formation_type)+'_results_collect_final_split_oct.csv', 'a', newline="") as f:
			# create the csv writer
			writer = csv.writer(f)

			# write a row to the csv file
			writer.writerow(csv_data)

		
		if not get_metrics:
			if self.all_args.save_gifs:
				imageio.mimsave(str(self.gif_dir) + '/'+str(self.all_args.model_name)+'random_'+str(self.all_args.num_agents)+'.gif', 
								all_frames, duration=self.all_args.ifi, loop=0)
