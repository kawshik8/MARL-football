import numpy as np
import torch
from torch import optim
from utils import *
from datetime import datetime
from models import *
import os
import copy
from tensorboardX import SummaryWriter
from collections import deque

class MADDPG:
    def __init__(self, args, envs, episodes_before_train=100):

        self.args = args
        self.env = envs

        
        
        self.global_state_actor = deepmind(args.history, nchannels=4)#Global_State_Net(args.history, nagents=args.n_agents)
        self.global_state_critic = deepmind(args.history, nchannels=3)#Global_State_Net(args.history, nagents=args.n_agents)

        if args.tie_actor_wts:
            self.actor = Actor(args.history, 19, global_state_net = self.global_state_actor)
        if args.tie_critic_wts:
            self.critic = Critic(args.history, args.action_dim, args.n_agents, nactions = 19, global_state_net = self.global_state_critic)

        self.actors = [Actor(args.history, 19) if not args.tie_actor_wts else self.actor for i in range(args.n_agents)]
        self.critics = [Critic(args.history, args.action_dim, args.n_agents, nactions = 19) if not args.tie_critic_wts else self.critic for i in range(args.n_agents)]

        # print(self.actor)
        # print(self.critic)

        self.actors_target = copy.deepcopy(self.actors)
        self.critics_target = copy.deepcopy(self.critics)


        self.n_agents = args.n_agents
        self.n_states = 115
        self.n_actions = 19
        self.memory = ReplayMemory(capacity = 1e7)
        self.batch_size = args.batch_size
        self.use_cuda = args.cuda
        self.episodes_before_train = episodes_before_train

        self.GAMMA = args.gamma # 0.95
        self.tau = args.polyak_tau # 0.01

        self.var = [1.0 for i in range(self.n_agents)]
        self.critics_optimizer = [torch.optim.Adam(x.parameters(),lr=0.001) for x in self.critics]
        self.actors_optimizer = [torch.optim.Adam(x.parameters(),lr=0.001) for x in self.actors]

        if self.use_cuda:
            
            for x in self.actors:
                x.cuda()
            self.critics.cuda()

            for x in self.actors_target:
                x.cuda()
            self.critics_target.cuda()

        self.steps_done = 0
        self.episode_done = 0

        self.n_episodes = 20000
        self.max_steps = 10000

        self.scale_reward = 0.01

        self.log_path = self.args.log_dir + self.args.env_name + '.log'
        self.logger = config_logger(self.log_path)

        no = 1
        while os.path.exists(self.args.log_dir + self.args.env_name + "_" + str(no)): 
            no +=1
        
        self.tbx = SummaryWriter(self.args.log_dir + self.args.env_name + "_" + str(no))

        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.num_updates = 0
        self.total_updates = self.args.total_frames // (self.args.batch_size * self.args.num_workers)

        # self.obs = np.zeros((self.args.num_workers, ) + self.envs.observation_space.shape, dtype=self.envs.observation_space.dtype.name)
        # self.obs[:] = self.envs.reset()
    
    # convert the numpy array to tensors
    def _get_tensors(self, obs):
        obs_tensor = torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32)
        # obs_tensor = torch.tensor(obs, dtype=torch.float32)
        # decide if put the tensor on the GPU
        if self.args.cuda:
            obs_tensor = obs_tensor.cuda()
        return obs_tensor

    def learn(self):

        num_updates = self.args.total_frames // (self.args.nsteps * self.args.num_workers)
        # get the reward to calculate other informations
        episode_rewards = torch.zeros([self.args.num_workers, 1])
        final_rewards = torch.zeros([self.args.num_workers, 1])

        # reward_record = []
        self.total_reward = deque([], capacity=100)

        # for update in range(num_updates):
        for i_episode in range(self.n_episodes):

            obs = self.env.reset()
            # print("raw obs: ",obs.shape)
            obs = self._get_tensors(obs)
            # print("get tensors: ", obs.shape)
            
            episode_reward = 0
            
            # rr = np.zeros((n_agents,))
            # for step in range(self.args.nsteps):
            for t in range(self.max_steps):

                
                # print(" obs shape: " ,obs.shape)
                # for i in range(obs.shape[1]):
                #     if obs[0][i] != obs[1][i]:
                #         print(i,obs[0][i],obs[1][i])

                actions = [self.actors[i](obs[i].unsqueeze(0)) for i in range(self.n_agents)]
                # print(actions)
                actions = [torch.tensor(select_actions(action), dtype=torch.long) for action in actions]
                # print(actions)
                actions = torch.stack(actions)
                # print(actions.shape)

                # actions = torch.max(actions, dim=-1)[1].detach().cpu().squeeze()

                obs, rewards, dones, _ = self.env.step(actions.numpy())
                # print(obs.shape)

                obs = self._get_tensors(obs)
                if t != self.max_steps - 1:
                    next_obs = obs
                else:
                    next_obs = None
                # print("next obs:",next_obs.shape)

                # print(rewards.shape)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                # print(rewards.shape)
                # self.total_reward += rewards
                episode_reward += rewards

                # print(obs.shape)
                # print(obs.reshape(self.n_agents, 72, 96, self.args.history, 4)[:,:,:,:3].view(self.n_agents, 72, 96, -1).shape)
                self.memory.push(obs.data, actions, next_obs, rewards)
                #obs.reshape(self.n_agents, 72, 96, self.args.history, 4)[:,:,:,:3].view(self.n_agents, 72, 96, -1)
                obs = next_obs
                

                c_loss, a_loss = self.update_policy()

                if dones:
                    break

            self.episode_done += 1
            
            print('Episode: ', i_episode, " reward: ", episode_reward, " position: ", self.memory.position)

            self.total_reward.append(episode_reward)
            # print(self.total_reward, torch.stack(self.total_reward))
            
            for i in range(self.args.n_agents):
                self.tbx.add_scalar('train/' + "total_reward_agent" + str(i), episode_reward[i], i_episode)

            self.tbx.add_scalar('train/' + "total_reward", episode_reward.sum(), i_episode)
            
            # print(total_reward)
            
            # reward_record.append(total_reward)

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = torch.cuda.ByteTensor if self.use_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = torch.stack(batch.states).type(FloatTensor)
            action_batch = torch.stack(batch.actions).type(FloatTensor)
            reward_batch = torch.stack(batch.rewards).type(FloatTensor)
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = torch.stack(
                [s for s in batch.next_states
                 if s is not None]).type(FloatTensor)

            # for current agent
            # print(state_batch.shape, action_batch.shape, reward_batch.shape, state_batch.reshape(self.batch_size, self.n_agents, self.args.history, 4, 72, 96)[:,agent,:,:3].reshape(self.batch_size, -1, 72, 96).shape)
            whole_state = state_batch.reshape(self.batch_size, self.n_agents, self.args.history, 4, 72, 96)[:,agent,:,:3].reshape(self.batch_size, -1, 72, 96)
            whole_action = action_batch
            self.critics_optimizer[agent].zero_grad()

            # print(whole_state.shape, whole_action.shape)
            current_Q = self.critics[agent](whole_state, whole_action)

            non_final_next_actions = [
                self.actors_target[i](non_final_next_states[:,
                                                            i,
                                                            :]) for i in range(
                                                                self.n_agents)]
            # print(len(non_final_next_actions), non_final_next_actions[0].shape)
            # non_final_next_actions = [torch.tensor(select_actions(action), dtype=torch.long) for action in non_final_next_actions]
            # print(len(non_final_next_actions), non_final_next_actions[0].shape, torch.stack(non_final_next_actions,dim=-1).shape)
            
            non_final_next_actions = torch.stack(non_final_next_actions,dim=1)
            # print(non_final_next_actions.shape)
            non_final_next_actions = (non_final_next_actions).max(dim=-1)[1]
            # print(non_final_next_actions.shape)

            target_Q = torch.zeros(
                self.batch_size).type(FloatTensor)

            # print(non_final_next_states.shape, non_final_next_actions.shape)
            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.reshape(self.batch_size, self.n_agents, self.args.history, 4, 72, 96)[:,agent,:,:3].reshape(self.batch_size, -1, 72, 96),
                non_final_next_actions
            ).squeeze()

            # print(target_Q.shape, reward_batch.shape)
            # scale_reward: to scale reward in Q functions
            # print(agent, reward_batch.shape)
            # print(agent, reward_batch[:, agent].shape)
            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent].unsqueeze(1) * self.scale_reward)

            # print(current_Q.shape, target_Q.shape)
            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critics_optimizer[agent].step()

            self.actors_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i).max(dim=-1)[1]
            ac = action_batch.clone()
            ac[:, agent] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actors_optimizer[agent].step()
            c_loss.append(loss_Q.detach().item())
            a_loss.append(actor_loss.detach().item())

            self.tbx.add_scalar('train/' + "actor_loss_agent_" + str(agent), actor_loss, self.num_updates+1)
            self.tbx.add_scalar('train/' + "critic_loss_agent_" + str(agent), loss_Q, self.num_updates+1)

        self.num_updates += 1

        

        if self.num_updates % self.args.display_interval == 0:
                self.logger.info('[{}] Update: {} / {}, Frames: {}, Rewards: {}, Min: {}, Max: {}, PL: {},'\
                    'VL: {}'.format(datetime.now(), self.num_updates, self.total_updates, (self.num_updates + 1)*self.args.nsteps*self.args.num_workers, \
                    torch.stack(self.total_reward).mean(dim=0).detach().numpy(), torch.stack(self.total_reward).min(dim=0)[0].detach().numpy(), torch.stack(self.total_reward).max(dim=0)[0].detach().numpy(), a_loss, c_loss))
                # save the model
                torch.save({"actor":self.actor.state_dict(), "critic":self.critic.state_dict()}, self.model_path + '/model.pt')

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            soft_update(self.critics_target, self.critics, self.tau)
            for i in range(self.n_agents):
                
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch):
        # state_batch: n_agents x state_dim
        actions = torch.zeros(
            self.n_agents,
            self.n_actions)
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            act = self.actors[i](sb.unsqueeze(0)).squeeze()

            act = torch.max(act,dim=1)[1]

            actions[i, :] = act
        self.steps_done += 1

        return actions
