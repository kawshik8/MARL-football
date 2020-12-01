import numpy as np
from itertools import count
from arguments import get_args
from train_example import create_single_football_env
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from models import *
import torch
from torch import from_numpy
from torch.optim.adam import Adam
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from collections import namedtuple
import copy
import os
import random
from collections import deque
from utils import config_logger
torch.autograd.set_detect_anomaly(True)

# Transition = namedtuple('Experience',('states', 'actions', 'next_states', 'rewards', 'dones'))

class Transition:
    def __init__(self, args):
        (states, actions, next_states, rewards, dones) = args
        self.states = states
        self.actions = actions 
        self.next_states = next_states
        self.rewards = rewards 
        self.dones = dones

class ReplayMemory:
    def __init__(self, capacity, history, nagents):
        self.capacity = capacity
        self.history = history 
        self.nagents = nagents
        self.states = np.zeros((int(capacity), nagents, history * 4, 72, 96))
        self.next_states = np.zeros((int(capacity), nagents, history * 4, 72, 96))
        self.actions = np.zeros((int(capacity), nagents))
        self.rewards = np.zeros((int(capacity), nagents))
        self.dones = np.zeros((int(capacity), nagents))
        self.position = 0
        self.current_capacity = 0

    def push(self, states, actions, next_states, rewards, dones):
        for i in range(states.shape[0]):
            self.states[self.position] = states[0]
            self.next_states[self.position] = next_states[0]
            self.rewards[self.position] = rewards[0]
            self.dones[self.position] = dones[0]
            self.actions[self.position] = actions[0]
            self.position = int((self.position + 1) % self.capacity)
            if self.current_capacity < self.capacity:
                self.current_capacity = self.current_capacity + 1

    def sample(self, batch_size):
        random_indices = np.random.choice(np.arange(self.current_capacity), batch_size, replace=False)

        return self.states[random_indices], self.actions[random_indices], self.next_states[random_indices],  self.rewards[random_indices], self.dones[random_indices]


    def __len__(self):
        return self.current_capacity

class MAAC:
    def __init__(self, args, envs, log=None):

        self.args = args
        self.envs = envs
        self.history = args.history
        self.n_states = 24
        self.max_grad_norm = args.max_grad_norm
        # print("n states: ", self.n_states)
        self.n_actions = 4
        self.memory_size = args.buffer_size
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.lr = args.lr
        self.reward_scale = args.reward_scale
        self.log_path = self.args.log_dir + self.args.env_name + '.log'
        self.log = config_logger(self.log_path)
        self.start_steps = args.start_steps
        self.tau = args.polyak_tau
        self.soft_update = args.soft_update
        self.n_agents = args.n_agents

        self.tbx = SummaryWriter(args.save_dir)
        
        self.memory = ReplayMemory(capacity=self.memory_size, history=args.history, nagents = args.n_agents)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.global_state_actor = backbone(args.history, nchannels=4)#Global_State_Net(args.history, nagents=args.n_agents)
        self.global_state_critic = backbone(args.history, nchannels=3)#Global_State_Net(args.history, nagents=args.n_agents)

        if args.tie_actor_wts:
            self.actor = Actor(args.history, 19, global_state_net = self.global_state_actor)
        if args.tie_critic_wts:
            self.critic = Critic(args.history, args.action_dim, args.n_agents, nactions = 19, global_state_net = self.global_state_critic)

        self.actors = [Actor(args.history, 19) if not args.tie_actor_wts else self.actor for i in range(args.n_agents)]
        self.critics = [Critic(args.history, args.action_dim, args.n_agents, nactions = 19) if not args.tie_critic_wts else self.critic for i in range(args.n_agents)]

        # print(self.actor)
        # print(self.critic)

        self.critics_target = copy.deepcopy(self.critics)

        for i in range(self.n_agents):
            self.critics_target[i].eval()

        self.target_alpha = -np.prod([4,])
        self.log_alpha = torch.zeros(1, requires_grad=False, device=self.device) + torch.log(torch.tensor(self.alpha))

        if not args.tie_actor_wts:
            self.actors_optimizer = [torch.optim.Adam(x.parameters(),lr=args.lr) for x in self.actors]
        else:
            self.actors_optimizer = torch.optim.Adam(self.actor.parameters(),lr=args.lr)
            
        if not args.tie_critic_wts:
            self.critics_optimizer = [torch.optim.Adam(x.parameters(),lr=args.lr) for x in self.critics]
        else:
            self.critics_optimizer = torch.optim.Adam(self.critic.parameters(),lr=args.lr)

        # self.alpha_opt = Adam([self.log_alpha], lr=args.alpha_lr)

        self.num_updates = 0
        self.num_steps = 128

        self.running_reward = deque([], maxlen=100)

    def unpack(self, batch):

        states, actions, next_states, rewards, dones = batch

        # print(torch.stack(states).shape, torch.stack(rewards).shape, torch.cat(actions).shape, torch.stack(next_states).shape, torch.stack(dones).shape)

        states = torch.from_numpy(states, dtype=torch.float).to(self.device)
        # states = states.view(self.batch_size, self.n_states * self.history)
        
        # print(torch.cat(batch.rewards)[0])
        rewards = torch.from_numpy(rewards, dtype=torch.float).to(self.device)
        # rewards = rewards.view(self.batch_size, 1)
        
        dones = torch.from_numpy(dones, dtype=torch.bool).to(self.device)
        # dones = dones.view(self.batch_size, 1)

        actions = torch.from_numpy(actions, dtype=torch.long).to(self.device)
        # actions = actions.view(self.batch_size, self.n_actions)

        next_states = torch.from_numpy(next_states, dtype=torch.float).to(self.device)
        # next_states = next_states.view(self.batch_size, self.n_states * self.history)

        # print(states.shape, actions.shape, next_states.shape, rewards.shape)

        return states, rewards, dones, actions, next_states

    def _get_tensors(self, obs):
        obs_tensor = torch.tensor(np.transpose(obs, (0, 1, 4, 2, 3)), dtype=torch.float32)
        # obs_tensor = torch.tensor(obs, dtype=torch.float32)
        # decide if put the tensor on the GPU
        if self.args.cuda:
            obs_tensor = obs_tensor.cuda()
        return obs_tensor

    def learn(self):
        start = 0
        end = self.args.n_episodes

        for i in range(start,end):
            states = np.array(self.envs.reset()).transpose(0,1,4,2,3)
            # states = self._get_tensors(states)
        
            episode_reward = 0
            negative_count = 0

            episodes_done = np.array([[0],[0]])

            for t in range(self.num_steps):

                if len(self.memory) > self.start_steps:
                    actions = self.choose_actions(states)
                else:
                    actions = np.stack([self.envs.action_space.sample() for i in range(self.args.num_workers)])

                # print(states.shape, actions.shape)

                next_state, reward, dones, _ = self.envs.step(actions)
                next_state = next_state.transpose(0,1,4,2,3)

                if self.args.test_model:
                    self.envs.render()

                # print(states.shape, actions.shape, next_state.shape)
                # print(reward)

                if not self.args.test_model:
                    # print(states.shape, states.dtype, actions.shape, actions.dtype, next_state.shape, next_state.dtype, reward.shape, reward.dtype, dones.shape, dones.dtype)
                    self.memory.push(states, actions, next_state, reward, dones)
                    # if len(self.memory) > 5:
                    #     states, actions, next_states, rewards, dones = self.memory.sample(5)
                    #     print(states.shape)
                    #     print(actions.shape)
                    #     print(next_states.shape)
                    #     print(rewards.shape)
                    #     print(dones.shape)
                    # alpha_loss, q_loss, policy_loss = self.train()

                episode_reward += reward

                # print(negative_count)


                # print(reward)
                # env.render()
                state = next_state

                for n, done in enumerate(dones):
                    if done:
                        states[n] = states[n] * 0
                        episodes_done[n] += 1
                states = states

                # print(t, len(self.memory))

            episode_reward = np.where(episodes_done > 0, episode_reward / episodes_done, episode_reward)
            
            self.running_reward.append(episode_reward)
            # print(self.running_reward)

            self.log.info("episode: {} | duration: {} \n mean episode reward among workers: {} \n running rewards: Mean: {}, Std: {}".format(i, t, np.mean(episode_reward,axis=0), np.mean(self.running_reward, axis=(0,1)), np.std(self.running_reward, axis=(0,1))))

    def train(self):
        for i in range(self.n_agents):
            self.critics[i].train()
            self.actors[i].train()

        if len(self.memory) < self.start_steps:
            return 0, 0, 0
        else:
            batch = self.memory.sample(self.batch_size)
            states, rewards, dones, actions, next_states = self.unpack(batch)

            # Calculating the Q-Value target
            with torch.no_grad():
                next_reparam_actions, next_log_probs, _ = self.policy_network.sample(next_states)
                next_q1 = self.q_value_target_network1(next_states, next_reparam_actions)
                next_q2 = self.q_value_target_network2(next_states, next_reparam_actions)
                next_q = torch.min(next_q1, next_q2)
                target_q = self.reward_scale * rewards + self.gamma * (1 - dones) * (next_q - self.alpha * next_log_probs)

            q1 = self.q_value_network1(states, actions)
            q2 = self.q_value_network2(states, actions)
            q1_loss = F.mse_loss(q1, target_q.detach())
            q2_loss = F.mse_loss(q2, target_q.detach())

            self.q_value1_opt.zero_grad()
            self.q_value2_opt.zero_grad()
            q_loss = (q1_loss + q2_loss)/2
            q_loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.q_value_network1.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.q_value_network2.parameters(), self.max_grad_norm)
            self.q_value1_opt.step()
            self.q_value2_opt.step()         

            # Calculating the Policy target
            reparam_actions, log_probs, _ = self.policy_network.sample(states)
            # with torch.no_grad():
            q1 = self.q_value_network1(states, reparam_actions)
            q2 = self.q_value_network2(states, reparam_actions)
            q = torch.min(q1, q2)

            policy_loss = ((self.alpha * log_probs) - q).mean()   

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            alpha_loss = -(self.log_alpha * (log_probs + self.target_alpha).detach()).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            self.alpha = self.log_alpha.exp()

            if self.soft_update:
                self.soft_update_target_network(self.q_value_network1, self.q_value_target_network1, self.tau)
            else:
                self.hard_update_target_network(self.q_value_network1, self.q_value_target_network1)
            self.q_value_target_network1.eval()
            if self.soft_update:
                self.soft_update_target_network(self.q_value_network2, self.q_value_target_network2, self.tau)
            else:
                self.hard_update_target_network(self.q_value_network1, self.q_value_target_network1)
            self.q_value_target_network2.eval()

            self.num_updates += 1

            self.tbx.add_scalar("Loss/alpha", alpha_loss.item(), self.num_updates)
            self.tbx.add_scalar("Loss/actor", policy_loss.item(), self.num_updates)
            self.tbx.add_scalar("Loss/critic_1", q1_loss.item(), self.num_updates)
            self.tbx.add_scalar("Loss/critic_2", q2_loss.item(), self.num_updates)
            self.tbx.add_scalar("Loss/critic", 0.5 * q_loss.item(), self.num_updates)

            # print(q1.shape, q2.shape)
            self.tbx.add_scalar("avg_Qvalue/critic_1", q1.mean().item(), self.num_updates)
            self.tbx.add_scalar("avg_Qvalue/critic_2", q2.mean().item(), self.num_updates)

            

            # total_norm = 0
            # for p in self.policy_network.parameters():
            #     param_norm = p.grad.data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** (1. / 2)

            # self.tbx.add_scalar("grad_norm/critic_1", q1.mean().item(), self.num_updates)

            # print("policy_network grad norm: ", total_norm)

            # total_norm = 0
            # for p in self.q_value_network1.parameters():
            #     param_norm = p.grad.data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** (1. / 2)

            # print("q_value_network 1: ", total_norm)

            # total_norm = 0
            # for p in self.q_value_network2.parameters():
            #     param_norm = p.grad.data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** (1. / 2)

            # print("q_value_network 2: ", total_norm)



            return alpha_loss.item(), 0.5 * (q_loss).item(), policy_loss.item()

    def choose_action(self, obs):

        actions = []
        print(obs.shape)
        for i in range(self.n_agents):
            action = self.actors[i](obs[:,i])
            if self.args.test_model:
                action = torch.max(action,dim=-1)[1]
            else:
                dist = torch.Categorical(action)
                action = dist.sample()
            actions.append(action)

        print(actions[0].shape)
        actions = torch.stack(actions)

        return actions.detach().numpy().cpu()

    @staticmethod
    def soft_update_target_network(local_network, target_network, tau=0.005):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    @staticmethod
    def hard_update_target_network(local_network, target_network):
        #print("hard update")
        target_network.load_state_dict(local_network.state_dict())
                                  

    def save_weights(self):
        torch.save(self.policy_network.state_dict(), "./weights.pth")

    def load_weights(self):
        self.policy_network.load_state_dict(torch.load("./weights.pth"))

    def set_to_eval_mode(self):
        self.policy_network.eval()

if __name__ == '__main__':
    args = get_args()

    envs = SubprocVecEnv([(lambda _i=i: create_single_football_env(args)) for i in range(args.num_workers)], context=None)
    maac_trainer = MAAC(args, envs)
    maac_trainer.learn()

    envs.close()

