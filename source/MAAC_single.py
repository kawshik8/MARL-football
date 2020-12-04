import numpy as np
from itertools import count
from arguments import get_args
from train_example import *
# from MARL_utils import *
from models import *
import torch
import torch.nn as nn
from torch import from_numpy
from torch.optim.adam import Adam
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from collections import namedtuple
import copy
import os
import random
import pickle
from collections import deque
from utils import config_logger
torch.autograd.set_detect_anomaly(True)
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import torch.distributed as dist
from torch.multiprocessing import Process

# Transition = namedtuple('Experience',('states', 'actions', 'next_states', 'rewards', 'dones'))

class Transition:
    def __init__(self, args):
        (states, actions, next_states, rewards, dones) = args
        self.states = states
        self.actions = actions 
        self.next_states = next_states
        self.rewards = rewards 
        self.dones = dones

        self.lock = lock

class ReplayMemory:
    def __init__(self, capacity, history, nagents, lock):
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
        self.lock = lock

    def push(self, states, actions, next_states, rewards, dones):
        # for i in range(states.shape[0]):

        # self.lock.acquire()
        self.states[self.position] = (states)
        self.next_states[self.position] = (next_states)
        self.rewards[self.position] = (rewards)
        self.dones[self.position] = dones
        self.actions[self.position] = (actions)

        
        self.position = int((self.position + 1) % self.capacity)
        if self.current_capacity < self.capacity:
            self.current_capacity = self.current_capacity + 1
        # self.lock.release()

    def sample(self, batch_size):
        random_indices = np.random.choice(np.arange(self.current_capacity), batch_size, replace=False)

        return self.states[random_indices], self.actions[random_indices], self.next_states[random_indices],  self.rewards[random_indices], self.dones[random_indices]

    def share_memory(self):
        self.states.share_memory_()
        self.next_states.share_memory_()
        self.actions.share_memory_()
        self.rewards.share_memory_()
        self.dones.share_memory_()

    def __len__(self):
        return self.current_capacity

class MAAC:
    def __init__(self, args, envs, memory=None, log=None):

        self.args = args
        self.envs = envs
        self.history = args.history
        self.max_grad_norm_critic = args.max_grad_norm_critic
        self.max_grad_norm_actor = args.max_grad_norm_actor
        # print("n states: ", self.n_states)
        self.n_actions = 19
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
        
        # if memory:
        self.memory = memory
        # else:
        #     self.memory = ReplayMemory(capacity=self.memory_size, history=args.history, nagents = args.n_agents)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        #Global_State_Net(args.history, nagents=args.n_agents)
        #Global_State_Net(args.history, nagents=args.n_agents)

        if args.tie_actor_wts:
            self.global_state_actor = backbone(args.history, nchannels=4)
            self.actor = Actor(args.history, 19, global_state_net = self.global_state_actor)
        if args.tie_critic_wts:
            self.global_state_critic = backbone(args.history, nchannels=3)
            self.critic = Critic(args.history, args.action_dim, args.n_agents, nactions = 19, global_state_net = self.global_state_critic, out_dim = self.n_actions)

        self.actors = [Actor(args.history, 19) if not args.tie_actor_wts else self.actor for i in range(args.n_agents)]
        self.critics = [Critic(args.history, args.action_dim, args.n_agents, nactions = 19, out_dim = self.n_actions) if not args.tie_critic_wts else self.critic for i in range(args.n_agents)]

        # print(self.actor)
        # print(self.critic)

        self.actors_target = copy.deepcopy(self.actors)
        self.target_critics = copy.deepcopy(self.critics)

        for i in range(self.n_agents):
            self.target_critics[i].eval()

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

    def scale_shared_grads(self, model):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1. / self.n_agents)
                
    def unpack(self, batch):

        states, actions, next_states, rewards, dones = batch

        # print(torch.stack(states).shape, torch.stack(rewards).shape, torch.cat(actions).shape, torch.stack(next_states).shape, torch.stack(dones).shape)

        states = torch.from_numpy(states).float().to(self.device)
        # states = states.view(self.batch_size, self.n_states * self.history)
        
        # print(torch.cat(batch.rewards)[0])
        rewards = torch.from_numpy(rewards).float().to(self.device)
        # rewards = rewards.view(self.batch_size, 1)
        
        dones = torch.from_numpy(dones).int().to(self.device)
        # dones = dones.view(self.batch_size, 1)

        actions = torch.from_numpy(actions).long().to(self.device)
        # actions = actions.view(self.batch_size, self.n_actions)

        next_states = torch.from_numpy(next_states).float().to(self.device)
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

        self.episode_no = 0
        episodes_done = 0

        i_t = 0
        for i in range(start, end):
            states = np.array(self.envs.reset()).transpose(0,3,1,2)
            # states = self._get_tensors(states)
        
            episode_reward = 0
            negative_count = 0

            if not self.args.test_model:
                for a_i in range(self.n_agents):
                    self.critics[a_i].eval()
                    self.actors[a_i].eval()

            for t in count():

                if len(self.memory) > self.start_steps:
                    actions,_,_,_ = self.choose_actions(np.expand_dims(states,axis=0))
                    actions = actions.detach().cpu().numpy().reshape(-1)
                else:
                    actions = self.envs.action_space.sample()

                # print(states.shape, actions.shape)

                next_state, reward, dones, _ = self.envs.step(actions)
                next_state = next_state.transpose(0,3,1,2)

                if self.args.test_model:
                    # if dist.get_rank() == 0:
                        self.envs.render()

                # print("rank: ",dist.get_rank(), len(self.memory))
                # print(states.shape, actions.shape, next_state.shape, dones)
                # print(reward)

                i_t += self.args.num_workers
                if not self.args.test_model:
                    # print(states.shape, states.dtype, actions.shape, actions.dtype, next_state.shape, next_state.dtype, reward.shape, reward.dtype, dones.shape, dones.dtype)
                    self.memory.push(states, actions, next_state, reward, dones)
                    for a_i in range(self.n_agents):
                        self.critics[a_i].train()
                        self.actors[a_i].train()
                    # if len(self.memory) > 5:
                    #     states, actions, next_states, rewards, dones = self.memory.sample(5)
                    #     print(states.shape)
                    #     print(actions.shape)
                    #     print(next_states.shape)
                    #     print(rewards.shape)
                    #     print(dones.shape)
                    if len(self.memory) > self.start_steps and len(self.memory) > self.batch_size and (i_t % 100) < self.args.num_workers:
                        batch = self.memory.sample(self.batch_size)
                        self.update_critics(batch)
                        self.update_actors(batch)
                        for a_i in range(self.n_agents):
                            self.soft_update_target_network(self.critics[a_i], self.target_critics[a_i], tau=0.005)
                            # soft_update_target_network(self.actors[i], self.target_actors[i], tau=0.005)
                        # alpha_loss, q_loss, policy_loss = self.train()

                    for a_i in range(self.n_agents):
                        self.critics[a_i].eval()
                        self.actors[a_i].eval()

                episode_reward += reward

                # print(negative_count)


                # print(reward)
                # env.render()
                state = next_state

                
                # for n, done in enumerate(dones):
                if dones:
                    
                    # states = states * 0
                    # print("episodes_done",episodes_done)
                    episodes_done += 1
                    self.episode_no += 1
                    break

                # print(t, len(self.memory))
            # episode_reward = np.where(episodes_done > 0, episode_reward / episodes_done, episode_reward)
            
            # print(episode_reward)
            self.running_reward.append(episode_reward)
            # print(self.running_reward)

            if not(len(self.memory) > self.start_steps and len(self.memory) > self.batch_size):
                # if dist.get_rank() == 0:
                    print("current buffer size: ", len(self.memory))

            self.log.info("iter: {} | episode: {} | duration: {} \n mean episode reward among workers: {} \n running rewards: Mean: {}, Std: {}".format(i, self.episode_no, t, np.mean(episode_reward,axis=0), np.mean(self.running_reward, axis=(0,1)), np.std(self.running_reward, axis=(0,1))))
            self.tbx.add_scalar('rewards/total_reward', np.mean(episode_reward), i)
            self.tbx.add_scalar('rewards/running_reward', np.mean(self.running_reward), i)

            if not self.args.test_model and i % 25 < self.args.num_workers:
                torch.save({"actor": ([self.actors[i].state_dict() for i in range(self.n_agents)] if not self.args.tie_actor_wts else self.actor.state_dict()), 
                            "critic": ([self.critics[i].state_dict() for i in range(self.n_agents)] if not self.args.tie_actor_wts else self.critic.state_dict()),
                            "actor_opt": ([self.actors_optimizer[i].state_dict() for i in range(self.n_agents)] if not self.args.tie_actor_wts else self.actors_optimizer.state_dict()),
                            "critic_opt": ([self.critics_optimizer[i].state_dict() for i in range(self.n_agents)] if not self.args.tie_actor_wts else self.critics_optimizer.state_dict()),
                            "num_updates": self.num_updates,
                            "episode_no": self.episode_no,
                            "running_reward": self.running_reward,
                            }, self.args.save_dir + "agent.ckpt")

                pickle.dump({"states": self.memory.states, 
                            "next_states": self.memory.next_states, 
                            "actions": self.memory.actions, 
                            "rewards": self.memory.rewards, 
                            "dones": self.memory.dones, 
                            "position": self.memory.position,
                            "curr_capacity": self.memory.current_capacity}, open(self.args.save_dir + "memory.pkl",'wb+'), protocol=4)


    def update_critics(self, batch):

        states, rewards, dones, actions, next_states = self.unpack(batch)

        # Calculating the Q-Value target
        with torch.no_grad():
            next_actions, next_log_probs, _, _ = self.choose_actions(next_states)

        q_losses = []
        grad_norms = []
        criterion = nn.MSELoss()
        for i in range(self.n_agents):
            self.critics[i].train()

            next_state_qs = self.target_critics[i](
                                                    next_states.reshape(self.batch_size, self.n_agents, self.args.history, 4, 72, 96)[:,i,:,:3].reshape(self.batch_size, -1, 72, 96),
                                                    F.one_hot(torch.cat([next_actions[:,:i], next_actions[:,i+1:]],dim=1),self.n_actions).float()
                                                    ).gather(-1, next_actions[:,i].unsqueeze(-1)).squeeze(-1) 
        
            qs = self.critics[i](
                                states.reshape(self.batch_size, self.n_agents, self.args.history, 4, 72, 96)[:,i,:,:3].reshape(self.batch_size, -1, 72, 96),
                                F.one_hot(torch.cat([actions[:,:i], actions[:,i+1:]],dim=1),self.n_actions).float()
                                ).gather(-1, actions[:,i].unsqueeze(-1)).squeeze(-1)

            # print(rewards[:,i].shape, next_log_probs[:,i].shape, dones[:,i].shape)
            target_qs = rewards[:,i] + self.gamma * (next_state_qs - self.alpha * next_log_probs[:,i]) * (1 - dones[:,i])

            # print(qs.shape, target_qs.shape, next_state_qs.shape)
            q_loss = criterion(qs, target_qs.detach())
            if self.args.tie_critic_wts:
                q_losses += [q_loss]
            else:
                self.critics_optimizer[i].zero_grad()
                q_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm(self.critics[i].parameters(), self.args.max_grad_norm_critic)
                grad_norms += [grad_norm]
                # pool_average_gradients(self.critics[i])
                self.critics_optimizer[i].step()

        if self.args.tie_critic_wts:
            q_loss = sum(q_losses)
            q_loss.backward()
            self.scale_shared_grads(self.critic)
            grad_norm = torch.nn.utils.clip_grad_norm(self.critic.parameters(), self.args.max_grad_norm_critic * self.n_agents)
            # pool_average_gradients(self.critic)
            self.critics_optimizer.step()
        
            if self.log is not None:
                self.tbx.add_scalar('losses/q_loss', q_loss, self.num_updates)
                self.tbx.add_scalar('grad_norms/q', grad_norm, self.num_updates)

        else:
            if self.log is not None:
                for i in range(self.n_agents):
                    self.tbx.add_scalar('losses/q_loss_' + str(i), q_losses[i], self.num_updates)
                    self.tbx.add_scalar('grad_norms/q_' + str(i), grad_norms[i], self.num_updates)
            
    def update_actors(self, batch):
        
        states, rewards, dones, actions, next_states = self.unpack(batch)

        all_actions = []
        all_probs = []
        all_log_probs = []
        all_entropies = []

        with torch.no_grad():
            all_actions, all_log_probs, all_entropies, all_probs = self.choose_actions(states)

        losses = []
        for i in range(self.n_agents):
            if not self.args.tie_actor_wts:
                self.actors_optimizer[i].zero_grad()

            all_qs = self.critics[i](
                                states.reshape(self.batch_size, self.n_agents, self.args.history, 4, 72, 96)[:,i,:,:3].reshape(self.batch_size, -1, 72, 96),
                                F.one_hot(torch.cat([actions[:,:i], actions[:,i+1:]],dim=1),self.n_actions).float()
                                )
            qs = all_qs.gather(-1, actions[:,i].unsqueeze(-1)) 
            marg_q = (all_qs * all_probs[:,i]).sum(dim=1, keepdim=True)

            loss = (all_log_probs[:,i] * (all_log_probs[:,i] * self.alpha - (qs - marg_q))).mean()
            # print(loss)

            for param in self.critics[i].parameters(): 
                param.requires_grad = False

            if not self.args.tie_actor_wts:
                loss.backward()
            else:
                losses += [loss]

            for param in self.critics[i].parameters(): 
                param.requires_grad = True

            if not self.args.tie_actor_wts:
                grad_norm = torch.nn.utils.clip_grad_norm(self.actors[i].parameters(), self.args.max_grad_norm_actor)
                # pool_average_gradients(self.actors[i])
                self.actors_optimizer[i].step()

                if self.log is not None:
                    self.tbx.add_scalar('losses/actor_loss_' + str(i), loss, self.num_updates)
                    self.tbx.add_scalar('grad_norms/pi_' + str(i), grad_norm, self.num_updates)

        if self.args.tie_actor_wts:
            loss = sum(losses)
            loss.backward()
            self.scale_shared_grads(self.actor)
            grad_norm = torch.nn.utils.clip_grad_norm(self.actor.parameters(), self.args.max_grad_norm_actor * self.n_agents)
            # pool_average_gradients(self.actor)
            self.actors_optimizer.step()

            if self.log is not None:
                self.tbx.add_scalar('losses/actor_loss', loss, self.num_updates)
                self.tbx.add_scalar('grad_norms/pi', grad_norm, self.num_updates)

        self.num_updates += 1

    def choose_actions(self, obs):

        actions = []
        log_probs = []
        entropies = []
        probs = []

        # print(obs.shape)
        for i in range(self.n_agents):
            action = self.actors[i](torch.tensor(obs[:,i], dtype=torch.float))
            if self.args.test_model:
                action = torch.max(action,dim=-1)[1]
            else:
                dist = torch.distributions.Categorical(logits=action)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            probs.append(dist.probs)

        # print(actions[0].shape)
        # print(actions)
        # print(log_probs)
        # print(entropies)
        # print(probs)
        actions = torch.stack(actions,dim=1)
        log_probs = torch.stack(log_probs,dim=1)
        entropies = torch.stack(entropies,dim=1)
        probs = torch.stack(probs,dim=1)

        return actions, log_probs, entropies, probs

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

""" Gradient averaging. """
def pool_average_gradients(model):
    size = float(dist.get_world_size())
    print("world size: ", size)
    for param in model.parameters():
        print("before: ", param.grad.data)
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        print("after: ",param.grad.data)
        param.grad.data /= size

def make_parallel_env(args, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = create_single_football_env(args)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    # if n_rollout_threads == 1:
    #     return DummyVecEnv([get_env_fn(0)])
    # else:
    return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(rank, size, args, memory):
    """ Distributed function to be implemented later. """
    print("Rank is ", dist.get_rank())
    env = create_single_football_env(args)
    maac_trainer = MAAC(args, env, memory)
    maac_trainer.learn()

    envs.close()
    pass

def init_process(rank, size, fn, args, memory, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, args, memory)


if __name__ == '__main__':
    args = get_args()

    # envs = make_parallel_env(args, args.num_workers, 1)
    # print(envs.action_space, envs.observation_space)
    # maac_trainer = MAAC(args, envs)
    # maac_trainer.learn()

    # envs.close()

    lock = torch.multiprocessing.Lock()

    memory = ReplayMemory(capacity=args.buffer_size, history=args.history, nagents = args.n_agents, lock=lock)
    # memory.share_memory()

    env = create_single_football_env(args)
    maac_trainer = MAAC(args, env, memory)
    maac_trainer.learn()

    env.close()

    # processes = []
    # for rank in range(args.num_workers):
    #     p = Process(target=init_process, args=(rank, args.num_workers, run, args, memory))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

