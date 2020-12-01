import numpy as np
import torch
from torch import optim
from utils import *
from datetime import datetime
from models import *
import os
import copy
from tensorboardX import SummaryWriter

class mappo_agent:
    def __init__(self, envs, args):
        self.envs = envs 
        self.args = args
        self.batch_size = args.batch_size
        # define the newtork...

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

        self.actors_target = copy.deepcopy(self.actors)
        self.critics_target = copy.deepcopy(self.critics)

        self.n_agents = args.n_agents

        self.use_cuda = args.cuda
        # if use the cuda...
        if self.use_cuda:
            
            for x in self.actors:
                x.cuda()
            self.critics.cuda()

            for x in self.actors_target:
                x.cuda()
            self.critics_target.cuda()

        # define the optimizer...
        if not args.tie_actor_wts:
            self.actors_optimizer = [torch.optim.Adam(x.parameters(),lr=args.lr) for x in self.actors]
        else:
            self.actors_optimizer = torch.optim.Adam(self.actor.parameters(),lr=args.lr)
        if not args.tie_critic_wts:
            self.critics_optimizer = [torch.optim.Adam(x.parameters(),lr=args.lr) for x in self.critics]
        else:
            self.critics_optimizer = torch.optim.Adam(self.critic.parameters(),lr=args.lr)

        # check saving folder..
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # env folder..
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        # logger folder
        if not os.path.exists(self.args.log_dir):
            os.mkdir(self.args.log_dir)
        self.log_path = self.args.log_dir + self.args.env_name + '.log'
        # get the observation
        self.batch_ob_shape = (self.args.num_workers * self.args.nsteps, ) + self.envs.observation_space.shape
        self.obs = np.zeros((self.args.num_workers, ) + self.envs.observation_space.shape, dtype=self.envs.observation_space.dtype.name)
        self.obs[:] = self.envs.reset()
        self.dones = torch.tensor([False for _ in range(self.args.num_workers)])

        self.log_path = self.args.log_dir + self.args.env_name + '.log'
        self.logger = config_logger(self.log_path)

        no = 1
        while os.path.exists(self.args.log_dir + self.args.env_name + "_" + str(no)): 
            no +=1

        self.tbx = SummaryWriter(self.args.log_dir + self.args.env_name + "_" + str(no))

    # start to train the network...
    def learn(self):
        num_updates = self.args.total_frames // (self.args.nsteps * self.args.num_workers)
        # get the reward to calculate other informations
        episode_rewards = torch.zeros([self.args.num_workers, 2])
        final_rewards = torch.zeros([self.args.num_workers, 2])
        for update in range(num_updates):
            mb_obs, mb_rewards, mb_actions, mb_dones, mb_values = [], [], [], [], []
            if self.args.lr_decay:
                self._adjust_learning_rate(update, num_updates)
            for step in range(self.args.nsteps):
                with torch.no_grad():
                    # get tensors
                    # print(self.obs.shape)
                    obs_tensor = self._get_tensors(self.obs)
                    # print(obs_tensor.shape)
                    # select actions
                    actions = [self.actors[i](obs_tensor[:,i]) for i in range(self.n_agents)]
                    # print(actions)
                    actions = [torch.tensor(select_actions(action), dtype=torch.long) for action in actions]
                    # print(actions)
                    actions = torch.stack(actions,dim=1)
                    # print(actions.shape)

                    values = [self.critics[agent](
                        obs_tensor.reshape(self.batch_size, self.n_agents, self.args.history, 4, 72, 96)[:,agent,:,:3].reshape(self.batch_size, -1, 72, 96),
                        actions,
                    ).squeeze() for agent in range(self.n_agents)] 
                    values = torch.stack(values,dim=1)

                # get the input actions
                input_actions = actions 

                # start to store information
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(actions)
                mb_dones.append(self.dones)
                mb_values.append(values.detach().cpu().numpy().squeeze())
                

                # start to excute the actions in the environment
                obs, rewards, dones, _ = self.envs.step(input_actions.numpy())
                print(rewards.shape)
                mb_rewards.append(rewards)

                # update dones
                # print(dones)
                self.dones = torch.tensor(dones)
                                
                

                # clear the observation
                for n, done in enumerate(dones):
                    if done:
                        self.obs[n] = self.obs[n] * 0
                self.obs = obs

                # process the rewards part -- display the rewards on the screen
                # print("rewards: ",rewards.shape)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                
                
                episode_rewards += rewards
                masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in dones], dtype=torch.float32)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks

            # process the rollouts
            mb_obs = np.asarray(mb_obs, dtype=np.float32)

            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            print(mb_rewards.shape)
            mb_actions = np.asarray(mb_actions, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            mb_values = np.asarray(mb_values, dtype=np.float32)
            print(mb_obs.shape, mb_rewards.shape, mb_actions.shape)
            print(mb_dones.shape, mb_values.shape)

            # compute the last state value
            with torch.no_grad():
                obs_tensor = self._get_tensors(self.obs)
                last_values, _ = self.net(obs_tensor)
                last_values = last_values.detach().cpu().numpy().squeeze()

            # start to compute advantages...
            mb_returns = np.zeros_like(mb_rewards)
            mb_advs = np.zeros_like(mb_rewards)
            lastgaelam = 0
            for t in reversed(range(self.args.nsteps)):
                if t == self.args.nsteps - 1:
                    nextnonterminal = 1.0 - self.dones
                    nextvalues = last_values
                else:
                    nextnonterminal = 1.0 - mb_dones[t + 1]
                    nextvalues = mb_values[t + 1]
                delta = mb_rewards[t] + self.args.gamma * nextvalues * nextnonterminal - mb_values[t]
                mb_advs[t] = lastgaelam = delta + self.args.gamma * self.args.tau * nextnonterminal * lastgaelam
            mb_returns = mb_advs + mb_values

            # after compute the returns, let's process the rollouts
            mb_obs = mb_obs.swapaxes(0, 1).reshape(self.batch_ob_shape)
            mb_actions = mb_actions.swapaxes(0, 1).flatten()
            mb_returns = mb_returns.swapaxes(0, 1).flatten()
            mb_advs = mb_advs.swapaxes(0, 1).flatten()

            # before update the network, the old network will try to load the weights
            self.old_net.load_state_dict(self.net.state_dict())

            # start to update the network
            pl, vl, ent = self._update_network(mb_obs, mb_actions, mb_returns, mb_advs)

            # display the training information
            if update % self.args.display_interval == 0:
                self.logger.info('[{}] Update: {} / {}, Frames: {}, Rewards: {:.3f}, Min: {:.3f}, Max: {:.3f}, PL: {:.3f},'\
                    'VL: {:.3f}, Ent: {:.3f}'.format(datetime.now(), update, num_updates, (update + 1)*self.args.nsteps*self.args.num_workers, \
                    final_rewards.mean().item(), final_rewards.min().item(), final_rewards.max().item(), pl, vl, ent))
                # save the model
                torch.save(self.net.state_dict(), self.model_path + '/model.pt')

    # update the network
    def _update_network(self, obs, actions, returns, advantages):
        inds = np.arange(obs.shape[0])
        nbatch_train = obs.shape[0] // self.args.batch_size
        for _ in range(self.args.epoch):
            np.random.shuffle(inds)
            for start in range(0, obs.shape[0], nbatch_train):
                # get the mini-batchs
                end = start + nbatch_train
                mbinds = inds[start:end]
                mb_obs = obs[mbinds]
                mb_actions = actions[mbinds]
                mb_returns = returns[mbinds]
                mb_advs = advantages[mbinds]
                # convert minibatches to tensor
                mb_obs = self._get_tensors(mb_obs)
                mb_actions = torch.tensor(mb_actions, dtype=torch.float32)
                mb_returns = torch.tensor(mb_returns, dtype=torch.float32).unsqueeze(1)
                mb_advs = torch.tensor(mb_advs, dtype=torch.float32).unsqueeze(1)
                # normalize adv
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)
                if self.args.cuda:
                    mb_actions = mb_actions.cuda()
                    mb_returns = mb_returns.cuda()
                    mb_advs = mb_advs.cuda()
                # start to get values
                mb_values, pis = self.net(mb_obs)
                # start to calculate the value loss...
                value_loss = (mb_returns - mb_values).pow(2).mean()
                # start to calculate the policy loss
                with torch.no_grad():
                    _, old_pis = self.old_net(mb_obs)
                    # get the old log probs
                    old_log_prob, _ = evaluate_actions(old_pis, mb_actions)
                    old_log_prob = old_log_prob.detach()
                # evaluate the current policy
                log_prob, ent_loss = evaluate_actions(pis, mb_actions)
                prob_ratio = torch.exp(log_prob - old_log_prob)
                # surr1
                surr1 = prob_ratio * mb_advs
                surr2 = torch.clamp(prob_ratio, 1 - self.args.clip, 1 + self.args.clip) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                # final total loss
                total_loss = policy_loss + self.args.vloss_coef * value_loss - ent_loss * self.args.ent_coef
                # clear the grad buffer
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.max_grad_norm)
                # update
                self.optimizer.step()
        return policy_loss.item(), value_loss.item(), ent_loss.item()

    # convert the numpy array to tensors
    def _get_tensors(self, obs):
        # print(obs.shape)
        obs_tensor = torch.tensor(np.transpose(obs, (0, 1, 4, 2, 3)), dtype=torch.float32)
        # obs_tensor = torch.tensor(obs, dtype=torch.float32)
        # decide if put the tensor on the GPU
        if self.args.cuda:
            obs_tensor = obs_tensor.cuda()
        return obs_tensor

    # adjust the learning rate
    def _adjust_learning_rate(self, update, num_updates):
        lr_frac = 1 - (update / num_updates)
        adjust_lr = self.args.lr * lr_frac
        for param_group in self.optimizer.param_groups:
             param_group['lr'] = adjust_lr