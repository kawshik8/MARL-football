import numpy as np
import torch
from torch import optim
from utils import *
from datetime import datetime
from models import *
import os
import copy
from tensorboardX import SummaryWriter
from torch.distributions import Categorical
from arguments import get_args
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from train_example import create_single_football_env
from collections import deque

class mappo_agent:
    def __init__(self, envs, args, env):
        self.envs = envs 
        self.env = env
        self.args = args
        self.batch_size = args.batch_size
        self.n_agents = args.ln_agents
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # define the newtork...

        self.global_state_actor = backbone(args.history, nchannels=4).to(self.device)
        self.global_state_critic = backbone(args.history, nchannels=3).to(self.device)

        if args.tie_actor_wts:
            self.actor = Actor(args.history, 19, global_state_net = self.global_state_actor).to(self.device)
        if args.tie_critic_wts:
            self.critic = Critic(args.history, args.action_dim, self.n_agents, nactions = 19, out_dim = 19, global_state_net = self.global_state_critic).to(self.device)

        self.actors = [Actor(args.history, 19).to(self.device) if not args.tie_actor_wts else self.actor for i in range(self.n_agents)]
        self.critics = [Critic(args.history, args.action_dim, self.n_agents, nactions = 19, out_dim = 19).to(self.device) if not args.tie_critic_wts else self.critic for i in range(self.n_agents)]

        # print(self.actor)
        # print(self.critic)

        self.actors_target = copy.deepcopy(self.actors)
        self.critics_target = copy.deepcopy(self.critics)

        self.use_cuda = args.cuda
        
        # if use the cuda...
        #for x in self.actors:
        #    x.to(self.device)
        #self.critics.to(self.device)

        #for x in self.actors_target:
        #    x.to(self.device)
        #self.critics_target.to(self.device)
            

        # define the optimizer...
        if not args.tie_actor_wts:
            self.actors_optimizer = [torch.optim.Adam(x.parameters(),lr=args.lr) for x in self.actors]
        else:
            self.actors_optimizer = torch.optim.Adam(self.actor.parameters(),lr=args.lr)
        if not args.tie_critic_wts:
            self.critics_optimizer = [torch.optim.Adam(x.parameters(),lr=args.lr) for x in self.critics]
        else:
            self.critics_optimizer = torch.optim.Adam(self.critic.parameters(),lr=args.lr)

        
        self.no = 1
        while os.path.exists(self.args.log_dir + self.args.env_name + "_" + str(self.no)): 
            self.no +=1
        
        self.tbx = SummaryWriter(self.args.log_dir + self.args.env_name + "_" + str(self.no))

        # env folder..
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # check saving folder..
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        
        # logger folder
        if not os.path.exists(self.args.log_dir):
            os.mkdir(self.args.log_dir)
        self.log_path = self.args.log_dir + self.args.env_name + "_" + str(self.no) + '/' + 'logs.log'

        # get the observation
        self.batch_ob_shape = (self.args.num_workers * self.args.nsteps, ) + self.envs.observation_space.shape
        self.obs = np.zeros((self.args.num_workers, ) + self.envs.observation_space.shape, dtype=self.envs.observation_space.dtype.name)
        self.obs[:] = self.envs.reset()
        self.dones = torch.tensor([False for _ in range(self.args.num_workers)])

        self.logger = config_logger(self.log_path)
        self.episodes_done = 0
        
        ckpt = torch.load('saved_models/2_vs_2/2_vs_2_1_agent.ckpt')
        if self.args.tie_actor_wts:
            self.actor.load_state_dict(ckpt['actor'])
            self.actors_optimizer.load_state_dict(ckpt['actor_opt'])
        else:
            for i in range(self.n_agents):
                self.actors[i].load_state_dict(ckpt['actor'][i])
                self.actors_optimizer[i].load_state_dict(ckpt['actor_opt'][i])
        if self.args.tie_critic_wts:
            self.critic.load_state_dict(ckpt['critic'])
            self.critics_optimizer.load_state_dict(ckpt['critic_opt'])
        else:
            for i in range(self.n_agents):
                self.critics[i].load_state_dict(ckpt['critic'][i])
                self.critics_optimizer[i].load_state_dict(ckpt['critic_opt'][i])

        self.actors_target = copy.deepcopy(self.actors)
        self.critics_target = copy.deepcopy(self.critics)

        self.last_update = ckpt['num_update'] 
       

    def test(self):


        total_reward = 0
        for episode in range(100):
            obs = self.env.reset()
            episode_reward = 0
            while True:
                action = self.env.action_space.sample()
                # print(action.shape)

                obs, rewards, dones, _ = self.env.step(action)

                episode_reward += rewards

                if dones:
                    break

            total_reward += episode_reward


        print(total_reward / 100)



    # start to train the network...
    def learn(self):
        num_updates = self.args.total_frames // (self.args.nsteps * self.args.num_workers)
        # get the reward to calculate other informations
        episode_rewards = torch.zeros([self.args.num_workers, self.n_agents])
        final_rewards = torch.zeros([self.args.num_workers, self.n_agents])
        
        episodes_done = np.zeros((self.args.num_workers,1))

        running_reward = deque([], maxlen=100)
        for update in range(self.last_update, num_updates):
            mb_obs, mb_rewards, mb_actions, mb_log_probs, mb_dones, mb_values = [], [], [], [], [], []
            if self.args.lr_decay:
                self._adjust_learning_rate(update, num_updates)
            episode_wise_rewards = torch.zeros([self.args.num_workers, self.n_agents])
            episodes_done = np.zeros((self.args.num_workers,1))
            for step in range(self.args.nsteps):
                with torch.no_grad():
                    # get tensors
                    # print(self.obs.shape)
                    obs_tensor = self._get_tensors(self.obs)
                    # print(obs_tensor.shape)
                    # select actions
                    actions = [self.actors[i](obs_tensor[:,i]) for i in range(self.n_agents)]
                    # print(actions)
                    acts = []
                    log_probs = []
                    for action in actions:
                        act, log_prob = self.select_actions(action)
                        act = torch.tensor(act, dtype = torch.long)
                        acts.append(act)
                        log_probs.append(log_prob)
                    actions = acts
                    # actions = [torch.tensor(self.select_actions(action), dtype=torch.long) for action in actions]
                    # print(actions)
                    actions = torch.stack(actions,dim=1).to(self.device)
                    log_probs = torch.stack(log_probs,dim=1).to(self.device)
                    # print(actions.shape)
                    values = [self.critics[agent](
                        obs_tensor.reshape(self.args.num_workers, self.n_agents, self.args.history, 4, 72, 96)[:,agent,:,:3].reshape(self.args.num_workers, -1, 72, 96),
                        F.one_hot(torch.cat([actions[:,:i],actions[:,i+1:]],dim=1)), 19).float().to(self.device),
                    ).gather(-1, actions[:,i].unsqueeze(-1)).squeeze(-1) for agent in range(self.n_agents)]
                    print(values[0].shape)
                    values = torch.stack(values,dim=1).to(self.device)
                    
                    
                    

                    # print(obs_tensor.shape, actions.shape, values.shape)
                # get the input actions 
                input_actions = actions 

                # start to store information
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(actions.detach().cpu().numpy())
                mb_log_probs.append(log_probs.detach().cpu().numpy())
                mb_dones.append(self.dones)
                mb_values.append(values.detach().cpu().numpy().squeeze())
                

                # start to excute the actions in the environment
                obs, rewards, dones, _ = self.envs.step(input_actions.cpu().numpy())
                # print(rewards.shape)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                episode_wise_rewards += rewards
                mb_rewards.append(rewards)

                # update dones
                # print(dones)
                self.dones = torch.tensor(dones)
                                
                

                # clear the observation
                for n, done in enumerate(dones):
                    if done:
                        self.obs[n] = self.obs[n] * 0
                        episodes_done[n] += 1
                self.obs = obs

                # process the rewards part -- display the rewards on the screen
                # print("rewards: ",rewards.shape)
                # rewards = torch.tensor(rewards, dtype=torch.float32)
                
                # for n, done in enumerate(dones):
                #     if done:
                #         states[n] = states[n] * 0
                #         # print("episodes_done",episodes_done)
                #         episodes_done[n] += 1
                #         self.episode_no += 1

                # print("rewards: ",rewards)
                episode_rewards += rewards
                
                masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in dones], dtype=torch.float32)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks

                # print("episode ",episode_rewards)
                # print("final ",final_rewards)

            # print(episode_rewards)
            # print(episodes_done, episode_wise_rewards)
            self.episodes_done += episodes_done.sum()
            episodes_done = torch.tensor(episodes_done, dtype=torch.float32)
            episode_wise_rewards = np.where(episodes_done > 0, episode_wise_rewards / episodes_done, episode_wise_rewards)
            # print(episode_wise_rewards)
            running_reward.append(episode_wise_rewards)
            # print(running_reward)
            # print(np.stack(running_reward).shape)

            # process the rollouts
            # print("before: ", len(mb_obs), mb_obs[0].shape, len(mb_rewards), mb_rewards[0].shape, len(mb_actions), mb_actions[0].shape, len(mb_dones), mb_dones[0].shape, len(mb_values), mb_values[0].shape)

            mb_obs = np.stack(mb_obs)
            mb_rewards = np.stack(mb_rewards)
            mb_actions = np.stack(mb_actions)
            mb_log_probs = np.stack(mb_log_probs)
            mb_dones = np.stack(mb_dones)
            mb_values = np.stack(mb_values)
            # print("after :", mb_obs.shape, mb_rewards.shape, mb_actions.shape, mb_dones.shape, mb_values.shape)

            # compute the last state value
            with torch.no_grad():
                obs_tensor = self._get_tensors(self.obs)
                last_values = [self.critics[agent](
                                                    obs_tensor.reshape(self.args.num_workers, self.n_agents, self.args.history, 4, 72, 96)[:,agent,:,:3].reshape(self.args.num_workers, -1, 72, 96),
                                                    F.one_hot(torch.cat([actions[:,:i],actions[:,i+1:]],dim=1)), 19).float().to(self.device),
                    ).gather(-1, actions[:,i].unsqueeze(-1)).squeeze(-1) for agent in range(self.n_agents)]
                last_values = torch.stack(last_values,dim=1)
                last_values = last_values.detach().cpu().numpy().squeeze()
                # mb_values = np.append(mb_values,last_values)
                
                last_values_agent = []
                for agent in range(self.n_agents):
                    last_value = [self.critics[agent](
                                                    obs_tensor.reshape(self.args.num_workers, self.n_agents, self.args.history, 4, 72, 96)[:,agent,:,:3].reshape(self.args.num_workers, -1, 72, 96),
                                                    F.one_hot(torch.cat(actions[:,:i],actions[:,i+1:]],dim=1)), 19).float().to(self.device)
                    last_values_agent.append(last_value)


            # print(mb_values.shape)

            # start to compute advantages...
            mb_returns = np.zeros_like(mb_rewards)
            mb_advs = np.zeros_like(mb_rewards)
            lastgaelam = 0
            for t in reversed(range(self.args.nsteps)):
                # print(t, args.nsteps)
                if t == self.args.nsteps - 1:
                    # print(self.dones.dtype, 1 - self.dones.int())
                    nextnonterminal = 1.0 - self.dones.numpy()
                    nextvalues = last_values
                else:
                    nextnonterminal = 1.0 - mb_dones[t + 1]
                    nextvalues = mb_values[t + 1]
                # print(mb_rewards[t].shape, nextvalues.shape, nextnonterminal.shape, mb_values[t].shape)
                delta = mb_rewards[t] + self.args.gamma * nextvalues * np.expand_dims(nextnonterminal,axis=-1) - mb_values[t]
                mb_advs[t] = lastgaelam = delta + self.args.gamma * self.args.tau * np.expand_dims(nextnonterminal,axis=-1) * lastgaelam
            mb_returns = mb_advs + mb_values

            # print("advs, returns :", mb_advs.shape, mb_returns.shape)
            # after compute the returns, let's process the rollouts
            mb_obs = mb_obs.swapaxes(0, 1).reshape(self.batch_ob_shape)
            mb_actions = mb_actions.swapaxes(0, 1).reshape(-1,self.n_agents)
            mb_returns = mb_returns.swapaxes(0, 1).reshape(-1,self.n_agents)
            mb_advs = mb_advs.swapaxes(0, 1).reshape(-1,self.n_agents)
            # print("after :", mb_obs.shape, mb_actions.shape, mb_returns.shape, mb_advs.shape)

            # before update the network, the old network will try to load the weights
            for a_i in range(self.n_agents):
                self.hard_update_target_network(self.critics[a_i], self.critics_target[a_i])
                self.hard_update_target_network(self.actors[a_i], self.actors_target[a_i])
                
            # start to update the network
            pl, vl, ent = self._update_network(mb_obs, mb_actions, mb_returns, mb_advs)

            # display the training information
            if update % self.args.display_interval == 0:
                self.logger.info('[{}] Update: {} / {}, Frames: {}, Rewards: {:.3f}, Min: {:.3f}, Max: {:.3f}, Running (100) : {:.3f}, PL: {},'\
                    'VL: {}, Ent: {}'.format(datetime.now(), update, num_updates, (update + 1)*self.args.nsteps*self.args.num_workers, \
                    final_rewards.mean().item(), final_rewards.min().item(), final_rewards.max().item(), np.mean(np.stack(running_reward)), pl, vl, ent))
                # save the model
                torch.save({"actor": ([self.actors[i].state_dict() for i in range(self.n_agents)] if not self.args.tie_actor_wts else self.actor.state_dict()), 
                        "critic": ([self.critics[i].state_dict() for i in range(self.n_agents)] if not self.args.tie_actor_wts else self.critic.state_dict()),
                        "actor_opt": ([self.actors_optimizer[i].state_dict() for i in range(self.n_agents)] if not self.args.tie_actor_wts else self.actors_optimizer.state_dict()),
                        "critic_opt": ([self.critics_optimizer[i].state_dict() for i in range(self.n_agents)] if not self.args.tie_actor_wts else self.critics_optimizer.state_dict()),
                        "num_update": update,
                        "running_reward": np.mean(np.stack(running_reward),axis=(0,1)),
                        }, self.model_path + '/' + self.args.env_name + "_" + str(self.no) + "_" + "agent.ckpt")

    # update the network
    def _update_network(self, obs, actions, returns, advantages):
        inds = np.arange(obs.shape[0])
        nbatch_train = self.args.batch_size
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
                mb_returns = torch.tensor(mb_returns, dtype=torch.float32)
                mb_advs = torch.tensor(mb_advs, dtype=torch.float32)
                # normalize adv
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)
                mb_actions = mb_actions.to(self.device)
                mb_returns = mb_returns.to(self.device)
                mb_advs = mb_advs.to(self.device)
                # start to get values
                pis = [self.actors[a_i](mb_obs[:,a_i]) for a_i in range(self.n_agents)]
                # print("logits: ", pis[0].max(dim=-1)[1])
                # pis = F.one_hot(torch.arange(10).unsqueeze(0))
                gumbel_pis = F.gumbel_softmax(logits=torch.stack(pis,dim=1), tau=1, hard=False, eps=1e-10, dim=-1).to(self.device)
                # print("gumbel softmax: ",pis[0].max(dim=-1)[1])
                # pis = F.softmax(self.add_gumbel(pis),dim=-1).long()

                # print(torch.stack(pis,dim=1).shape, self.batch_size, mb_obs.shape)

                mb_values = [self.critics[a_i](
                                                mb_obs.reshape(nbatch_train, self.n_agents, self.args.history, 4, 72, 96)[:,a_i,:,:3].reshape(nbatch_train, -1, 72, 96),
                                                gumbel_pis,
                                                ).squeeze() for a_i in range(self.n_agents)]
                mb_values = torch.stack(mb_values, dim=1)            

                # print(mb_returns.shape, mb_values.shape, mb_actions.shape)
                # start to calculate the value loss...
                value_loss = (mb_returns - mb_values).pow(2).mean(dim=0)
                # start to calculate the policy loss
                with torch.no_grad():
                    old_pis = [self.actors_target[a_i](mb_obs[:,a_i]) for a_i in range(self.n_agents)]
                    old_log_prob = [self.evaluate_actions(old_pis[a_i],mb_actions[:,a_i])[0] for a_i in range(self.n_agents)]
                    old_pis = torch.stack(old_pis,dim=1)
                    old_log_prob = torch.cat(old_log_prob,dim=1)
                    
                    # get the old log probs
                    # old_log_prob, _ = self.evaluate_actions(old_pis, mb_actions)
                    old_log_prob = old_log_prob.detach()
                # evaluate the current policy
                log_prob = []
                ent_loss = []

                for a_i in range(self.n_agents):
                    lprob, ent = self.evaluate_actions(pis[a_i], mb_actions[:,a_i])
                    log_prob.append(lprob)
                    ent_loss.append(ent)

                log_prob = torch.cat(log_prob,dim=1)
                ent_loss = torch.stack(ent_loss)

                # print(log_prob.shape, ent_loss.shape, old_log_prob.shape)

                prob_ratio = torch.exp(log_prob - old_log_prob)
                # surr1
                surr1 = prob_ratio * mb_advs
                surr2 = torch.clamp(prob_ratio, 1 - self.args.clip, 1 + self.args.clip) * mb_advs

                # print(surr1.shape, surr2.shape)
                policy_loss = -torch.min(surr1, surr2).mean(dim=0)
                # print("policy loss: ", policy_loss.shape, "value_loss: ", value_loss.shape, "ent_loss: ",ent_loss.shape)

                # final total loss
                total_loss = policy_loss + self.args.vloss_coef * value_loss - ent_loss * self.args.ent_coef

                if self.args.tie_actor_wts:
                    self.actors_optimizer.zero_grad()
                else:
                    for a_i in range(self.n_agents):
                        self.actors_optimizer[a_i].zero_grad()

                if self.args.tie_critic_wts:
                    self.critics_optimizer.zero_grad()
                else:
                    for a_i in range(self.n_agents):

                        self.critics_optimizer[a_i].zero_grad()

                for a_i in range(self.n_agents):
                    if a_i == self.n_agents - 1:
                        total_loss[a_i].backward()
                    else:
                        total_loss[a_i].backward(retain_graph=True)

                # clear the grad buffer
                if self.args.tie_critic_wts:
                    self.scale_shared_grads(self.critic)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm_critic)
                    self.critics_optimizer.step()
                else:
                    for a_i in range(self.n_agents):
                        torch.nn.utils.clip_grad_norm_(self.critics[a_i].parameters(), self.args.max_grad_norm_critic)
                        self.critics_optimizer[a_i].step()

                if self.args.tie_actor_wts:
                    self.scale_shared_grads(self.actor)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm_actor)
                    self.actors_optimizer.step()
                else:
                    for a_i in range(self.n_agents):
                        torch.nn.utils.clip_grad_norm_(self.actors[a_i].parameters(), self.args.max_grad_norm_actor)
                        self.actors_optimizer[a_i].step()

        return policy_loss.detach().cpu().numpy(), value_loss.detach().cpu().numpy(), ent_loss.detach().cpu().numpy()

    def scale_shared_grads(self, model):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1. / self.n_agents)

    # convert the numpy array to tensors
    def _get_tensors(self, obs):
        # print(obs.shape)
        obs_tensor = torch.tensor(np.transpose(obs, (0, 1, 4, 2, 3)), dtype=torch.float32)
        # obs_tensor = torch.tensor(obs, dtype=torch.float32)
        # decide if put the tensor on the GPU
        obs_tensor = obs_tensor.to(self.device)
        return obs_tensor

    # adjust the learning rate
    def _adjust_learning_rate(self, update, num_updates):
        lr_frac = 1 - (update / num_updates)
        adjust_lr = self.args.lr * lr_frac
        for param_group in self.optimizer.param_groups:
             param_group['lr'] = adjust_lr

    def select_actions(self, pi):
        actions = Categorical(logits=pi).sample()
        log_probs = Categorical(logits=pi).log_prob(actions).detach().cpu().numpy().squeeze()
        # return actions
        return actions.detach().cpu().numpy().squeeze(), log_probs

    # evaluate actions
    def evaluate_actions(self, pi, actions):
        cate_dist = Categorical(logits=pi)
        log_prob = cate_dist.log_prob(actions).unsqueeze(-1)
        entropy = cate_dist.entropy().mean()
        return log_prob, entropy

    def hard_update_target_network(self, local_network, target_network):
        #print("hard update")
        target_network.load_state_dict(local_network.state_dict())

if __name__ == '__main__':


    args = get_args()

    envs = SubprocVecEnv([(lambda _i=i: create_single_football_env(args)) for i in range(args.num_workers)], context=None)
    env = create_single_football_env(args)
    mappo_trainer = mappo_agent(envs, args, env)
    # mappo_trainer.test()
    mappo_trainer.learn()

    envs.close()

        
