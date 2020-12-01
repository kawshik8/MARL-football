import torch
from torch import nn
from torch.nn import functional as F

"""
this network is modified for the google football

"""

class backbone(nn.Module):
    def __init__(self, history=4, nhidden=512, nchannels=4):
        super(backbone, self).__init__()
        self.conv1 = nn.Conv2d(history*nchannels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.fc1 = nn.Linear(32 * 5 * 8, nhidden)        
        # start to do the init...
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
        # init the bias...
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        nn.init.constant_(self.fc1.bias.data, 0)
        
    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = x.view(-1, 32 * 5 * 8)
        x = F.relu(self.fc1(x))

        return x

# in the initial, just the nature CNN
class cnn_net(nn.Module):
    def __init__(self, history, num_actions, nhidden=512):
        super(cnn_net, self).__init__()
        self.cnn_layer = backbone(history)
        self.critic = nn.Linear(nhidden, 1)
        self.actor = nn.Linear(nhidden, num_actions)

        # init the linear layer..
        nn.init.orthogonal_(self.critic.weight.data)
        nn.init.constant_(self.critic.bias.data, 0)
        # init the policy layer...
        nn.init.orthogonal_(self.actor.weight.data, gain=0.01)
        nn.init.constant_(self.actor.bias.data, 0)

    def forward(self, inputs):
        x = self.cnn_layer(inputs / 255.0)
        value = self.critic(x)
        pi = self.actor(x)
        return value, pi
        
class fc_net(nn.Module):
    def __init__(self, history, num_actions, nagents = 1, global_state = True):
        super(fc_net, self).__init__()
        self.fc1 = nn.Linear(115*history, 256)        
        self.fc2 = nn.Linear(256, 32)        
        self.critic = nn.Linear(32, 1)
        self.actor = nn.Linear(32, num_actions)

        nn.init.orthogonal_(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc1.bias.data, 0)
        nn.init.constant_(self.fc2.bias.data, 0)
        
        # init the linear layer..
        nn.init.orthogonal_(self.critic.weight.data)
        nn.init.constant_(self.critic.bias.data, 0)
        # init the policy layer...
        nn.init.orthogonal_(self.actor.weight.data, gain=0.01)
        nn.init.constant_(self.actor.bias.data, 0)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs / 255.0))
        x = F.relu(self.fc2(x))
        
        value = self.critic(x)
        pi = self.actor(x)
        return value, pi

class Critic(nn.Module):
    def __init__(self, history, dim_action=1, n_agents=1, nhidden=512, nactions = 1, global_state_net=None):
        super(Critic, self).__init__()
        self.n_agents = n_agents

        if global_state_net is None:
            self.state_net = backbone(history)
        else:
            self.state_net = global_state_net

        self.dim_action = dim_action
        if self.dim_action > 1:
            self.action_embedding = nn.Linear(nactions, dim_action)

        self.FC1 = nn.Linear(nhidden + n_agents * dim_action, 256)
        self.FC2 = nn.Linear(256, 128)
        self.FC3 = nn.Linear(128, 1)

    # obs: batch_size * obs_dim
    def forward(self, state, actions):

        # print("state: ", state.shape)
        state = self.state_net(state / 255.0)

        # print("state: ", state.shape)
        if self.dim_action > 1:
            # print(actions.shape, self.action_embedding)
            actions = self.action_embedding(actions)
            # print("actions befor .e reshaping: ",actions.shape)
            actions = actions.view(actions.size(0), self.n_agents * self.dim_action)

        # print("actions after reshaping: ",actions.shape)
        combined = torch.cat([state, actions], 1)
        # print(combined.shape, self.dim_action)
        # print(self.FC1)
        result = F.relu(self.FC1(combined))

        return self.FC3(F.relu(self.FC2(result)))


class Actor(nn.Module):
    def __init__(self, history, dim_action, nhidden=512, global_state_net=None):
        super(Actor, self).__init__()
        
        if global_state_net is None:
            self.state_net = backbone(history)
        else:
            self.state_net = global_state_net

        self.FC1 = nn.Linear(nhidden, 256)
        self.FC2 = nn.Linear(256, dim_action)

    # action output between -2 and 2
    def forward(self, state):

        state = self.state_net(state / 255.0)

        result = F.relu(self.FC1(state))
        result = self.FC2(result)

        return result

    def sample(self, state):
        out = self.forward(state)



if __name__=='__main__':

    history = 4
    nagents = 2
    batch_size = 16
    action_dim = 32
    nactions = 24
    
    state_net = backbone(history)
    actor = Actor(history, nactions, global_state_net = state_net)
    critic = Critic(history, action_dim, nagents, nactions = nactions, global_state_net = state_net)
    actors = [actor for i in range(nagents)]
    critic = [critic for i in range(nagents)]

    a = torch.rand(batch_size,history*4,72,96)
    print(a.shape)

    actions = [actors[i](a) for i in range(len(actors))]

    for i in actions:
        print(i.shape)

    actions = torch.stack(actions,dim=1)
    print(actions.shape)

    actions = torch.max(actions,dim=-1)[1]
    print(actions.shape)

    values = [critic[i](a, actions) for i in range(len(actors))]
    values = torch.stack(values,dim=1).squeeze()
    print(values.shape)



