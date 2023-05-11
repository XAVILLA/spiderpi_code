import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from actor_critic import ActorCritic, get_activation
from utils_rsl import unpad_trajectories
import pdb

class ActorCriticRecurrentCNNs2r(ActorCritic):
    is_recurrent = True
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        rnn_type='lstm',
                        rnn_hidden_size=256,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)

        super().__init__(num_actor_obs=rnn_hidden_size,
                         num_critic_obs=rnn_hidden_size,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std)

        activation = get_activation(activation)

        self.memory_a = MemoryCNNS2R(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size,activation=activation)
        self.memory_c = MemoryCNNS2R(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size,activation=activation)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))
    
    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class MemoryCNNS2R(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256,activation=get_activation('elu')):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        rnn_input_size = 18+3+10
        self.rnn = rnn_cls(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=num_layers) #TODO: need to change input size
        self.hidden_states = None
        self.activation = activation
        self.depth_image_size = (24, 32)
        self.depth_image_flat_dim = self.depth_image_size[0]*self.depth_image_size[1]

        #32*24 depth image
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8, 8, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.cnn_list = [
            self.conv1, 
            self.conv2, 
            self.pool1,
            self.conv3,
            self.pool2
            ]
        self.fc1 = nn.Linear(8 * 6 * 4, 10) 
        self.linear_list = [
            self.fc1
        ]
    
    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        # pdb.set_trace()
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            split_input = input.split((21,self.depth_image_flat_dim), dim=2) #TODO need to change according to depth/heightmap
            #TODO: CNN for height map
            step_num = split_input[1].shape[0]
            env_num = split_input[1].shape[1]
            x = split_input[1]
            x = x.reshape((step_num*env_num, 1, self.depth_image_size[0], self.depth_image_size[1]))
            for net in self.cnn_list:
                x = net(x)
            x = torch.flatten(x, 1)
            for net in self.linear_list:
                x = net(x)
            x = x.reshape((step_num, env_num, 10))
            input = torch.cat((split_input[0], x), dim=2)
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            split_input = input.split((21,self.depth_image_flat_dim), dim=1) #TODO need to change according to depth/heightmap
            #TODO: CNN for height map
            env_num = split_input[1].shape[0]
            x = split_input[1]
            x = x.reshape((env_num, 1, self.depth_image_size[0], self.depth_image_size[1]))
            for net in self.cnn_list:
                x = net(x)
            x = torch.flatten(x, 1)
            for net in self.linear_list:
                x = net(x)
            x = x.reshape((env_num, 10))
            input = torch.cat((split_input[0], x), dim=1)
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
            # print(type(input))
            # print(out.size())
        # print('detach')
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0