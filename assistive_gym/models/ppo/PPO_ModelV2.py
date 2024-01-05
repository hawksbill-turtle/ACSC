from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from torch import nn

class PPO_ModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        obs_space = obs_space.shape[0]
        act_space = act_space.shape[0]
        
        layers = []
        layers.append(SlimFC(in_size=obs_space, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        layers.append(SlimFC(in_size=100, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        self.logit_hidden_layers = nn.Sequential(*layers)
        
        self.logit_fn = SlimFC(in_size=100, out_size=num_outputs, initializer=normc_initializer(0.01), activation_fn=None)

        layers = []
        layers.append(SlimFC(in_size=obs_space, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        layers.append(SlimFC(in_size=100, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        self.value_hidden_layers = nn.Sequential(*layers)
        
        self.value_fn = SlimFC(in_size=100, out_size=1, initializer=normc_initializer(0.01), activation_fn=None)

    def forward(self, input_dict, state, seq_lens):
        logit_latent = self.logit_hidden_layers(input_dict['obs'])
        value_latent = self.value_hidden_layers(input_dict['obs'])
        self._value_out = self.value_fn(value_latent).flatten()
        return self.logit_fn(logit_latent), state

    def value_function(self):
        return self._value_out
