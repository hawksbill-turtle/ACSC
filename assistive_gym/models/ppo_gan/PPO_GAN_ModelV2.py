from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
import torch
from torch import nn

class PPO_GAN_ModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        obs_space = obs_space.shape[0]
        act_space = act_space.shape[0]
        
        n_task = model_config['custom_model_config']['num_task']

        self.device = (torch.device("cuda")
                        if torch.cuda.is_available() else torch.device("cpu"))
        
        ## Logit and Gen
        layers = []
        layers.append(SlimFC(in_size=obs_space, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        layers.append(SlimFC(in_size=100, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        layers.append(SlimFC(in_size=100, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        layers.append(SlimFC(in_size=100, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        self.logit_hidden_layers = nn.Sequential(*layers)

        self.logit_fn = SlimFC(in_size=100, out_size=num_outputs, initializer=normc_initializer(0.01), activation_fn=None)

        self.label_emb_logit = nn.Embedding(n_task, 100)

        layers = []
        layers.append(SlimFC(in_size=100, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        layers.append(SlimFC(in_size=100, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        layers.append(SlimFC(in_size=100, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        layers.append(SlimFC(in_size=100, out_size=obs_space, initializer=normc_initializer(1.0), activation_fn='tanh'))
        self.logit_gen_layers = nn.Sequential(*layers)
        
        
        ## Value and Gen
        layers = []
        layers.append(SlimFC(in_size=obs_space, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        layers.append(SlimFC(in_size=100, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        layers.append(SlimFC(in_size=100, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        layers.append(SlimFC(in_size=100, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        self.value_hidden_layers = nn.Sequential(*layers)
        
        self.value_fn = SlimFC(in_size=100, out_size=1, initializer=normc_initializer(0.01), activation_fn=None)

        self.label_emb_value = nn.Embedding(n_task, 100)

        layers = []
        layers.append(SlimFC(in_size=100, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        layers.append(SlimFC(in_size=100, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        layers.append(SlimFC(in_size=100, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        layers.append(SlimFC(in_size=100, out_size=obs_space, initializer=normc_initializer(1.0), activation_fn='tanh'))
        self.value_gen_layers = nn.Sequential(*layers)


        ## Dis
        layers = []
        layers.append(SlimFC(in_size=obs_space, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        layers.append(SlimFC(in_size=100, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        layers.append(SlimFC(in_size=100, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        layers.append(SlimFC(in_size=100, out_size=100, initializer=normc_initializer(1.0), activation_fn='tanh'))
        self.dis_hidden_layers = nn.Sequential(*layers)

        self.adv_layer = nn.Sequential(SlimFC(in_size=100, out_size=1, initializer=normc_initializer(1.0), activation_fn='sigmoid'))
        self.aux_layer = nn.Sequential(SlimFC(in_size=100, out_size=n_task, initializer=normc_initializer(1.0), activation_fn=None))


    def forward(self, input_dict, state, seq_lens):
        self.logit_latent = self.logit_hidden_layers(input_dict['obs'])
        self.value_latent = self.value_hidden_layers(input_dict['obs'])
        self._value_out = self.value_fn(self.value_latent).flatten()

        return self.logit_fn(self.logit_latent), state

    def value_function(self):
        return self._value_out
    
    def generate_function(self, num_task, task):
        task = task.to(self.device)
        logit_latent = torch.mul(torch.cat([self.logit_latent]*num_task, dim=0), self.label_emb_logit(task))
        value_latent = torch.mul(torch.cat([self.value_latent]*num_task, dim=0), self.label_emb_value(task))

        l_gen = self.logit_gen_layers(logit_latent)
        v_gen = self.value_gen_layers(value_latent)

        return l_gen, v_gen

    def discriminate_function(self, state):
        out = self.dis_hidden_layers(state)
        _validity = self.adv_layer(out)
        _label = self.aux_layer(out)

        return _validity, _label




