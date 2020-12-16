import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This module defines the transformer models.
"""

tensor = torch.FloatTensor

# utility models

class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()

        layers = []
        lin = layer_sizes.pop(0)

        for i, ls in enumerate(layer_sizes):
            layers.append(nn.Linear(lin, ls))
            if i < len(layer_sizes) - 1:
                layers.append(nn.ReLU())
            lin = ls

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLP_TwoLayers_Norm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# main model

class Transformer(nn.Module):
    """
    Wrapper around the pytorch Transformer module.
    """
    def __init__(self, n_heads, d_model, num_layers, dim_feedforward,
                 dropout=0., query_token=0):
        super().__init__()

        # this assumes the core vector in in zeroth position
        self.query_token = query_token

        tfm_layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward,
            dropout
        )
        norm = nn.LayerNorm(d_model)
        self.tfm = nn.TransformerEncoder(tfm_layer, num_layers, norm)
        # TODO: no custom weight init

    def forward(self, input):
        # input has size [seq, batch, feature_dim]
        # we output the transformer value corresponding to the fixed query index
        return self.tfm(input)[self.query_token]


class SceneTransformer(nn.Module):
    """
    Contains a transformer layer
    for relating the different objects, a policy head that is a linear
    projection of the 0th output of the transformer, and a value head that
    is a linear projection of the policy head.
    """
    def __init__(self, obs_features, n_actions, core_size, factor_size):

        super().__init__()

        self.discrete = True

        self.query_token = 0
        self.max_objects = 9
        self.core_size = core_size
        self.factor_size = factor_size

        # observation encodings
        self.obs_features = obs_features

        self.core_embedding = nn.Linear(core_size, obs_features)
        self.factor_embedding = nn.Linear(factor_size, obs_features)

        tfm_layer = nn.TransformerEncoderLayer(
            obs_features,
            1,
            64,
            0.
        )
        norm = nn.LayerNorm(obs_features)
        self.tfm = nn.TransformerEncoder(tfm_layer, 1, norm)
        self.mlp = MLP([obs_features, 64, 64])
        self.policy_proj = nn.Linear(64, n_actions)
        self.value_proj = nn.Linear(64, 1)

    def encode_observations(self, obs_list):
        F = obs_list[0].shape[0] # 63

        n = len(obs_list)
        m = self.max_objects - n
        if m < 0:
            print(f"Dropped {- m} objects !")
            obs_list = obs_list[:self.max_objects]

        # encode core

        vecs = [tensor(obs_list[0]).unsqueeze(0)]
        # vecs = [self.core_embedding(tensor(obs_list[0])).unsqueeze(0)]

        # encode factors, if any
        if obs_list[1:]:
            for obs in obs_list[1:]:
                f = obs.shape[0] # f < F
                v = torch.cat([tensor(obs), torch.zeros(F - f)]).unsqueeze(0)
                vecs.append(v)
                # vecs.append(self.factor_embedding(tensor(obs)).unsqueeze(0))

        # pad with zero-vectors if necessary
        if m > 0:
            vecs.append(torch.zeros(m, F))

        obs_tensor = torch.cat(vecs, 0)
        return obs_tensor

    def forward(self, seq_input):
        # modified version of the forward
        # seq_input :: [B, seq, 63]
        seq_input = seq_input.transpose(0, 1)
        # cores and factors have different encodings
        cores = seq_input[0:1]
        factors = seq_input[1:, ..., :self.factor_size]

        cores = self.core_embedding(cores)
        factors = self.factor_embedding(factors)

        tfm_input = torch.cat([cores, factors], 0)
        tfm_out = self.tfm(tfm_input)[self.query_token]
        out = F.relu(self.mlp(tfm_out))
        probs = F.softmax(self.policy_proj(out))

        return probs

    def get_value(self, seq_input):
        # modified version of the get_value pass
        # seq_input :: [B, seq, 63]
        seq_input = seq_input.transpose(0, 1)
        # cores and factors have different encodings
        cores = seq_input[0:1]
        factors = seq_input[1:, ..., :self.factor_size]

        cores = self.core_embedding(cores)
        factors = self.factor_embedding(factors)

        tfm_input = torch.cat([cores, factors], 0)
        tfm_out = self.tfm(tfm_input)[self.query_token]
        out = F.relu(self.mlp(tfm_out))
        value = self.value_proj(out)

        return value

    # utilities

    def calc_kl(self, p, q, get_mean=True): # TODO: does not return a list
        '''
        Calculates E KL(p||q):
        E[sum p(x) log(p(x)/q(x))]
        Inputs:
        - p, first probability distribution (NUM_SAMPLES, NUM_ACTIONS)
        - q, second probability distribution (NUM_SAMPLES, NUM_ACTIONS)
        Returns:
        - Empirical KL from p to q
        '''
        p, q = p.squeeze(), q.squeeze()
        # assert shape_equal_cmp(p, q)
        kl = (p * (torch.log(p) - torch.log(q))).sum(-1)
        return kl

    def entropies(self, p):
        '''
        p is probs of shape (batch_size, action_space). return mean entropy
        across the batch of states
        '''
        entropies = (p * torch.log(p)).sum(dim=1)
        return entropies

    def get_loglikelihood(self, p, actions):
        '''
        Inputs:
        - p, batch of probability tensors
        - actions, the actions taken
        '''
        try:
            if len(actions)>1:
                actions = actions[:,0]
            dist = torch.distributions.categorical.Categorical(p)
            return dist.log_prob(actions)
        except Exception as e:
            raise ValueError("Numerical error")

    def sample(self, probs):
        '''
        given probs, return: actions sampled from P(.|s_i), and their
        probabilities
        - s: (batch_size, state_dim)
        Returns actions:
        - actions: shape (batch_size,)
        '''
        dist = torch.distributions.categorical.Categorical(probs)
        actions = dist.sample()
        return actions.long()

# class TransformerDiscPolicy(DiscPolicy):
#     """
#     Prefixes the Fully-connected DiscPolicy with the Transformer module.
#     """
#     def __init__(self, state_dim, action_dim, init, # DiscPolicy args
#                  n_heads, d_model, num_layers, dim_feedforward, # transformer args
#                  hidden_sizes=HIDDEN_SIZES, share_weights=False, # DiscPolicy kwargs
#                  dropout=0., query_token=0): # transformer kwargs
#
#         super().__init__(
#             state_dim=state_dim,
#             action_dim=action_dim,
#             init=init,
#             hidden_sizes=hidden_sizes,
#             time_in_state=False,
#             share_weights=True
#         )
#
#         self.transformer = Transformer(
#             n_heads=n_heads,
#             d_model=d_model,
#             num_layers=num_layers,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             query_token=query_token
#         )
#
#     def forward(self, input):
#         tfm_out = self.transformer(input)
#         return super().forward(tfm_out)
#
#     def get_value(self, input):
#         tfm_out = self.transformer(input)
#         return super().get_value(tfm_out)
#
# class TransformerDiscPolicyv2(TransformerDiscPolicy):
#     """
#     This version takes in directly the states as output from the BabyAI custom env.
#     """
#     def __init__(self, state_dim, action_dim, init, # DiscPolicy args
#                  n_heads, d_model, num_layers, dim_feedforward, # transformer args
#                  hidden_sizes=HIDDEN_SIZES, share_weights=False, # DiscPolicy kwargs
#                  dropout=0., query_token=0): # transformer kwargs
#
#         super().__init__(state_dim, action_dim, init,
#                  n_heads, d_model, num_layers, dim_feedforward,
#                  hidden_sizes=hidden_sizes, share_weights=share_weights,
#                  dropout=dropout, query_token=query_token)
#
#         self.core_encoder = nn.Linear(63, d_model)
#         self.factor_encoder = nn.Linear(23, d_model)
#
#     def encode_observation(self, obs_list):
#         core_embedded = self.core_embedding(torch.FloatTensor(obs_list[0]))
#         core_embedded = core_embedded.unsqueeze(0)
#         if obs_list[1:]:
#             factors_embedded = self.factor_embedding(cpu_tensorize(obs_list[1:]))
#         else:
#             factors_embedded = ch.zeros(0, self.model_size)
#
#         embedded_obs = ch.cat([core_embedded, factors_embedded], 0).unsqueeze(1)
#
#     def forward(self, obs_list):
#         embedded_obs = self.encode_observation(obs_list)
#         return super().forward(embedded_obs).squeeze(0)
#
#     def get_value(self, obs_list):
#         embedded_obs = self.encode_observation(obs_list)
#         return super().get_value(embedded_obs).squeeze(0)
#
# ## Retrieving networks
# # Make sure to add newly created networks to these dictionaries!
#
# POLICY_NETS = {
#     "DiscPolicy": DiscPolicy,
#     "CtsPolicy": CtsPolicy
# }
#
# VALUE_NETS = {
#     "ValueNet": ValueDenseNet,
# }
#
# def policy_net_with_name(name):
#     return POLICY_NETS[name]
#
# def value_net_with_name(name):
#     return VALUE_NETS[name]
#
