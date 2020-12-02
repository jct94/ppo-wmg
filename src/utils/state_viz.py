"""
This file provides utils for vizualizing the state of relational memory
architectures,
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from agents.networks.shared.transformer import Transformer, TorchTransformer

def get_attention_weight_matrix(network, input, memos):
    # aw shape: Bx(L*H)xNxN, L number of layers, H number of heads,
    # N number of tokens
    # what about batch size ?
    tfm_input = network.embed_obs_and_memos(input, memos)

    with torch.no_grad():
        if isinstance(network.tfm, Transformer):
            attn_mats = []
            for layer in network.tfm.layers:
                self_attn_layer = layer.attention
                # compute attention score
                queries = self_attn_layer.query(tfm_input)
                keys = self_attn_layer.key(tfm_input)
                split_q = self_attn_layer.split_heads_apart(queries)
                split_k = self_attn_layer.split_heads_apart(keys)
                split_k_transposed = split_k.transpose(-1, -2)

                attn = torch.matmul(split_q, split_k_transposed)

                attn = attn * self_attn_layer.dot_product_scale
                attn = nn.Softmax(-1)(attn)

                attn_mats.append(attn)

            attn_mats = torch.cat(attn_mats, -3)

        elif isinstance(network.tfm, TorchTransformer):
            attn_mats = []
            # TODO(laetitia): complete in case of pytorch transformer

    return attn_mats

def get_state_cos_sim(state_vectors):
    # get cosine similarity between state vectors
    # state_vectors is a list or matrix of vectors
    if state_vectors == []:
        return torch.zeros(0, 0)
    with torch.no_grad():
        if isinstance(state_vectors, list):
            state_vectors = torch.cat(state_vectors, 0)
        num_vecs = len(state_vectors)
        vec_dim = state_vectors.shape[-1]
        vectors_horizontal = state_vectors.expand(
            num_vecs,
            num_vecs,
            vec_dim
        )
        vectors_vertical = vectors_horizontal.transpose(0, 1)
        horizontal_norm = vectors_horizontal.pow(2).sum(-1).pow(0.5)
        vertical_norm = vectors_vertical.pow(2).sum(-1).pow(0.5)
        normprod = horizontal_norm * vertical_norm
        prod = (vectors_horizontal * vectors_vertical).sum(-1)

        sim_matrix = prod / normprod

    return sim_matrix

# Window class for vizualizing stuff during  a run of the agent
# Based on minigrid's Window class

class MatrixWindow():
    """
    Holds two vizualization screens, one for the attention weight matrix, one for the
    memo cosine similarity.
    """
    # TODO: what happens when the number of memos is unlimited ?
    # TODO: how to handle variable numbers of objects ?
    def __init__(self,
                 num_attn_weight,
                 num_memos,
                 aw_layer=0,
                 aw_head=0,
                 title="attention weights and memo similarity"):
        self.fig = None

        self.num_attn_weight = num_attn_weight
        self.num_memos = num_memos

        # the head index and the layer index of the attention weights to
        # visualize.
        self.aw_layer = aw_layer
        self.aw_head = aw_head

        self.aw_obj = None # for attention weights
        self.cos_sim_obj = None # for cosine sim of memos

        self.fig, self.axs = plt.subplots(2)

        self.fig.canvas.set_window_title(title)

        for ax in self.axs:
            ax.set_xticks([], [])
            ax.set_yticks([], [])

        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)

    def show_mats(self, network, obs, memos):
        """
        Shows the matrices or update the matrices being shown. The inputs are the
        transformer of interest, the transformer inputs, and the memos.
        """
        awm = get_attention_weight_matrix(network, obs, memos)

        num_layers = len(network.tfm.layers)
        idx = num_layers * self.aw_layer + self.aw_head

        matrix = awm[0, idx]

        # the attention-weight matrix can be too big or too small
        N = len(matrix)
        if N > self.num_attn_weight:
            matrix = matrix[:self.num_attn_weight, :self.num_attn_weight]
        elif N < self.num_attn_weight:
            zero_matrix = torch.zeros(self.num_attn_weight, self.num_attn_weight)
            zero_matrix[:N, :N] = matrix
            matrix = zero_matrix

        sim = get_state_cos_sim(memos)

        # same for the similarity matrix
        M = len(sim)
        if M > self.num_memos:
            sim = sim[:self.num_memos, :self.num_memos]
        elif M < self.num_memos:
            zero_sim = torch.zeros(self.num_memos, self.num_memos)
            zero_sim[:M, :M] = sim
            sim = zero_sim

        if self.aw_obj is None or self.cos_sim_obj is None:
            self.aw_obj = self.axs[0].matshow(matrix)
            self.cos_sim_obj = self.axs[1].matshow(sim)

        self.aw_obj.set_data(matrix)
        self.cos_sim_obj.set_data(sim)
        self.fig.canvas.draw()

        plt.pause(0.001)

    def set_caption(self, text):
        """
        Set/update the caption text below the image
        """

        plt.xlabel(text)

    def reg_key_handler(self, key_handler):
        """
        Register a keyboard event handler
        """

        # Keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', key_handler)

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def close(self):
        """
        Close the window
        """

        plt.close()
