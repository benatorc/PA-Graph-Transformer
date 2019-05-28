import torch
import torch.nn as nn

from modules.conv_layer import GraphConv
from modules.attention import AttentionLayer
import pdb


class MolConvNet(nn.Module):
    def __init__(self, args, use_attn=False):
        super(MolConvNet, self).__init__()
        self.args = args
        self.use_attn = use_attn

        self.conv_layer = GraphConv(args)
        self.output_size = args.hidden_size

        if self.use_attn:
            self.attn_layer = AttentionLayer(args)
            self.output_size += args.hidden_size

    def forward(self, mol_graph, stats_tracker=None):
        graph_inputs, scope = mol_graph.get_graph_inputs()
        atom_h = self.conv_layer(graph_inputs)

        attn_context = None
        if self.use_attn:
            attn_context = self.attn_layer(atom_h, scope)
        if attn_context is not None:
            atom_h = torch.cat([atom_h, attn_context], dim=1)

        return atom_h
