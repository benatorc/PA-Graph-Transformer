import torch
import torch.nn as nn

import graph.mol_features as mol_features
import pdb


class GraphConv(nn.Module):
    def __init__(self, args):
        """Creates graph conv layers for molecular graphs."""
        super(GraphConv, self).__init__()
        self.args = args
        hidden_size = args.hidden_size

        self.n_atom_feats = mol_features.N_ATOM_FEATS
        self.n_bond_feats = mol_features.N_BOND_FEATS

        # Weights for the message passing network
        self.W_message_i = nn.Linear(self.n_atom_feats + self.n_bond_feats,
                                     hidden_size, bias=False,)
        if args.no_share:
            self.W_message_h = nn.ModuleList(
                [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(args.depth - 1)])
        else:
            self.W_message_h = nn.Linear(hidden_size, hidden_size, bias=False,)
        self.W_message_o = nn.Linear(self.n_atom_feats + hidden_size,
                                     hidden_size)

        self.dropout = nn.Dropout(args.dropout)

    def index_select_nei(self, input, dim, index):
        # Reshape index because index_select expects a 1-D tensor. Reshape the
        # output afterwards.
        target = torch.index_select(
            input=input,
            dim=0,
            index=index.view(-1)
        )
        return target.view(index.size() + input.size()[1:])

    def forward(self, graph_inputs):
        fatoms, fbonds, agraph, bgraph = graph_inputs

        # nei_input_h is size [# bonds, hidden_size]
        nei_input_h = self.W_message_i(fbonds)
        # message_h is size [# bonds, hidden_size]
        message_h = nn.ReLU()(nei_input_h)

        for i in range(self.args.depth - 1):
            # nei_message_h is [# bonds, # max neighbors, hidden_size]
            nei_message_h = self.index_select_nei(
                input=message_h,
                dim=0,
                index=bgraph)

            # Sum over the nieghbors, now [# bonds, hidden_size]
            nei_message_h = nei_message_h.sum(dim=1)
            if self.args.no_share:
                nei_message_h = self.W_message_h[i](nei_message_h)
            else:
                nei_message_h = self.W_message_h(nei_message_h)  # Shared weights

            message_h = nn.ReLU()(nei_input_h + nei_message_h)

        # Collect the neighbor messages for atom aggregation
        nei_message_h = self.index_select_nei(
            input=message_h,
            dim=0,
            index=agraph,
        )
        # Aggregate the messages
        nei_message_h = nei_message_h.sum(dim=1)
        atom_input = torch.cat([fatoms, nei_message_h], dim=1)
        atom_input = self.dropout(atom_input)

        atom_h = nn.ReLU()(self.W_message_o(atom_input))
        return atom_h
