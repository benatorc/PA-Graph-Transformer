import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, args):
        super(AttentionLayer, self).__init__()
        self.args = args
        hidden_size = args.hidden_size

        self.W_attn_h = nn.Linear(hidden_size, hidden_size)
        self.W_attn_o = nn.Linear(hidden_size, 1)

    def forward(self, atom_h, scope):
        hidden_size = self.args.hidden_size

        atom_attn_h = []
        for (st, le) in scope:
            cur_atom_h = atom_h.narrow(0, st, le)
            cur_atom_h_1 = cur_atom_h.reshape([1, -1, hidden_size])
            cur_atom_h_2 = cur_atom_h.reshape([-1, 1, hidden_size])

            # all pairs: [# atoms, # atoms, hidden_size]
            atom_pairs_input = cur_atom_h_1 + cur_atom_h_2
            atom_pairs_h = nn.ReLU()(self.W_attn_h(atom_pairs_input))

            # Different attention scores other than Sigmoid
            # [# atoms, # atoms, 1]
            atom_pairs_scores = nn.Sigmoid()(self.W_attn_o(atom_pairs_h))

            # [# atoms, # atoms, hidden_size]
            cur_atom_attn_h = atom_pairs_scores * cur_atom_h
            # [# atoms, hidden_size]
            cur_atom_attn_h = torch.sum(cur_atom_attn_h, dim=1)
            atom_attn_h.append(cur_atom_attn_h)
        atom_attn_h = torch.cat(atom_attn_h, dim=0)
        return atom_attn_h
