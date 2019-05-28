import torch
import torch.nn as nn

from models.mol_conv_net import MolConvNet
from models.mol_transformer import MolTransformer
import pdb


class PropPredictor(nn.Module):
    def __init__(self, args, n_classes=1):
        super(PropPredictor, self).__init__()
        self.args = args
        hidden_size = args.hidden_size

        model = None
        if args.model_type == 'conv_net':
            model = MolConvNet(args, use_attn=False)
        elif args.model_type == 'conv_net_attn':
            model = MolConvNet(args, use_attn=True)
        elif args.model_type == 'transformer':
            model = MolTransformer(args)
        else:
            assert(False)
        self.model = model

        self.W_p_h = nn.Linear(model.output_size, hidden_size)  # Prediction
        self.W_p_o = nn.Linear(hidden_size, n_classes)

    def aggregate_atom_h(self, atom_h, scope):
        mol_h = []
        for (st, le) in scope:
            cur_atom_h = atom_h.narrow(0, st, le)

            if self.args.agg_func == 'sum':
                mol_h.append(cur_atom_h.sum(dim=0))
            elif self.args.agg_func == 'mean':
                mol_h.append(cur_atom_h.mean(dim=0))
            else:
                assert(False)
        mol_h = torch.stack(mol_h, dim=0)
        return mol_h

    def forward(self, mol_graph, stats_tracker, output_attn=False):
        attn_list = None
        if self.args.model_type == 'transformer':
            atom_h, attn_list = self.model(mol_graph, stats_tracker)
        else:
            atom_h = self.model(mol_graph, stats_tracker)

        scope = mol_graph.scope
        mol_h = self.aggregate_atom_h(atom_h, scope)
        mol_h = nn.ReLU()(self.W_p_h(mol_h))
        mol_o = self.W_p_o(mol_h)

        if not output_attn:
            return mol_o
        else:
            return mol_o, attn_list
