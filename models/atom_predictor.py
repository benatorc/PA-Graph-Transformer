import torch
import torch.nn as nn

from models.mol_conv_net import MolConvNet
from models.mol_transformer import MolTransformer
import pdb


class AtomPredictor(nn.Module):
    def __init__(self, args, n_classes=1):
        super(AtomPredictor, self).__init__()
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

    def forward(self, mol_graph, pair_idx, stats_tracker, output_attn=False):
        attn_list = None
        if self.args.model_type == 'transformer':
            atom_h, attn_list = self.model(mol_graph, stats_tracker)
        else:
            atom_h = self.model(mol_graph, stats_tracker)

        all_pairs_h = []
        scope = mol_graph.scope
        for mol_idx, (st, le) in enumerate(scope):
            cur_atom_h = atom_h.narrow(0, st, le)
            cur_pair_idx = torch.tensor(pair_idx[mol_idx],
                                        device=self.args.device)
            n_pairs = cur_pair_idx.size()[0]
            cur_pair_idx = cur_pair_idx.view(-1)

            selected_atom_h = torch.index_select(
                input=cur_atom_h, dim=0, index=cur_pair_idx,)
            n_feats = selected_atom_h.size()[1]

            selected_atom_h = selected_atom_h.view([n_pairs, 2, n_feats])
            atom_pair_h = torch.mean(selected_atom_h, 1)
            all_pairs_h.append(atom_pair_h)

        all_pairs_h = torch.cat(all_pairs_h, dim=0)
        all_pairs_h = nn.ReLU()(self.W_p_h(all_pairs_h))
        all_pairs_o = self.W_p_o(all_pairs_h)

        return all_pairs_o
