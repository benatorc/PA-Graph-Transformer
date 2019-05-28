import torch
import torch.nn as nn
import numpy as np

import graph.mol_features as mol_features
from utils import model_utils, path_utils
import pdb


class MolTransformer(nn.Module):
    def __init__(self, args):
        super(MolTransformer, self).__init__()
        self.args = args
        hidden_size = args.hidden_size
        n_heads, d_k = args.n_heads, args.d_k

        n_atom_feats = mol_features.N_ATOM_FEATS
        n_path_feats = path_utils.get_num_path_features(args)

        # W_atom_i: input atom embedding
        self.W_atom_i = nn.Linear(n_atom_feats, n_heads * d_k, bias=False)

        # W_attn_h: compute atom attention score
        # W_message_h: compute the new atom embeddings
        n_score_feats = 2 * d_k + n_path_feats
        if args.no_share:
            self.W_attn_h = nn.ModuleList([
                nn.Linear(n_score_feats, d_k) for _ in range(args.depth - 1)])
            self.W_attn_o = nn.ModuleList([
                nn.Linear(d_k, 1) for _ in range(args.depth - 1)])
            self.W_message_h = nn.ModuleList([
                nn.Linear(n_score_feats, d_k) for _ in range(args.depth - 1)])
        else:
            self.W_attn_h = nn.Linear(n_score_feats, d_k)
            self.W_attn_o = nn.Linear(d_k, 1)
            self.W_message_h = nn.Linear(n_score_feats, d_k)

        # W_atom_o: the output embedding
        self.W_atom_o = nn.Linear(n_atom_feats + n_heads * d_k, hidden_size)
        self.dropout = nn.Dropout(args.dropout)

        self.output_size = hidden_size

    def get_attn_input(self, atom_h, path_input, max_atoms):
        # attn_input is concatentation of atom pair embeddings and path input
        atom_h1 = atom_h.unsqueeze(2).expand(-1, -1, max_atoms, -1)
        atom_h2 = atom_h.unsqueeze(1).expand(-1, max_atoms, -1, -1)
        atom_pairs_h = torch.cat([atom_h1, atom_h2], dim=3)
        attn_input = torch.cat([atom_pairs_h, path_input], dim=3)

        return attn_input

    def compute_attn_probs(self, attn_input, attn_mask, layer_idx, eps=1e-20):
        # attn_scores is [batch, atoms, atoms, 1]
        if self.args.no_share:
            attn_scores = nn.LeakyReLU(0.2)(
                self.W_attn_h[layer_idx](attn_input))
            attn_scores = self.W_attn_o[layer_idx](attn_scores) * attn_mask
        else:
            attn_scores = nn.LeakyReLU(0.2)(
                self.W_attn_h(attn_input))
            attn_scores = self.W_attn_o(attn_scores) * attn_mask

        # max_scores is [batch, atoms, 1, 1], computed for stable softmax
        max_scores = torch.max(attn_scores, dim=2, keepdim=True)[0]
        # exp_attn is [batch, atoms, atoms, 1]
        exp_attn = torch.exp(attn_scores - max_scores) * attn_mask
        # sum_exp is [batch, atoms, 1, 1], add eps for stability
        sum_exp = torch.sum(exp_attn, dim=2, keepdim=True) + eps

        # attn_probs is [batch, atoms, atoms, 1]
        attn_probs = (exp_attn / sum_exp) * attn_mask
        return attn_probs

    def compute_nei_score(self, attn_probs, path_mask):
        # Compute the fraction of attn weights in the neighborhood
        nei_probs = attn_probs * path_mask.unsqueeze(3)
        nei_scores = torch.sum(nei_probs, dim=2)
        avg_score = torch.sum(nei_scores) / torch.sum(nei_scores != 0).float()
        return avg_score.item()

    def avg_attn(self, attn_probs, n_heads, batch_sz, max_atoms):
        if n_heads > 1:
            attn_probs = attn_probs.view(n_heads, batch_sz, max_atoms, max_atoms)
            attn_probs = torch.mean(attn_probs, dim=0)
        return attn_probs

    def forward(self, mol_graph, stats_tracker=None):
        atom_input, scope = mol_graph.get_atom_inputs()
        max_atoms = model_utils.compute_max_atoms(scope)
        atom_input_3D, atom_mask = model_utils.convert_to_3D(
            atom_input, scope, max_atoms, self.args.device, self.args.self_attn)
        path_input, path_mask = mol_graph.path_input, mol_graph.path_mask

        batch_sz, _, _ = atom_input_3D.size()
        n_heads, d_k = self.args.n_heads, self.args.d_k

        # Atom mask allows all valid atoms
        # Path mask allows only atoms in the neighborhood
        if self.args.mask_neigh:
            attn_mask = path_mask
        else:
            attn_mask = atom_mask.float()
        attn_mask = attn_mask.unsqueeze(3)

        if n_heads > 1:
            attn_mask = attn_mask.repeat(n_heads, 1, 1, 1)
            path_input = path_input.repeat(n_heads, 1, 1, 1)
            path_mask = path_mask.repeat(n_heads, 1, 1)  # Used to compute neighbor score

        atom_input_h = self.W_atom_i(atom_input_3D).view(batch_sz, max_atoms, n_heads, d_k)
        atom_input_h = atom_input_h.permute(2, 0, 1, 3).contiguous().view(-1, max_atoms, d_k)

        attn_list, nei_scores = [], []

        # atom_h should be [batch_size * n_heads, atoms, # features]
        atom_h = atom_input_h
        for layer_idx in range(self.args.depth - 1):
            attn_input = self.get_attn_input(atom_h, path_input, max_atoms)

            attn_probs = self.compute_attn_probs(attn_input, attn_mask, layer_idx)
            attn_list.append(self.avg_attn(attn_probs, n_heads, batch_sz, max_atoms))
            nei_scores.append(self.compute_nei_score(attn_probs, path_mask))
            attn_probs = self.dropout(attn_probs)

            if self.args.no_share:
                attn_h = self.W_message_h[layer_idx](
                    torch.sum(attn_probs * attn_input, dim=2))
            else:
                attn_h = self.W_message_h(
                    torch.sum(attn_probs * attn_input, dim=2))
            atom_h = nn.ReLU()(attn_h + atom_input_h)

        # Concat heads
        atom_h = atom_h.view(n_heads, batch_sz, max_atoms, -1)
        atom_h = atom_h.permute(1, 2, 0, 3).contiguous().view(batch_sz, max_atoms, -1)

        atom_h = model_utils.convert_to_2D(atom_h, scope)
        atom_output = torch.cat([atom_input, atom_h], dim=1)
        atom_h = nn.ReLU()(self.W_atom_o(atom_output))

        nei_scores = np.array(nei_scores)
        if stats_tracker is not None:
            stats_tracker.add_stat('nei_score', np.mean(nei_scores), 1)

        return atom_h, attn_list
