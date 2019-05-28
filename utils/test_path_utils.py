import argparse
import sys
import rdkit.Chem as Chem

from preprocess.shortest_paths import get_shortest_paths
from utils import path_utils

import pdb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_path_length', type=int, default=5)
    parser.add_argument('--self_attn', action='store_true', default=False)
    parser.add_argument('--no_truncate', action='store_true', default=False)
    parser.add_argument('--p_embed', action='store_true', default=False)
    parser.add_argument('--ring_embed', action='store_true', default=False)
    args = parser.parse_args(sys.argv[2:])

    args.shortest_paths = None
    args.path_inputs = None
    return args


def test_path_input():
    args = get_args()
    args.max_path_length = 3
    args.self_attn = True
    args.ring_embed = True
    smiles = ['C1CC1CO', 'o2c1ccccc1cc2OC']
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    n_atoms = [m.GetNumAtoms() for m in mols]

    shortest_paths = [get_shortest_paths(m, 5) for m in mols]

    path_input1, path_mask1 = path_utils.get_path_input(
        [mols[0]], [shortest_paths[0]], n_atoms[0], args, output_tensor=False)
    path_input2, path_mask2 = path_utils.get_path_input(
        [mols[1]], [shortest_paths[1]], n_atoms[1], args, output_tensor=False)

    path_input1 = path_input1.squeeze(0)  # Remove batch dimension
    path_mask1 = path_mask1.squeeze(0)  # Remove batch dimension
    path_input2 = path_input2.squeeze(0)  # Remove batch dimension
    path_mask2 = path_mask2.squeeze(0)  # Remove batch dimension

    path_input, path_mask = path_utils.merge_path_inputs(
        [path_input1, path_input2], [path_mask1, path_mask2], max(n_atoms), args)
