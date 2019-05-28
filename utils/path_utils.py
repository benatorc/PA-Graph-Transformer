import torch
import numpy as np

import graph.mol_features as mol_features
import pdb


def ordered_pair(a1, a2):
    if a1 > a2:
        return (a2, a1)
    else:
        return (a1, a2)


def get_path_input(mols, shortest_paths, max_atoms, args, output_tensor=True):
    batch_size, max_path_length = len(shortest_paths), args.max_path_length
    n_path_features = get_num_path_features(args)

    path_input = []
    path_mask = []
    for mol_idx, mol in enumerate(mols):
        paths_dict, pointer_dict, ring_dict = shortest_paths[mol_idx]

        for atom_1 in range(max_atoms):
            for atom_2 in range(max_atoms):
                path_atoms, path_length, mask_ind = get_path_atoms(
                    atom_1, atom_2, paths_dict, pointer_dict, max_path_length,
                    truncate=not args.no_truncate, self_attn=args.self_attn)

                path_features = get_path_features(
                    mol, path_atoms, path_length, max_path_length,
                    args.p_embed)
                if args.ring_embed:
                    ring_features = get_ring_features(
                        ring_dict, ordered_pair(atom_1, atom_2))
                    path_features = np.concatenate(
                        [path_features, ring_features], axis=0)
                path_input.append(path_features)

                path_mask.append(mask_ind)

    if output_tensor:
        path_input = torch.tensor(path_input, device=args.device)
        path_input = path_input.view(
            [batch_size, max_atoms, max_atoms, n_path_features])
        path_mask = torch.tensor(path_mask, device=args.device)
        path_mask = path_mask.view([batch_size, max_atoms, max_atoms])
    else:
        path_input = np.stack(path_input, axis=0)
        path_input = np.reshape(
            path_input, [batch_size, max_atoms, max_atoms, n_path_features])
        path_mask = np.array(path_mask).reshape(
            [batch_size, max_atoms, max_atoms])
    return path_input, path_mask


def merge_path_inputs(path_inputs, path_masks, max_atoms, args):
    """Merge path input matrices. Does not create CUDA tensors intentionally"""
    batch_size, max_path_length = len(path_inputs), args.max_path_length
    num_features = get_num_path_features(args)
    feature_shape = [batch_size, max_atoms, max_atoms, num_features]
    mask_shape = [batch_size, max_atoms, max_atoms]

    padded_path_inputs = torch.zeros(feature_shape)
    padded_mask = torch.zeros(mask_shape)

    for idx, path_input in enumerate(path_inputs):
        path_input = torch.tensor(path_input)
        n_atoms = path_input.size()[0]
        padded_path_inputs[idx, :n_atoms, :n_atoms] = path_input

        path_mask = torch.tensor(path_masks[idx])
        padded_mask[idx, :n_atoms, :n_atoms] = path_mask
    padded_mask = padded_mask
    return padded_path_inputs, padded_mask


def get_num_path_features(args):
    """Returns the number of path features for the model."""
    num_features = 0
    num_features = args.max_path_length * mol_features.N_BOND_FEATS
    if args.p_embed:
        num_features += args.max_path_length + 2
    if args.ring_embed:
        n_ring_feats = 1  # Same ring membership
        n_ring_feats += 4  # Same ring non/aromatic 5 or 6
        num_features += n_ring_feats
    return num_features


def get_path_atoms(atom_1, atom_2, paths_dict, pointer_dict, max_path_length,
                   truncate=True, self_attn=False):
    """Given a pair of atom indices, returns the list of atoms on the path.

    Args:
        atom_1: The start atom on the path.
        atom_2: The end atom on the path.
        paths_dict: A mapping from atom pairs to paths.
        pointer_dict: A mapping from atom pairs to truncated paths. The values
            of the dict is the atom ending the truncated path
        max_path_length: The maximum path length to return.
        truncate: Boolean determining whether or not paths above the max length
            should be truncated (returned as no path)
    """
    path_start, path_end = atom_1, atom_2
    path_greater_max = False  # Indicator for path length exceeding max

    # pointer dict contains atom pairs that are longer than max path length
    if (atom_1, atom_2) in pointer_dict:
        path_greater_max = True
        if not truncate:
            path_start, path_end = atom_1, pointer_dict[(atom_1, atom_2)]

    path_atoms = []
    if (path_start, path_end) in paths_dict:
        path_atoms = paths_dict[(path_start, path_end)]
    elif (path_end, path_start) in paths_dict:
        path_atoms = paths_dict[(path_end, path_start)][::-1]  # Reverse list

    # Because the path lengths when computing shortest paths can be different
    # than the ones required for training, truncate if necessary
    if len(path_atoms) - 1 > max_path_length:
        path_atoms = [] if truncate else path_atoms[:max_path_length+1]
        path_greater_max = True

    mask_ind = 1
    path_length = 0 if path_atoms == [] else len(path_atoms) - 1
    if path_greater_max:
        path_length = max_path_length + 1
        mask_ind = 0
    if not self_attn:
        if atom_1 == atom_2:
            mask_ind = 0  # Get rid of self atom

    return list(path_atoms), path_length, mask_ind


def get_path_features(rd_mol, path_atoms, path_length, max_path_length,
                      p_embed=False):
    """Returns a feature array for the path.

    Args:
        rd_mol: The rdkit mol object, used to extract features.
        path_atoms: A list of atoms in the path. If no path exist, empty array.
        path_length: The length of the path.
        max_path_length: The maximum length of the path considered.
        p_embed: Whether or not to use position embedding.
    """
    # Compute the list of bonds from the path atoms
    path_bonds = []
    for path_idx in range(len(path_atoms) - 1):
        atom_1 = path_atoms[path_idx]
        atom_2 = path_atoms[path_idx + 1]

        bond = rd_mol.GetBondBetweenAtoms(atom_1, atom_2)
        assert bond is not None
        path_bonds.append(bond)

    features = []
    for path_idx in range(max_path_length):
        path_bond = path_bonds[path_idx] if path_idx < len(path_bonds) else None
        features.append(mol_features.get_path_bond_feature(path_bond))
    if p_embed:
        position_feature = np.zeros(max_path_length + 2)
        position_feature[path_length] = 1
        features.append(position_feature)
    return np.concatenate(features, axis=0)


def get_ring_features(ring_dict, atom_pair):
    ring_features = np.zeros(5)
    if atom_pair in ring_dict:
        ring_features[0] = 1  # Ring membership
        rings = ring_dict[atom_pair]
        for (ring_size, aromatic) in rings:
            if ring_size == 5 and not aromatic:
                ring_features[1] = 1
            elif ring_size == 5 and aromatic:
                ring_features[2] = 1
            elif ring_size == 6 and not aromatic:
                ring_features[3] = 1
            elif ring_size == 6 and aromatic:
                ring_features[4] = 1
    return ring_features
