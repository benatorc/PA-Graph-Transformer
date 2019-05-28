import pickle
import argparse
import tqdm
import rdkit.Chem as Chem

import pdb


def ordered_pair(a1, a2):
    if a1 > a2:
        return (a2, a1)
    else:
        return (a1, a2)


def get_ring_paths(rd_mol):
    rings_dict = {}
    ssr = [list(x) for x in Chem.GetSymmSSSR(rd_mol)]
    for ring in ssr:
        ring_sz = len(ring)
        is_aromatic = True
        for atom_idx in ring:
            if not rd_mol.GetAtoms()[atom_idx].GetIsAromatic():
                is_aromatic = False
                break
        for ring_idx, atom_idx in enumerate(ring):
            for other_idx in ring[ring_idx:]:
                atom_pair = ordered_pair(atom_idx, other_idx)
                if atom_pair not in rings_dict:
                    rings_dict[atom_pair] = [(ring_sz, is_aromatic)]
                else:
                    if (ring_sz, is_aromatic) not in rings_dict[atom_pair]:
                        rings_dict[atom_pair].append((ring_sz, is_aromatic))
    return rings_dict


def get_shortest_paths(rd_mol, max_path_length=5):
    """Returns the shortest paths for the given rd_mol.

    For every pair of atoms, if they are connected by a path <= max_path_length
        the atoms on that path are enumerated. Otherwise, if there is a path,
        then the truncated path is given.
    The returned object is a tuple of dictionaries. The first dictionary
        contains the shortest paths for those with <= max_path_length. The
        second contains a pointer to the truncated path. The third contains
        the ring information for atoms inside rings.
    """
    fragments = Chem.rdmolops.GetMolFrags(rd_mol)

    def get_atom_frag(fragments, atom_idx):
        """Returns the fragment the atom belongs to."""
        for frag in fragments:
            if atom_idx in frag:
                return frag
        assert False  # All valid indices should belong to a fragment

    n_atoms = rd_mol.GetNumAtoms()
    paths_dict = {}  # Contains atom pairs with path length <= max_path_length
    pointer_dict = {}  # Contains atom pairs with path length > max_path_length

    for atom_idx in range(n_atoms):
        atom_frag = get_atom_frag(fragments, atom_idx)  # List of atoms in the atom's fragment

        # this iteration avoids visiting the same atom pair more than once
        for other_idx in range(atom_idx + 1, n_atoms, 1):
            if other_idx not in atom_frag:
                continue

            shortest_path = Chem.rdmolops.GetShortestPath(
                rd_mol, atom_idx, other_idx)

            path_length = len(shortest_path) - 1  # shortest_path counts atoms
            if path_length > max_path_length:
                pointer_dict[
                    (atom_idx, other_idx)] = shortest_path[max_path_length]
                pointer_dict[
                    (other_idx, atom_idx)] = shortest_path[-1 - max_path_length]
            else:
                paths_dict[(atom_idx, other_idx)] = shortest_path

    ring_dict = get_ring_paths(rd_mol)
    return paths_dict, pointer_dict, ring_dict


def read_mol_smiles(data_path):
    data_file = open(data_path, 'r+')

    smiles_list = []
    for line in data_file.readlines():
        smiles = line.split(',')[0]
        smiles_list.append(smiles)

    data_file.close()
    return smiles_list


def parse_mol(data_dir, max_path_length):
    shortest_paths = {}
    data_path = '%s/raw.csv' % data_dir
    smiles_list = read_mol_smiles(data_path)

    for smiles in tqdm.tqdm(smiles_list):
        rd_mol = Chem.MolFromSmiles(smiles)
        if rd_mol is None:
            print('Parsing invalid molecule.')
            pdb.set_trace()
        shortest_paths[smiles] = get_shortest_paths(
            rd_mol, max_path_length)
    sp_output_path = '%s/shortest_paths.p' % data_dir
    pickle.dump(shortest_paths, open(sp_output_path, 'wb'))
    print('Saved Shortest Paths to: %s' % sp_output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, default='')
    parser.add_argument('-max_path_length', type=int, default=5)
    args = parser.parse_args()

    parse_mol(args.data_dir, args.max_path_length)


if __name__ == '__main__':
    main()
