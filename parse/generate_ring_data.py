import argparse
import tqdm
import rdkit.Chem as Chem
import random

import pdb


MAX_IT = 200


def ordered_pair(pair):
    a1, a2 = pair
    if a1 < a2:
        return (a1, a2)
    else:
        return (a2, a1)


def read_smiles(data_path):
    data_file = open(data_path, 'r+')

    smiles_list = []
    for line in data_file.readlines():
        smiles = line.strip().split(',')[0]
        smiles_list.append(smiles)
    return smiles_list


def generate_ring_data(smiles, n_pos, n_neg):
    ring_data = {}

    mol = Chem.MolFromSmiles(smiles)
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]

    n_rings = len(ssr)
    ring_indices = list(range(n_rings))
    if n_rings <= 1:  # Only collect data when at least 2 rings are present
        return ring_data

    for idx in range(n_pos):
        ring_idx = random.sample(ring_indices, 1)[0]
        ring_atoms = ssr[ring_idx]

        for i in range(MAX_IT):
            atom_pair = ordered_pair(random.sample(ring_atoms, 2))
            if atom_pair not in ring_data:
                ring_data[atom_pair] = 1
                break
    for idx in range(n_neg):
        ring_pair = random.sample(ring_indices, 2)
        ring_1 = ssr[ring_pair[0]]
        ring_2 = ssr[ring_pair[1]]

        for i in range(MAX_IT):
            a1 = random.sample(ring_1, 1)[0]
            a2 = random.sample(ring_2, 1)[0]

            if a1 == a2:
                continue

            same_ring = False
            for ring in ssr:
                if a1 in ring and a2 in ring:
                    same_ring = True
                    break
            if same_ring:
                continue

            atom_pair = ordered_pair([a1, a2])
            if atom_pair not in ring_data:
                ring_data[atom_pair] = 0
                break
    return ring_data


def generate_ring_position_data(smiles, max_samples):
    ring_data = {}

    mol = Chem.MolFromSmiles(smiles)
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]

    filtered_ssr = []
    for ring in ssr:
        if len(ring) == 6:
            is_aromatic = mol.GetAtoms()[ring[0]].GetIsAromatic()
            if is_aromatic:
                filtered_ssr.append(ring)
    ssr = filtered_ssr

    n_rings = len(ssr)
    if n_rings == 0:
        return ring_data
    ring_indices = list(range(n_rings))

    def get_non_ring_neighbor(mol, atom_idx, ring):
        atom = mol.GetAtoms()[atom_idx]
        for neigh_atom in atom.GetNeighbors():
            neigh_idx = neigh_atom.GetIdx()
            if neigh_idx not in ring:
                return neigh_idx
        return None

    for idx in range(MAX_IT):
        ring = ssr[random.sample(ring_indices, 1)[0]]
        atom_pair = ordered_pair(random.sample(ring, 2))
        a1, a2 = atom_pair

        n1 = get_non_ring_neighbor(mol, a1, ring)
        n2 = get_non_ring_neighbor(mol, a2, ring)

        if n1 is not None and n2 is not None:
            path = Chem.rdmolops.GetShortestPath(mol, a1, a2)
            path_length = len(path) - 1

            neigh_pair = ordered_pair([n1, n2])
            if neigh_pair not in ring_data:
                ring_data[neigh_pair] = path_length - 1  # Make length 0 indexed
        if len(ring_data) >= max_samples:
            break
    return ring_data


def is_conjugated_path(mol, path_atoms):
    for idx, atom in enumerate(path_atoms[:-1]):
        next_atom = path_atoms[idx+1]
        bond = mol.GetBondBetweenAtoms(atom, next_atom)
        if not bond.GetIsConjugated():
            return False
    return True


def generate_complex_ring_data(smiles, n_pos, n_neg):
    ring_data = {}

    mol = Chem.MolFromSmiles(smiles)
    n_atoms = mol.GetNumAtoms()
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]

    grouped_ssr = []
    # Want to group together fused rings:
    for ring_idx, ring in enumerate(ssr):
        matched = False

        for ring_group in grouped_ssr:
            if matched:
                break
            for compare_ring in ring_group:
                if len(set(ring).intersection(set(compare_ring))) > 0:
                    ring_group.append(ring)
                    matched = True
                    break
        if not matched:
            grouped_ssr.append([ring])

    multi_ssr = []
    for ring_group in grouped_ssr:
        if len(ring_group) > 1:
            multi_ssr.append(ring_group)

    if len(multi_ssr) == 0:
        return ring_data

    def get_neigh_if_conjugated(mol, atom_idx, ring):
        atom = mol.GetAtoms()[atom_idx]
        for neigh_atom in atom.GetNeighbors():
            neigh_idx = neigh_atom.GetIdx()
            if not mol.GetAtoms()[neigh_idx].IsInRing():
                bond = mol.GetBondBetweenAtoms(atom_idx, neigh_idx)
                if bond.GetIsConjugated():
                    return neigh_idx
        return atom_idx

    for i in range(MAX_IT):
        ring_group = random.sample(multi_ssr, 1)[0]
        ring_pair = random.sample(ring_group, 2)

        a1 = random.sample(ring_pair[0], 1)[0]
        a1 = get_neigh_if_conjugated(mol, a1, ring_pair[0])
        a2 = random.sample(ring_pair[1], 1)[0]
        a2 = get_neigh_if_conjugated(mol, a2, ring_pair[1])

        if a1 == a2:
            continue

        atom_pair = ordered_pair([a1, a2])
        if atom_pair in ring_data:
            continue

        path = Chem.rdmolops.GetShortestPath(mol, a1, a2)
        if len(path) < 4:
            continue
        is_conjugated = is_conjugated_path(mol, path)
        if not is_conjugated:
            continue

        ring_data[atom_pair] = 1
        if len(ring_data) >= n_pos:
            break

    for i in range(MAX_IT):
        ring = random.sample(ssr, 1)[0]
        a1 = random.sample(ring, 1)[0]

        a2 = random.randint(0, n_atoms-1)
        if a1 == a2:
            continue

        atom_pair = ordered_pair([a1, a2])
        if atom_pair in ring_data:
            continue

        path = Chem.rdmolops.GetShortestPath(mol, a1, a2)
        if len(path) > 6:
            continue
        is_conjugated = is_conjugated_path(mol, path)
        if is_conjugated:
            continue

        ring_data[atom_pair] = 0
        if len(ring_data) >= n_pos + n_neg:
            break
    return ring_data


def test():
    test_smiles = 'Clc1ccc2OCC(=O)N(CCN3CCC(CC3)NCc4ccc5OCC(=O)Nc5n4)c2c1'
    generate_complex_ring_data(test_smiles, 5, 5)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, default='')
    parser.add_argument('-output_dir', type=str, default='')
    parser.add_argument('-n_pos', type=int, default=5)
    parser.add_argument('-n_neg', type=int, default=5)
    parser.add_argument('-n_samples', type=int, default=5)
    parser.add_argument('-type', type=str, default='')
    args = parser.parse_args()

    assert args.type in ['membership', 'position', 'complex']

    smiles_list = read_smiles(args.data_path)
    output_file = open('%s/raw.csv' % args.output_dir, 'w+')

    for smiles in tqdm.tqdm(smiles_list):
        if args.type == 'membership':
            ring_data = generate_ring_data(smiles, args.n_pos, args.n_neg)
        elif args.type == 'complex':
            ring_data = generate_complex_ring_data(smiles, args.n_pos, args.n_neg)
        else:
            ring_data = generate_ring_position_data(smiles, args.n_samples)
        if len(ring_data) == 0:
            continue
        data_str = '%s' % smiles
        for atom_pair, label in ring_data.items():
            label_str = ',%d-%d:%d' % (atom_pair[0], atom_pair[1], label)
            data_str += label_str
        output_file.write('%s\n' % data_str)

    output_file.close()


if __name__ == '__main__':
    main()
    # test()
