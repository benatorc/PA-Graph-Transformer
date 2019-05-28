import rdkit.Chem as Chem

from preprocess.shortest_paths import get_shortest_paths
import pdb


def assert_dict_equal(source, target):
    assert len(target) == len(source)

    for k, v in target.items():
        assert v == source[k]


def test_simple():
    test_smiles = 'CC(O)C'
    rd_mol = Chem.MolFromSmiles(test_smiles)

    paths_dict, pointer_dict, rings_dict = get_shortest_paths(rd_mol)
    paths_dict_target = {
        (0, 1): (0, 1),
        (0, 2): (0, 1, 2),
        (0, 3): (0, 1, 3),
        (1, 2): (1, 2),
        (1, 3): (1, 3),
        (2, 3): (2, 1, 3)}
    assert_dict_equal(paths_dict, paths_dict_target)
    assert_dict_equal(pointer_dict, {})

    paths_dict, pointer_dict, rings_dict = get_shortest_paths(
        rd_mol, max_path_length=1)
    paths_dict_target = {
        (0, 1): (0, 1),
        (1, 2): (1, 2),
        (1, 3): (1, 3)}
    pointer_dict_target = {
        (0, 2): 1,
        (2, 0): 1,
        (0, 3): 1,
        (3, 0): 1,
        (2, 3): 1,
        (3, 2): 1}
    assert_dict_equal(paths_dict, paths_dict_target)
    assert_dict_equal(pointer_dict, pointer_dict_target)


def test_complex():
    test_smiles = 'COCCCNc1nc(NC(C)C)nc(SC)n1'

    paths_dict, pointer_dict, rings_dict = get_shortest_paths(
        Chem.MolFromSmiles(test_smiles), max_path_length=3)

    assert (5, 8) in paths_dict
    assert (13, 15) in paths_dict
    assert (13, 16) in paths_dict

    assert (8, 5) not in paths_dict  # only include ordered pairs
    assert (5, 9) not in paths_dict  # path length is 4
    assert (8, 16) not in paths_dict  # path length is 4

    assert (8, 5) not in pointer_dict  # should not be in pointers either
    assert (5, 9) in pointer_dict
    assert (8, 16) in pointer_dict
    assert (9, 5) in pointer_dict
    assert (16, 8) in pointer_dict


def test_simple_ring():
    test_smiles = 'C1NCCCC1'
    test_mol = Chem.MolFromSmiles(test_smiles)
    n_atoms = test_mol.GetNumAtoms()

    paths_dict, pointer_dict, rings_dict = get_shortest_paths(
        test_mol, max_path_length=5)

    count = 0
    for i in range(n_atoms):
        for j in range(i, n_atoms, 1):
            count += 1
            assert (i, j) in rings_dict
            assert rings_dict[(i, j)] == [(6, False)]
    assert len(rings_dict) == count


def test_simple_aro_ring():
    test_smiles = 'c1ccccc1'
    test_mol = Chem.MolFromSmiles(test_smiles)
    n_atoms = test_mol.GetNumAtoms()

    paths_dict, pointer_dict, rings_dict = get_shortest_paths(
        test_mol, max_path_length=3)

    count = 0
    for i in range(n_atoms):
        for j in range(i, n_atoms, 1):  # Count all pairs including itself
            count += 1
            assert (i, j) in rings_dict
            assert rings_dict[(i, j)] == [(6, True)]
    assert len(rings_dict) == count


def test_fused_ring():
    test_smiles = 'o2c1ccccc1cc2'
    test_mol = Chem.MolFromSmiles(test_smiles)

    paths_dict, pointer_dict, rings_dict = get_shortest_paths(
        test_mol, max_path_length=3)

    assert (0, 1) in paths_dict
    assert (0, 4) not in paths_dict

    assert (6, 1) not in rings_dict
    assert (1, 6) in rings_dict
    assert rings_dict[(1, 6)] == [(5, True), (6, True)]

    assert (1, 3) in rings_dict
    assert rings_dict[(1, 3)] == [(6, True)]
