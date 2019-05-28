import torch
import numpy as np
import rdkit.Chem as Chem

import graph.mol_features as mol_features
import pdb


# Supress warnings from rdkit
from rdkit import rdBase
from rdkit import RDLogger
rdBase.DisableLog('rdApp.error')
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


class Atom:
    def __init__(self, idx, rd_atom=None, is_dummy=False):
        """Initialize the atom object to keep track of its attributes.

        Args:
            idx: The index of the atom in the original molecule.
            rd_atom: If provided the rdkit atom object, used to extract
                features.
        """
        self.idx = idx
        self.bonds = []
        self.is_dummy = is_dummy

        if is_dummy:
            self.symbol = '*'  # Default wildcard/dummy symbol

        if rd_atom is not None:
            self.symbol = rd_atom.GetSymbol()
            self.fc = rd_atom.GetFormalCharge()
            self.degree = rd_atom.GetDegree()
            self.exp_valence = rd_atom.GetExplicitValence()
            self.imp_valence = rd_atom.GetImplicitValence()
            self.aro = int(rd_atom.GetIsAromatic())

    def add_bond(self, bond):
        self.bonds.append(bond)  # Includes all the incoming bond indices


class Bond:
    def __init__(self, idx, out_atom_idx, in_atom_idx, rd_bond=None):
        """Initialize the bond object to keep track of its attributes."""
        self.idx = idx
        self.out_atom_idx = out_atom_idx
        self.in_atom_idx = in_atom_idx

        if rd_bond is not None:
            self.bond_type = rd_bond.GetBondType()
            self.is_conjugated = int(rd_bond.GetIsConjugated())
            self.is_in_ring = int(rd_bond.IsInRing())


class Molecule:
    def __init__(self, atoms, bonds):
        self.atoms = atoms
        self.bonds = bonds

    def get_bond(self, atom_1, atom_2):
        # If bond does not exist between atom_1 and atom_2, return None
        for bond in self.atoms[atom_1].bonds:
            if atom_2 == bond.out_atom_idx or atom_2 == bond.in_atom_idx:
                return bond
        return None


class MolGraph:
    def __init__(self, smiles_list, args, path_input=None, path_mask=None):
        """Initialize the molecular graph inputs for the smiles list.

        Args:
            smiles_list: The input smiles strings in a list
        """
        self.smiles_list = smiles_list
        self.args = args
        self.device = args.device

        self.mols = []  # Molecule objects list
        self.scope = []  # Tuples of (st, le) for atoms for mols
        self.rd_mols = []

        self.path_input = path_input
        self.path_mask = path_mask
        self.ap_mapping = None

        self._parse_molecules(smiles_list)
        self.n_mols = len(self.mols)

    def get_n_atoms(self):
        assert self.scope != []
        return self.scope[-1][0] + self.scope[-1][1]

    def _parse_molecules(self, smiles_list):
        """Turn the smiles into atom and bonds through rdkit.

        Every bond is recorded as two directional bonds, and for each atom,
            keep track of all the incoming bonds, since these are necessary for
            aggregating the final atom feature output in the conv net.

        Args:
            smiles_list: A list of input smiles strings. Assumes that the given
                strings are valid.
            max_atoms: If provided, truncate graphs to this size.
        """
        def skip_atom(atom_idx, max):
            return (max != 0) and (atom_idx >= max)

        a_offset = 0  # atom offset

        for smiles in smiles_list:
            rd_mol = Chem.MolFromSmiles(smiles)
            self.rd_mols.append(rd_mol)

            mol_atoms = []
            mol_bonds = []
            for rd_atom in rd_mol.GetAtoms():
                atom_idx = rd_atom.GetIdx()
                mol_atoms.append(Atom(idx=atom_idx, rd_atom=rd_atom))

            for rd_bond in rd_mol.GetBonds():
                atom_1_idx = rd_bond.GetBeginAtom().GetIdx()
                atom_2_idx = rd_bond.GetEndAtom().GetIdx()

                bond_idx = len(mol_bonds)
                new_bond = Bond(bond_idx, atom_1_idx, atom_2_idx, rd_bond)
                mol_bonds.append(new_bond)
                mol_atoms[atom_2_idx].add_bond(new_bond)  # bond is 1 -> 2

                bond_idx = len(mol_bonds)
                new_bond = Bond(bond_idx, atom_2_idx, atom_1_idx, rd_bond)
                mol_bonds.append(new_bond)
                mol_atoms[atom_1_idx].add_bond(new_bond)  # bond is 2 -> 1

            new_mol = Molecule(mol_atoms, mol_bonds)
            self.mols.append(new_mol)

            self.scope.append((a_offset, len(mol_atoms)))
            a_offset += len(mol_atoms)

    def get_atom_inputs(self, output_tensors=True):
        """Constructs only the atom inputs for the batch of molecules."""
        fatoms = []

        for mol_idx, mol in enumerate(self.mols):
            atoms = mol.atoms

            for atom_idx, atom in enumerate(atoms):
                atom_features = mol_features.get_atom_features(atom)
                fatoms.append(atom_features)

        fatoms = np.stack(fatoms, axis=0)
        if output_tensors:
            fatoms = torch.tensor(fatoms, device=self.device).float()
        return fatoms, self.scope

    def get_graph_inputs(self, output_tensors=True):
        """Constructs the graph inputs for the conv net.

        Returns:
            A tuple of tensors/numpy arrays that contains the input to the GCN.
        """
        n_atom_feats = mol_features.N_ATOM_FEATS
        n_bond_feats = mol_features.N_BOND_FEATS
        max_neighbors = mol_features.MAX_NEIGHBORS

        # The feature matrices for the atoms and bonds
        fatoms = []
        fbonds = [np.zeros(n_atom_feats + n_bond_feats)]  # Zero padded
        # The graph matrices for aggregation in conv net
        agraph = []
        bgraph = [np.zeros([1, max_neighbors])]  # Zero padded
        b_offset = 1  # Account for padding

        for mol_idx, mol in enumerate(self.mols):
            atoms, bonds = mol.atoms, mol.bonds
            cur_agraph = np.zeros([len(atoms), max_neighbors])
            cur_bgraph = np.zeros([len(bonds), max_neighbors])

            for atom_idx, atom in enumerate(atoms):
                atom_features = mol_features.get_atom_features(atom)
                fatoms.append(atom_features)
                for nei_idx, bond in enumerate(atom.bonds):
                    cur_agraph[atom.idx, nei_idx] = bond.idx + b_offset
            for bond in bonds:
                out_atom = atoms[bond.out_atom_idx]
                bond_features = np.concatenate([
                    mol_features.get_atom_features(out_atom),
                    mol_features.get_bond_features(bond)], axis=0)
                fbonds.append(bond_features)
                for i, in_bond in enumerate(out_atom.bonds):
                    if bonds[in_bond.idx].out_atom_idx != bond.in_atom_idx:
                        cur_bgraph[bond.idx, i] = in_bond.idx + b_offset

            agraph.append(cur_agraph)
            bgraph.append(cur_bgraph)
            b_offset += len(bonds)

        fatoms = np.stack(fatoms, axis=0)
        fbonds = np.stack(fbonds, axis=0)
        agraph = np.concatenate(agraph, axis=0)
        bgraph = np.concatenate(bgraph, axis=0)

        if output_tensors:
            fatoms = torch.tensor(fatoms, device=self.device).float()
            fbonds = torch.tensor(fbonds, device=self.device).float()
            agraph = torch.tensor(agraph, device=self.device).long()
            bgraph = torch.tensor(bgraph, device=self.device).long()

        graph_inputs = [fatoms, fbonds, agraph, bgraph]
        return (graph_inputs, self.scope)


if __name__ == '__main__':
    smiles = 'c1ccccc1O'
    mol_graph = MolGraph([smiles], None)
