r"""
FOX.ff.bonded_calculate
=======================

A module for calculating bonded interactions using harmonic + cosine potentials.

.. math::

    V_{bonds} = k_{r} (r - r_{0})^2

    V_{angles} = k_{\theta} (\theta - \theta_{0})^2

    V_{diehdrals} = k_{\phi} [1 + \cos(n \phi - \delta)]

    V_{impropers} = k_{\omega} (\omega - \omega_{0})^2


Inter-ligand non-covalent interactions:

.. math::

    V_{LJ} = 4 \varepsilon
        \left(
            \left(
                \frac{\sigma}{r}
            \right )^{12} -
            \left(
                \frac{\sigma}{r}
            \right )^6
        \right )

    V_{Coulomb} = \frac{1}{4 \pi \varepsilon_{0}} \frac{q_{i} q_{j}}{r_{ij}}

"""

from typing import Union
from itertools import permutations

import numpy as np
import pandas as pd

from scm.plams import Units

from .lj_calculate import psf_to_atom_dict
from ..classes.multi_mol import MultiMolecule
from ..io.read_psf import PSFContainer
from ..io.read_prm import PRMContainer

__all__ = ['get_bonded']


def get_bonded(mol: Union[str, MultiMolecule],
               psf: Union[str, PSFContainer],
               prm: Union[str, PRMContainer]) -> pd.DataFrame:
    r"""Collect forcefield parameters and calculate all intra-ligand interactions in **mol**.

    Forcefield parameters are collected from the provided **psf** and **prm** files.

    Parameters
    ----------
    mol : :class:`str` or :class:`MultiMolecule`
        A MultiMolecule instance or the path+filename of an .xyz file.

    psf : :class:`str` or :class:`PSFContainer`
        A PSFContainer instance or the path+filename of a .psf file.
         Used for setting :math:`q` and creating atom-subsets.

    prm : :class:`str` or :class:`PRMContainer`, optional
        A PRMContainer instance or the path+filename of a .prm file.
        Used for setting :math:`\sigma` and :math:`\varepsilon`.

    Returns
    -------
    4x :class:`pandas.Series` and/or ``None``
        Four series with the potential energies of all bonds, angles, proper and
        improper dihedral angles.
        A Series is replaced with ``None`` if no parameters are available for that particular
        section.
        Units are in atomic units.

    """
    # Read the .psf file and switch from 1- to 0-based atomic indices
    if not isinstance(psf, PSFContainer):
        psf = PSFContainer.read(psf)
    else:
        psf = psf.copy()
    psf.bonds -= 1
    psf.angles -= 1
    psf.dihedrals -= 1
    psf.impropers -= 1

    # Read the molecule
    if not isinstance(mol, MultiMolecule):
        mol = MultiMolecule.from_xyz(mol)
    mol.atoms = psf_to_atom_dict(psf)

    # Extract parameters from the .prm file
    bonds, angles, dihedrals, impropers = process_prm(prm)

    # Calculate the various potential energies
    if bonds is not None:
        set_V_bonds(bonds, mol, psf.bonds)
        bonds = bonds['V'] * Units.conversion_ratio('kcal/mol', 'au')

    if angles is not None:
        set_V_angles(angles, mol, psf.angles)
        angles = angles['V'] * Units.conversion_ratio('kcal/mol', 'au')

    if dihedrals is not None:
        set_V_dihedrals(dihedrals, mol, psf.dihedrals)
        dihedrals = dihedrals['V'] * Units.conversion_ratio('kcal/mol', 'au')

    if impropers is not None:
        set_V_impropers(impropers, mol, psf.impropers)
        impropers = impropers['V'] * Units.conversion_ratio('kcal/mol', 'au')

    return bonds, angles, dihedrals, impropers


def process_prm(prm: Union[PRMContainer, str]):
    """Extract all bond, angle, dihedral and improper parameters from **prm**."""
    if not isinstance(prm, PRMContainer):
        prm = PRMContainer.read(prm)
    else:
        prm = prm.copy()

    bonds = prm.bonds
    if bonds is not None:
        bonds = bonds.set_index([0, 1])[[2, 3]]
        bonds[:] = bonds.astype(float)
        bonds['V'] = np.nan

    angles = prm.angles
    if angles is not None:
        angles = angles.set_index([0, 1, 2])[[3, 4]]
        angles[:] = angles.astype(float)
        angles[4] *= np.radians(1)
        angles['V'] = np.nan

    dihedrals = prm.dihedrals
    if dihedrals is not None:
        dihedrals = dihedrals.set_index([0, 1, 2, 3])[[4, 5, 6]]
        dihedrals[:] = dihedrals.astype(float)
        dihedrals[6] *= np.radians(1)
        dihedrals['V'] = np.nan

    impropers = prm.impropers
    if impropers is not None:
        impropers = impropers.set_index([0, 1, 2, 3])[[4, 6]]
        impropers[:] = impropers.astype(float)
        impropers[6] *= np.radians(1)
        impropers['V'] = np.nan

    return bonds, angles, dihedrals, impropers


def set_V_bonds(df: pd.DataFrame, mol: MultiMolecule, bond_idx: np.ndarray) -> None:
    """Calculate and set :math:`V_{bonds}` in **df**.

    Parameters
    ----------
    df : :class:`pd.DataFrame`
        A DataFrame with atom pairs and parameters.

    mol : :class:`MultiMolecule`
        A MultiMolecule instance.

    bond_idx : :math:`i*2` :class:`numpy.ndarray`
         A 2D numpy array with all atom-pairs defining bonds.

    """
    symbol = mol.symbol
    distance = _dist(mol, bond_idx)

    iterator = df.iloc[:, 0:2].iterrows()
    for i, item in iterator:
        j = np.all(symbol[bond_idx] == i, axis=1)
        j |= np.all(symbol[bond_idx[:, ::-1]] == i, axis=1)  # Consider all valid permutations
        df.at[i, 'V'] = get_V_harmonic(distance[:, j], *item).sum()


def set_V_angles(df: pd.DataFrame, mol: MultiMolecule, angle_idx: np.ndarray) -> None:
    """Calculate and set :math:`V_{angles}` in **df**.

    Parameters
    ----------
    df : :class:`pd.DataFrame`
        A DataFrame with atom pairs and parameters.

    mol : :class:`MultiMolecule`
        A MultiMolecule instance.

    bond_idx : :math:`i*3` :class:`numpy.ndarray`
         A 2D numpy array with all atom-pairs defining bonds.

    """
    symbol = mol.symbol
    angle = _angle(mol, angle_idx)

    iterator = df.iloc[:, 0:2].iterrows()
    for i, item in iterator:
        j = np.all(symbol[angle_idx] == i, axis=1)
        j |= np.all(symbol[angle_idx[:, ::-1]] == i, axis=1)  # Consider all valid permutations
        df.at[i, 'V'] = get_V_harmonic(angle[:, j], *item).sum()


def set_V_dihedrals(df: pd.DataFrame, mol: MultiMolecule, dihed_idx: np.ndarray) -> None:
    """Calculate and set :math:`V_{dihedrals}` in **df**.

    Parameters
    ----------
    df : :class:`pd.DataFrame`
        A DataFrame with atom pairs and parameters.

    mol : :class:`MultiMolecule`
        A MultiMolecule instance.

    bond_idx : :math:`i*4` :class:`numpy.ndarray`
         A numpy array with all atom-pairs defining proper dihedral angles.

    """
    symbol = mol.symbol
    dihedral = _dihed(mol, dihed_idx)

    iterator = df.iloc[:, 0:3].iterrows()
    for i, item in iterator:
        j = np.all(symbol[dihed_idx] == i, axis=1)
        j |= np.all(symbol[dihed_idx[:, ::-1]] == i, axis=1)  # Consider all valid permutations
        df.at[i, 'V'] = get_V_cos(dihedral[:, j], *item).sum()


def set_V_impropers(df: pd.DataFrame, mol: MultiMolecule, improp_idx: np.ndarray) -> None:
    """Calculate and set :math:`V_{impropers}` in **df**.

    Parameters
    ----------
    df : :class:`pd.DataFrame`
        A DataFrame with atom pairs and parameters.

    mol : :class:`MultiMolecule`
        A MultiMolecule instance.

    bond_idx : :math:`i*2` :class:`numpy.ndarray`
         A numpy array with all atom-pairs defining improper dihedral angles.

    """
    symbol = mol.symbol
    improper = _dihed(mol, improp_idx)

    iterator = df.iloc[:, 0:2].iterrows()
    for i, item in iterator:
        j = np.zeros(len(improp_idx), dtype=bool)
        for k in permutations([1, 2, 3], r=3):
            k = (0,) + k
            j |= np.all(symbol[improp_idx[:, k]] == i, axis=1)  # Consider all valid permutations
        df.at[i, 'V'] = get_V_harmonic(improper[:, j], *item).sum()


def _dist(mol: np.ndarray, ij: np.ndarray) -> np.ndarray:
    """Return an array with :code:`len(mol), len(ij)` distances (same unit as **mol**)."""
    i, j = ij.T
    return np.linalg.norm(mol[:, i] - mol[:, j], axis=-1)


def _angle(mol: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """Return an array with :code:`len(mol), len(ijk)` angles (radian)."""
    i, j, k = ijk.T

    vec1 = (mol[:, i] - mol[:, j])
    vec2 = (mol[:, k] - mol[:, j])
    vec1 /= np.linalg.norm(vec1, axis=-1)[..., None]
    vec2 /= np.linalg.norm(vec2, axis=-1)[..., None]

    return np.arccos((vec1 * vec2).sum(axis=2))


def _dihed(mol: np.ndarray, ijkm: np.ndarray) -> np.ndarray:
    """Return an array with :code:`len(mol), len(ijkm)` dihedral angles (radian)."""
    i, j, k, m = ijkm.T
    b0 = mol[:, i] - mol[:, j]
    b1 = mol[:, k] - mol[:, j]
    b2 = mol[:, m] - mol[:, k]

    b1 /= np.linalg.norm(b1, axis=-1)[..., None]
    v = b0 - (b0 * b1).sum(axis=2)[..., None] * b1
    w = b2 - (b2 * b1).sum(axis=2)[..., None] * b1

    x = (v * w).sum(axis=2)
    y = (np.cross(b1, v) * w).sum(axis=2)
    ret = np.arctan2(y, x)
    return np.abs(ret)


def get_V_harmonic(x: np.ndarray, k: float, x0: float) -> float:
    r"""Calculate the harmonic potential energy: :math:`\sum_{i} k (x_{i} - x_{0})^2`.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        An array of geometry parameters such as distances, angles or improper dihedral angles.
        Units should be in

    x_ref : :class:`float`
        The equilibrium value of **x**; units should be in Angstroem or radian.

    k : :class:`float`
        The force constant :math:`k`; units should be in kcal/mol/Angstroem**2 or kcal/mol/rad**2.

    """
    V = k * (x - x0)**2
    return V.mean(axis=0)


def get_V_cos(phi: np.ndarray, k: float, n: int, delta: float = 0.0) -> float:
    r"""Calculate the cosine potential energy: :math:`\sum_{i} k_{\phi} [1 + \cos(n \phi_{i} - \delta)]`.

    Parameters
    ----------
    phi : :class:`numpy.ndarray`
        An array of dihedral angles; units should be in radian.

    k : :class:`float`
        The force constant :math:`k`; units should be in kcal/mol.

    n : :class:`int`
        The multiplicity :math:`n`.

    delta : :class:`float`
        The phase-correction :math:`\delta`; units should be in radian.

    """  # noqa
    V = k * np.cos(n * phi - delta)
    return V.mean(axis=0)


"""
prm_file = '/Users/basvanbeek/Documents/GitHub/auto-FOX/FOX/examples/ligand.prm'
psf_file = '/Users/basvanbeek/Downloads/mol.psf'
xyz_file = get_example_xyz()

psf = PSFContainer.read(psf_file)
psf.bonds -= 1
psf.angles -= 1
psf.impropers -= 1

mol = MultiMolecule.from_xyz(xyz_file)
mol.atoms = psf_to_atom_dict(psf)

# non-covalent

nb_mol = mol.delete_atoms(['Cd', 'Se'])
nb_mol.guess_bonds()
molecule = nb_mol.as_Molecule(0)[0]
molecule.set_atoms_id(start=0)
at_count = len(molecule) // (psf.residue_id.max() - 1)


def dfs(at1: Atom, id_list: list, i: int, exclude: Set[Atom], depth: int = 0):
    exclude.add(at1)
    for bond in at1.bonds:
        at2 = bond.other_end(at1)
        if at2 in exclude:
            continue
        elif depth > 3:
            id_list += [i, at2.id]
        dfs(at2, id_list, i, exclude, depth=1+depth)


def gather_idx(molecule: Molecule) -> Generator[List[int], None, None]:
    for i, at in enumerate(molecule):
        id_list = []
        dfs(at, id_list.append, i, set())
        yield id_list


idx = np.fromiter(chain.from_iterable(gather_idx(molecule)), dtype=int)
idx += len(mol.atoms['Cd']) + len(mol.atoms['Se'])
idx.shape = -1, 2
"""
