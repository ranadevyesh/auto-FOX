"""A Module for the MultiMolecule class."""

from __future__ import annotations

from collections import abc
from itertools import (chain, combinations_with_replacement, zip_longest, islice)
from typing import (
    Sequence, Optional, Union, List, Hashable, Callable, Iterable, Dict, Tuple, Any
)

import numpy as np
import pandas as pd
from scipy import (signal, constants)
from scipy.spatial import cKDTree
from scipy.fftpack import fft
from scipy.spatial.distance import cdist

from scm.plams import (Atom, Bond, PeriodicTable)

from .molecule_utils import Molecule
from .multi_mol_magic import _MultiMolecule
from ..io.read_kf import read_kf
from ..io.read_xyz import read_multi_xyz
from ..functions.rdf import (get_rdf, get_rdf_lowmem, get_rdf_df)
from ..functions.adf import (get_adf, get_adf_df)

try:
    import dask
    DASK: bool = True
except ImportError:
    DASK: bool = False

__all__ = ['MultiMolecule']

MolSubset = Union[None, slice, int]
AtomSubset = Union[
    None, slice, range, int, str, Sequence[int], Sequence[str], Sequence[Sequence[int]]
]


class MultiMolecule(_MultiMolecule):
    """A class designed for handling a and manipulating large numbers of molecules.

    More specifically, different conformations of a single molecule as derived from, for example,
    an intrinsic reaction coordinate calculation (IRC) or a molecular dymanics trajectory (MD).
    The class has access to four attributes (further details are provided under parameters):

    Parameters
    ----------
    coords : :math:`m*n*3` |np.ndarray|_ [|np.float64|_]
        A 3D array with the cartesian coordinates of :math:`m` molecules with :math:`n` atoms.

    atoms : dict [str, list [int]]
        A dictionary with atomic symbols as keys and matching atomic indices as values.
        Stored in the :attr:`MultiMolecule.atoms` attribute.

    bonds : :math:`k*3` |np.ndarray|_ [|np.int64|_]
        A 2D array with indices of the atoms defining all :math:`k` bonds
        (columns 1 & 2) and their respective bond orders multiplied by 10 (column 3).
        Stored in the :attr:`MultiMolecule.bonds` attribute.

    properties : dict
        A Settings instance for storing miscellaneous user-defined (meta-)data.
        Is devoid of keys by default.
        Stored in the :attr:`MultiMolecule.properties` attribute.

    Attributes
    ----------
    atoms : dict [str, list [int]]
        A dictionary with atomic symbols as keys and matching atomic indices as values.

    bonds : :math:`k*3` |np.ndarray|_ [|np.int64|_]
        A 2D array with indices of the atoms defining all :math:`k` bonds
        (columns 1 & 2) and their respective bond orders multiplied by 10 (column 3).

    properties : |plams.Settings|_
        A Settings instance for storing miscellaneous user-defined (meta-)data.
        Is devoid of keys by default.

    """

    def delete_atoms(self, atom_subset: AtomSubset) -> MultiMolecule:
        """Create a copy of this instance with all atoms in **atom_subset** removed.

        Parameters
        ----------
        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        Returns
        -------
        |FOX.MultiMolecule|_
            A new :class:`.MultiMolecule` instance with all atoms in **atom_subset** removed.

        Raises
        ------
        TypeError
            Raised if **atom_subset** is ``None``.

        """
        if atom_subset is None:
            raise TypeError("'None' is an invalid value for 'atom_subset'")

        # Delete atoms
        at_subset = np.array(self._get_atom_subset(atom_subset, as_sequence=True))
        idx = np.arange(0, self.shape[1])[~at_subset]
        ret = self[:, idx].copy()

        # Update :attr:`.MultiMolecule.atoms`
        symbols = self.symbol[idx]
        ret.atoms = {}
        for i, at in enumerate(symbols):
            try:
                ret.atoms[at].append(i)
            except KeyError:
                ret.atoms[at] = [i]

        return ret

    def guess_bonds(self, atom_subset: AtomSubset = None) -> None:
        """Guess bonds within the molecules based on atom type and inter-atomic distances.

        Bonds are guessed based on the first molecule in this instance
        Performs an inplace modification of **self.bonds**

        Parameters
        ----------
        atom_subset : |Sequence|_
            A tuple of atomic symbols. Bonds are guessed between all atoms
            whose atomic symbol is in **atom_subset**.
            If ``None``, guess bonds for all atoms in this instance.

        """
        at_subset = np.asarray(sorted(self._get_atom_subset(atom_subset, as_sequence=True)))

        # Guess bonds
        mol = self.as_Molecule(mol_subset=0, atom_subset=atom_subset)[0]
        mol.guess_bonds()
        mol.fix_bond_orders()
        self.bonds = MultiMolecule.from_Molecule(mol, subset='bonds').bonds

        # Update indices in **self.bonds** to account for **atom_subset**
        self.atom1 = at_subset[self.atom1]
        self.atom2 = at_subset[self.atom2]
        self.bonds[:, 0:2].sort(axis=1)
        idx = self.bonds[:, 0:2].argsort(axis=0)[:, 0]
        self.bonds = self.bonds[idx]

    def slice_mol(self, start: int = 0,
                  stop: Optional[int] = None,
                  step: int = 1,
                  inplace: bool = False) -> Optional[MultiMolecule]:
        """Construct a new :class:`MultiMolecule` instance by slicing this instance.

        Equivalent to :code:`MultiMolecule[start:stop:step].copy()` or
        :code:`MultiMolecule[start:stop:step]` depending on the value of **inplace**.

        Parameters
        ----------
        start : int
            Start of the interval.

        stop : int
            End of the interval.

        inplace : bool
            Instead of returning the new coordinates, perform an inplace update of this instance.

        Returns
        -------
        |None|_ or |FOX.MultiMolecule|_:
            If **inplace** is ``True``, return a new :class:`MultiMolecule` instance.

        """
        if inplace:
            self[:] = self[start:stop:step]
            return None
        else:
            return self[start:stop:step].copy()

    def random_slice(self, start: int = 0,
                     stop: Union[int, None] = None,
                     p: float = 0.5,
                     inplace: bool = False) -> Optional[MultiMolecule]:
        """Construct a new :class:`MultiMolecule` instance by randomly slicing this instance.

        The probability of including a particular element is equivalent to **p**.

        Parameters
        ----------
        start : int
            Start of the interval.

        stop : int
            End of the interval.

        p : float
            The probability of including each particular molecule in this instance.
            Values must be between ``0.0`` (0%) and ``1.0`` (100%).

        inplace : bool
            Instead of returning the new coordinates, perform an inplace update of this instance.

        Returns
        -------
        |None|_ or |FOX.MultiMolecule|_:
            If **inplace** is ``True``, return a new :class:`MultiMolecule` instance.

        Raises
        ------
        ValueError
            Raised if **p** is smaller than ``0.0`` or larger than ``1.0``.

        """
        if p <= 0.0 or p >= 1.0:
            raise ValueError("The supplied probability, 'p': {:f}, must be larger "
                             "than 0.0 and smaller than 1.0".format(p))

        stop = stop or self.shape[0]
        idx_range = np.arange(start, stop)
        size = int(p * len(idx_range))
        idx = np.random.choice(idx_range, size=size, replace=False)

        if inplace:
            self[:] = self[idx]
            return None
        else:
            return self[idx].copy()

    def reset_origin(self, mol_subset: MolSubset = None,
                     atom_subset: AtomSubset = None,
                     inplace: bool = True) -> Optional[MultiMolecule]:
        """Reallign all molecules in this instance.

        All molecules in this instance are rotating and translating, by performing a partial partial
        Procrustes superimposition with respect to the first molecule in this instance.

        The superimposition is carried out with respect to the first molecule in this instance.

        Parameters
        ----------
        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        inplace : bool
            Instead of returning the new coordinates, perform an inplace update of this instance.

        Returns
        -------
        |None|_ or |FOX.MultiMolecule|_:
            If **inplace** is ``True``, return a new :class:`MultiMolecule` instance.

        """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Remove translations
        coords = self[i, j, :] - self[i, j, :].mean(axis=1)[:, None, :]

        # Peform a singular value decomposition on the covariance matrix
        H = np.swapaxes(coords[0:], 1, 2) @ coords[0]
        U, _, Vt = np.linalg.svd(H)
        V, Ut = np.swapaxes(Vt, 1, 2), np.swapaxes(U, 1, 2)

        # Construct the rotation matrix
        rotmat = np.ones_like(U)
        rotmat[:, 2, 2] = np.linalg.det(V @ Ut)
        rotmat *= V@Ut

        # Return or perform an inplace update of this instance
        if inplace:
            self[i, j, :] = coords @ np.swapaxes(rotmat, 1, 2)
            return None
        else:
            return coords @ rotmat

    def sort(self, sort_by: Union[str, Sequence[int]] = 'symbol',
             reverse: bool = False,
             inplace: bool = True) -> Optional[MultiMolecule]:
        """Sort the atoms in this instance and **self.atoms**, performing in inplace update.

        Parameters
        ----------
        sort_by : |str|_ or |Sequence|_ [|int|_]
            The property which is to be used for sorting.
            Accepted values: ``"symbol"`` (*i.e.* alphabetical), ``"atnum"``, ``"mass"``,
            ``"radius"`` or ``"connectors"``.
            See the plams.PeriodicTable_ module for more details.
            Alternatively, a user-specified sequence of indices can be provided for sorting.

        reverse : bool
            Sort in reversed order.

        inplace : bool
            Instead of returning the new coordinates, perform an inplace update of this instance.

        Returns
        -------
        |None|_ or |FOX.MultiMolecule|_:
            If **inplace** is ``True``, return a new :class:`MultiMolecule` instance.

        """
        # Create and, potentially, sort a list of indices
        if isinstance(sort_by, str):
            sort_by_array = self._get_atomic_property(prop=sort_by)
            _idx_range = range(self.shape[0])
            idx_range = np.array([i for _, i in sorted(zip(sort_by_array, _idx_range))])
        else:
            idx_range = np.asarray(sort_by)
            assert sort_by.shape[0] == self.shape[1]

        # Reverse or not
        if reverse:
            idx_range.reverse()

        # Inplace update or return a copy
        if inplace:
            mol = self
        else:
            mol = self.copy()

        # Sort this instance
        mol[:] = mol[:, idx_range]

        # Refill **self.atoms**
        symbols = mol.symbol[idx_range]
        mol.atoms = {}
        for i, at in enumerate(symbols):
            try:
                mol.atoms[at].append(i)
            except KeyError:
                mol.atoms[at] = [i]

        # Sort **self.bonds**
        if mol.bonds is not None:
            mol.atom1 = idx_range[mol.atom1]
            mol.atom2 = idx_range[mol.atom2]
            mol.bonds[:, 0:2].sort(axis=1)
            idx = mol.bonds[:, 0:2].argsort(axis=0)[:, 0]
            mol.bonds = mol.bonds[idx]

        # Inplace update or return a copy
        if inplace:
            return None
        else:
            return mol

    def residue_argsort(self, concatenate: bool = True) -> Union[List[List[int]], np.ndarray]:
        """Return the indices that would sort this instance by residue number.

        Residues are defined based on moleculair fragments based on **self.bonds**.

        Parameters
        ----------
        concatenate : bool
            If ``False``, returned a nested list with atomic indices.
            Each sublist contains the indices of a single residue.

        Returns
        -------
        :math:`n` |np.ndarray|_ [|np.int64|_]:
            A 1D array of indices that would sort :math:`n` atoms this instance.

        """
        # Define residues
        plams_mol = self.as_Molecule(mol_subset=0)[0]
        frags = plams_mol.separate_mod()
        symbol = self.symbol

        # Sort the residues
        core = []
        ligands = []
        for frag in frags:
            if len(frag) == 1:
                core += frag
            else:
                i = np.array(frag)
                argsort = np.argsort(symbol[i])
                ligands.append(i[argsort].tolist())
        core.sort()
        ligands.sort()

        ret = [core] + ligands
        if concatenate:
            return np.concatenate(ret)
        return ret

    def get_center_of_mass(self, mol_subset: MolSubset = None,
                           atom_subset: AtomSubset = None) -> np.ndarray:
        """Get the center of mass.

        Parameters
        ----------
        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        Returns
        -------
        :math:`m*3` |np.ndarray|_ [|np.float64|_]:
            A 2D array with the centres of mass of :math:`m` molecules with :math:`n` atoms.

        """
        coords = self.as_mass_weighted(mol_subset, atom_subset)
        return coords.sum(axis=1) / self.mass.sum()

    def get_bonds_per_atom(self, atom_subset: AtomSubset = None) -> np.ndarray:
        """Get the number of bonds per atom in this instance.

        Parameters
        ----------
        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        Returns
        -------
        :math:`n` |np.ndarray|_ [|np.int64|_]:
            A 1D array with the number of bonds per atom, for all :math:`n` atoms in this instance.

        """
        j = self._get_atom_subset(atom_subset, as_sequence=True)
        if self.bonds is None:
            return np.zeros(len(j), dtype=int)
        return np.bincount(self.bonds[:, 0:2].flatten(), minlength=self.shape[1])[j]

    """################################## Root Mean Squared ################################## """

    def _get_time_averaged_prop(self, method: Callable,
                                atom_subset: AtomSubset = None,
                                kwarg: Dict[str, Any] = {}) -> pd.DataFrame:
        """A method for constructing time-averaged properties.

        Parameters
        ----------
        method : |Callable|_
            A function, method or class used for constructing a specific time-averaged property.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        kwarg : dict [str, object]
            Keyword arguments that will be supplied to **method**.

        Returns
        -------
        |pd.DataFrame|_:
            A DataFrame containing a time-averaged property.

        """
        # Prepare arguments
        loop, at_subset = self._get_loop(atom_subset)

        # Get the time-averaged property
        if loop:
            data = [method(atom_subset=at, **kwarg) for at in at_subset]
        else:
            data = method(atom_subset=at_subset, **kwarg)

        # Construct and return the dataframe
        idx_range = np.arange(0, self.shape[1])
        idx = pd.Index(idx_range, name='Abritrary atomic index')
        column_range, data = self._get_rmsf_columns(data, idx, loop=loop, atom_subset=at_subset)
        columns = pd.Index(column_range, name='Atoms')
        return pd.DataFrame(data, index=idx, columns=columns)

    def _get_average_prop(self, method: Callable,
                          atom_subset: AtomSubset = None,
                          kwarg: Dict[str, Any] = {}) -> pd.DataFrame:
        """A method for constructing properties averaged over atomic subsets.

        Parameters
        ----------
        Method : |Callable|_
            A function, method or class used for constructing a specific atomic subset-averaged
            property.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        kwarg : dict [str, object]
            Keyword arguments that will be supplied to **method**.

        Returns
        -------
        |pd.DataFrame|_:
            A DataFrame containing an atomic subset-averaged property.

        """
        # Prpare arguments
        loop, at_subset = self._get_loop(atom_subset)

        # Calculate and averaged property
        if loop:
            data = np.array([method(atom_subset=at, **kwarg) for at in at_subset]).T
        else:
            data = method(atom_subset=atom_subset, **kwarg).T

        # Construct and return the dataframe
        column_range = self._get_rmsd_columns(loop, atom_subset)
        columns = pd.Index(column_range, name='Atoms')
        return pd.DataFrame(data, columns=columns)

    def init_average_velocity(self, timestep: float = 1.0,
                              rms: bool = False,
                              mol_subset: MolSubset = None,
                              atom_subset: AtomSubset = None) -> pd.DataFrame:
        """Calculate the average atomic velocty.

        The average velocity (in fs/A) is calculated for all atoms in **atom_subset** over the
        course of a trajectory.

        The velocity is averaged over all atoms in a particular atom subset.

        Parameters
        ----------
        timestep : float
            The stepsize, in femtoseconds, between subsequent frames.

        rms : bool
            Calculate the root-mean squared average velocity instead.

        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if
            ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        Returns
        -------
        |pd.DataFrame|_:
            A dataframe holding :math:`m-1` velocities averaged over one or more atom subsets.

        """
        kwarg = {'mol_subset': mol_subset, 'timestep': timestep, 'rms': rms}
        df = self._get_average_prop(self.get_average_velocity, atom_subset, kwarg)
        df.index.name = 'Time / fs'
        return df

    def init_time_averaged_velocity(self, timestep: float = 1.0,
                                    rms: bool = False,
                                    mol_subset: MolSubset = None,
                                    atom_subset: AtomSubset = None) -> pd.DataFrame:
        """Calculate the time-averaged velocty.

        The time-averaged velocity (in fs/A) is calculated for all atoms in **atom_subset** over the
        course of a trajectory.

        Parameters
        ----------
        timestep : float
            The stepsize, in femtoseconds, between subsequent frames.

        rms : bool
            Calculate the root-mean squared time-averaged velocity instead.

        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        Returns
        -------
        |pd.DataFrame|_:
            A dataframe holding :math:`m-1` time-averaged velocities.

        """
        kwarg = {'mol_subset': mol_subset, 'timestep': timestep, 'rms': rms}
        return self._get_time_averaged_prop(self.get_time_averaged_velocity, atom_subset, kwarg)

    def init_rmsd(self, mol_subset: MolSubset = None,
                  atom_subset: AtomSubset = None,
                  reset_origin: bool = True) -> pd.DataFrame:
        """Initialize the RMSD calculation, returning a dataframe.

        Parameters
        ----------
            mol_subset : slice
                Perform the calculation on a subset of molecules in this instance, as
                determined by their moleculair index.
                Include all :math:`m` molecules in this instance if ``None``.

            atom_subset : |Sequence|_
                Perform the calculation on a subset of atoms in this instance, as
                determined by their atomic index or atomic symbol.
                Include all :math:`n` atoms per  molecule in this instance if ``None``.

            reset_origin : bool
                Reset the origin of each molecule in this instance by means of
                a partial Procrustes superimposition, translating and rotating the molecules.

        Returns
        -------
        |pd.DataFrame|_:
            A dataframe of RMSDs with one column for every string or list of ints in
            **atom_subset**.
            Keys consist of atomic symbols (*e.g.* ``"Cd"``) if **atom_subset** contains strings,
            otherwise a more generic 'series ' + str(int) scheme is adopted (*e.g.* ``"series 2"``).
            Molecular indices are used as index.

        """
        if reset_origin:
            self.reset_origin()
        kwarg = {'mol_subset': mol_subset}
        df = self._get_average_prop(self.get_rmsd, atom_subset, kwarg)
        df.index.name = 'XYZ frame number'
        return df

    def init_rmsf(self, mol_subset: MolSubset = None,
                  atom_subset: AtomSubset = None,
                  reset_origin: bool = True) -> pd.DataFrame:
        """Initialize the RMSF calculation, returning a dataframe.

        Parameters
        ----------
        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        reset_origin : bool
            Reset the origin of each molecule in this instance by means of
            a partial Procrustes superimposition, translating and rotating the molecules.

        Returns
        -------
        |pd.DataFrame|_:
            A dataframe of RMSFs with one column for every string or list of ints in
            **atom_subset**.
            Keys consist of atomic symbols (*e.g.* ``"Cd"``) if **atom_subset** contains strings,
            otherwise a more generic 'series ' + str(int) scheme is adopted (*e.g.* ``"series 2"``).
            Molecular indices are used as indices.

        """
        if reset_origin:
            self.reset_origin()
        kwarg = {'mol_subset': mol_subset}
        return self._get_time_averaged_prop(self.get_rmsf, atom_subset, kwarg)

    def get_average_velocity(self, timestep: float = 1.0,
                             rms: bool = False,
                             mol_subset: MolSubset = None,
                             atom_subset: AtomSubset = None) -> np.ndarray:
        """Return the mean or root-mean squared velocity.

        Parameters
        ----------
        timestep : float
            The stepsize, in femtoseconds, between subsequent frames.

        rms : bool
            Calculate the root-mean squared average velocity instead.

        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if
            ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        Returns
        -------
        :math:`m-1` |np.ndarray|_ [|np.float64|_]:
            A 1D array holding :math:`m-1` velocities averaged over one or more atom subsets.

        """
        if not rms:
            return self.get_velocity(timestep, mol_subset=mol_subset,
                                     atom_subset=atom_subset).mean(axis=1)
        else:
            v = self.get_velocity(timestep, mol_subset=mol_subset, atom_subset=atom_subset)
            return MultiMolecule(v, self.atoms).get_rmsd(mol_subset)

    def get_time_averaged_velocity(self, timestep: float = 1.0,
                                   rms: bool = False,
                                   mol_subset: MolSubset = None,
                                   atom_subset: AtomSubset = None) -> np.ndarray:
        """Return the mean or root-mean squared velocity (mean = time-averaged).

        Parameters
        ----------
        timestep : float
            The stepsize, in femtoseconds, between subsequent frames.

        rms : bool
            Calculate the root-mean squared average velocity instead.

        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        Returns
        -------
        :math:`n` |np.ndarray|_ [|np.float64|_]:
            A 1D array holding :math:`n` time-averaged velocities.

        """
        if not rms:
            return self.get_velocity(timestep, mol_subset=mol_subset,
                                     atom_subset=atom_subset).mean(axis=0)
        else:
            v = self.get_velocity(timestep, mol_subset=mol_subset, atom_subset=atom_subset)
            return MultiMolecule(v, self.atoms).get_rmsf(mol_subset)

    def get_velocity(self, timestep: float = 1.0,
                     norm: bool = True,
                     mol_subset: MolSubset = None,
                     atom_subset: AtomSubset = None) -> np.ndarray:
        """Return the atomic velocties.

        The velocity (in fs/A) is calculated for all atoms in **atom_subset** over the course of a
        trajectory.

        Parameters
        ----------
        timestep : float
            The stepsize, in femtoseconds, between subsequent frames.

        norm : bool
            If ``True`` return the norm of the :math:`x`, :math:`y` and :math:`z`
            velocity components.

        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        Returns
        -------
        :math:`m*n` or :math:`m*n*3` |np.ndarray|_ [|np.float64|_]:
            A 2D or 3D array of atomic velocities, the number of dimensions depending on the
            value of **norm** (``True`` = 2D; ``False`` = 3D).

        """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Slice the XYZ array and reset the origin
        xyz = self[i, j]
        xyz.reset_origin()

        if norm:
            return np.gradient(np.linalg.norm(xyz, axis=2), timestep, axis=0)
        else:
            return np.gradient(xyz, timestep, axis=0)

    def get_rmsd(self, mol_subset: MolSubset = None,
                 atom_subset: AtomSubset = None) -> np.ndarray:
        """Calculate the root mean square displacement (RMSD).

        The RMSD is calculated with respect to the first molecule in this instance.

        Parameters
        ----------
        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        Returns
        -------
        |pd.DataFrame|_:
            A dataframe with the RMSD as a function of the XYZ frame numbers.

        """
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Calculate and return the RMSD per molecule in this instance
        dist = np.linalg.norm(self[i, j, :] - self[0, j, :], axis=2)
        return np.sqrt(np.einsum('ij,ij->i', dist, dist) / dist.shape[1])

    def get_rmsf(self, mol_subset: MolSubset = None,
                 atom_subset: AtomSubset = None) -> np.ndarray:
        """Calculate the root mean square fluctuation (RMSF).

        Parameters
        ----------
        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        Returns
        -------
        |pd.DataFrame|_:
            A dataframe with the RMSF as a function of atomic indices.

        """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Calculate the RMSF per molecule in this instance
        mean_coords = np.mean(self[i, j, :], axis=0)[None, ...]
        displacement = np.linalg.norm(self[i, j, :] - mean_coords, axis=2)**2
        return np.mean(displacement, axis=0)

    @staticmethod
    def _get_rmsd_columns(loop: bool,
                          atom_subset: AtomSubset = None) -> Sequence[Hashable]:
        """Return the columns for the RMSD dataframe.

        Parameters
        ----------
        loop : bool
            If ``True``, return a single column name.
            If ``False``, return a sequence with multiple column names.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        Returns
        -------
        Sequence[Hashable]:
            A sequence with column names for atomic subset-averaged properties.

        """
        if loop:  # Plan A: **atom_subset** is a *list* of *str* or nested *list* of *int*
            if isinstance(atom_subset[0], str):  # Use atomic symbols or general indices as keys
                columns = atom_subset
            else:
                columns = np.arange(len(atom_subset))
        else:  # Plan B: **atom_subset** is something else
            if isinstance(atom_subset, str):  # Use an atomic symbol or a general index as keys
                columns = [atom_subset]
            else:
                columns = [0]

        return columns

    def _get_rmsf_columns(self, rmsf: np.ndarray,
                          index: Sequence[Hashable],
                          loop: bool,
                          atom_subset: AtomSubset = None
                          ) -> Tuple[Sequence[Hashable], np.ndarray]:
        """Return the columns and data for the RMSF dataframe.

        Parameters
        ----------
        rmsf : |np.ndarray|_ [|np.float64|_]
            An array with a time-veraged property.

        index : |Sequence|_
            The index for the time-averaged property.

        loop : bool
            If ``True``, return a single column name.
            If ``False``, return a sequence with multiple column names.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        Returns
        -------
        Sequence[Hashable] and |np.ndarray|_ [|np.float64|_]:
            A sequence with column names for time-averaged properties
            and an array with the time-averaged data.

        """
        if loop:  # Plan A: **atom_subset** is a *list* of *str* or nested *list* of *int*
            if isinstance(atom_subset[0], str):  # Use atomic symbols or general indices as keys
                columns = atom_subset
            else:
                columns = np.arange(len(atom_subset))

            # Create and fill a padded array
            data = np.full((len(rmsf), index.shape[0]), np.nan)
            k = 0
            for i, j in enumerate(rmsf):
                data[i, k:k+len(j)] = j
                k += len(j)
            data = data.T

        else:  # Plan B: **atom_subset** is something else
            if isinstance(atom_subset, str):  # Use an atomic symbol or a general index as keys
                columns = [atom_subset]
            else:
                columns = [0]

            # Create and fill a padded array
            data = np.full((index.shape[0]), np.nan)
            idx = self._get_atom_subset(atom_subset)
            data[idx] = rmsf

        return columns, data

    def _get_loop(self, atom_subset: AtomSubset) -> Tuple[bool, AtomSubset]:
        """Figure out if the supplied subset warrants a for loop or not.

        Parameters
        ----------
        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        Returns
        -------
        bool and |np.ndarray|_ [|np.float64|_]:
            A boolean and (nested) iterable consisting of integers.

        Raises
        ------
        TypeError
            Raised if **atom_subset** is of an invalid type.

        """
        if atom_subset is None:
            return False, slice(None)
        elif isinstance(atom_subset, (range, slice)):
            return False, atom_subset
        elif isinstance(atom_subset, str):
            return False, self.atoms[atom_subset]
        elif isinstance(atom_subset, int):
            return False, [atom_subset]
        elif isinstance(atom_subset[0], (int, np.integer)):
            return False, atom_subset
        elif isinstance(atom_subset[0], str):
            return True, [self.atoms[i] for i in atom_subset]
        elif isinstance(atom_subset[0][0], (int, np.integer)):
            return True, atom_subset

        err = "'{}' of type '{}' is an invalid argument for 'atom_subset'"
        raise TypeError(err.format(str(atom_subset), atom_subset.__class__.__name__))

    """#############################  Determining shell structures  ######################### """

    def init_shell_search(self, mol_subset: MolSubset = None,
                          atom_subset: AtomSubset = None,
                          rdf_cutoff: float = 0.5
                          ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Calculate and return properties which can help determining shell structures.

        The following two properties are calculated and returned:

        * The mean distance (per atom) with respect to the center of mass (*i.e.* a modified RMSF).
        * A series mapping abritrary atomic indices in the RMSF to the actual atomic indices.
        * The radial distribution function (RDF) with respect to the center of mass.

        Parameters
        ----------
        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        rdf_cutoff : float
            Remove all values in the RDF below this value (Angstrom).
            Usefull for dealing with divergence as the "inter-atomic" distance approaches 0.0 A.

        Returns
        -------
        |pd.DataFrame|_, |pd.Series|_ and |pd.DataFrame|_:
            Returns the following items:
                * A dataframe holding the mean distance of all atoms with respect the to center
                  of mass.

                * A series mapping the indices from 1. to the actual atomic indices.

                * A dataframe holding the RDF with respect to the center of mass.

        """
        def _get_mean_dist(mol_cp, at):
            ret = np.linalg.norm(mol_cp[:, mol_cp.atoms[at]], axis=2).mean(axis=0)
            at_idx = np.argsort(ret)
            return at_idx, sorted(ret)

        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        at_subset = atom_subset or tuple(self.atoms.keys())

        # Calculate the mean distance (per atom) with respect to the center of mass
        # Conceptually similar an RMSF, the "fluctuation" being with respect to the center of mass
        dist_mean = []
        mol_cp = self.copy()[i]
        mol_cp -= mol_cp.get_center_of_mass()[:, None, :]
        at_idx, dist_mean = zip(*[_get_mean_dist(mol_cp, at) for at in at_subset])

        # Create Series containing the actual atomic indices
        at_idx = list(chain.from_iterable(at_idx))
        idx_series = pd.Series(np.arange(0, self.shape[1]), name='Actual atomic index')
        idx_series.loc[0:len(at_idx)-1] = at_idx
        idx_series.index.name = 'Arbitrary atomic index'

        # Cast the modified RMSF results in a dataframe
        index = np.arange(0, self.shape[1])
        kwarg = {'loop': True, 'atom_subset': at_subset}
        columns, data = mol_cp._get_rmsf_columns(dist_mean, index, **kwarg)
        rmsf = pd.DataFrame(data, columns=columns, index=index)
        rmsf.columns.name = 'Distance from origin\n  /  Ångström'
        rmsf.index.name = 'Arbitrary atomic index'

        # Calculate the RDF with respect to the center of mass
        at_dummy = np.zeros_like(mol_cp[:, 0, :])[:, None, :]
        mol_cp = MultiMolecule(np.hstack((mol_cp, at_dummy)), atoms=mol_cp.atoms)
        mol_cp.atoms['origin'] = [mol_cp.shape[1] - 1]
        at_subset = ('origin', ) + at_subset
        with np.errstate(divide='ignore', invalid='ignore'):
            rdf = mol_cp.init_rdf(at_subset)
        del rdf['origin origin']
        rdf = rdf.loc[rdf.index >= rdf_cutoff, [i for i in rdf.columns if 'origin' in i]]

        return rmsf, idx_series, rdf

    @staticmethod
    def get_at_idx(rmsf: pd.DataFrame,
                   idx_series: pd.Series,
                   dist_dict: Dict[str, List[float]]) -> Dict[str, List[int]]:
        """Create subsets of atomic indices.

        The subset is created (using **rmsf** and **idx_series**) based on
        distance criteria in **dist_dict**.

        For example, ``dist_dict = {'Cd': [3.0, 6.5]}`` will create and return a dictionary with
        three keys: One for all atoms whose RMSF is smaller than 3.0, one where the RMSF is
        between 3.0 and 6.5, and finally one where the RMSF is larger than 6.5.

        Examples
        --------
        .. code:: python

            >>> dist_dict = {'Cd': [3.0, 6.5]}
            >>> idx_series = pd.Series(np.arange(12))
            >>> rmsf = pd.DataFrame({'Cd': np.arange(12, dtype=float)})
            >>> get_at_idx(rmsf, idx_series, dist_dict)

            {'Cd_1': [0, 1, 2],
             'Cd_2': [3, 4, 5],
             'Cd_3': [7, 8, 9, 10, 11]
            }

        Parameters
        ----------
        rmsf : |pd.DataFrame|_
            A dataframe holding the results of an RMSF calculation.

        idx_series : |pd.Series|_
            A series mapping the indices from **rmsf** to actual atomic indices.

        dist_dict : dict [str, list [float]]
            A dictionary with atomic symbols (see **rmsf.columns**)
            and a list of interatomic distances.

        Returns
        -------
        |dict|_ [|str|_, |list|_ [|int|_]]
            A dictionary with atomic symbols as keys, and matching atomic indices as values.

        Raises
        ------
        KeyError
            Raised if a key in **dist_dict** is absent from **rmsf**.

        """
        # Double check if all keys in **dist_dict** are available in **rmsf.columns**
        for key in dist_dict:
            if key not in rmsf:
                err = "'{}' was found in 'dist_dict' yet is absent from 'rmsf'"
                raise KeyError(err.format(key))

        ret = {}
        for key, value in rmsf.items():
            try:
                dist_range = sorted(dist_dict[key])
            except KeyError:
                dist_range = [np.inf]
            dist_min = 0.0
            name = key + '_{:d}'

            for i, dist_max in enumerate(dist_range, 1):
                idx = rmsf[(value >= dist_min) & (value < dist_max)].index
                if idx.any():
                    ret[name.format(i)] = sorted(idx_series[idx].values.tolist())
                dist_min = dist_max

            idx = rmsf[(rmsf[key] > dist_max)].index
            if idx.any():
                ret[name.format(i+1)] = sorted(idx_series[idx].values.tolist())
        return ret

    """#############################  Radial Distribution Functions  ######################### """

    def init_rdf(self, atom_subset: AtomSubset = None,
                 dr: float = 0.05,
                 r_max: float = 12.0,
                 low_mem: bool = False):
        """Initialize the calculation of radial distribution functions (RDFs).

        RDFs are calculated for all possible atom-pairs in **atom_subset** and returned as a
        dataframe.

        Parameters
        ----------
        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        dr : float
            The integration step-size in Ångström, *i.e.* the distance between
            concentric spheres.

        r_max : float
            The maximum to be evaluated interatomic distance in Ångström.

        low_mem : bool
            If ``True``, use a slower but more memory efficient method for constructing the RDFs.

        Returns
        -------
        |pd.DataFrame|_:
            A dataframe of radial distribution functions, averaged over all conformations in
            **xyz_array**.
            Keys are of the form: at_symbol1 + ' ' + at_symbol2 (*e.g.* ``"Cd Cd"``).
            Radii are used as index.

        """
        # If **atom_subset** is None: extract atomic symbols from they keys of **self.atoms**
        at_subset = atom_subset or tuple(sorted(self.atoms.keys(), key=str))
        atom_pairs = self.get_pair_dict(at_subset, r=2)

        # Construct an empty dataframe with appropiate dimensions, indices and keys
        df = get_rdf_df(atom_pairs, dr, r_max)
        kwarg = {'dr': dr, 'r_max': r_max}

        # Fill the dataframe with RDF's, averaged over all conformations in this instance
        if low_mem:  # Slower low memory approach
            for i in range(self.shape[0]):
                for key, at in atom_pairs.items():
                    dist_mat = self.get_dist_mat(mol_subset=i, atom_subset=at)
                    df[key] += get_rdf_lowmem(dist_mat, **kwarg)
            df.loc[0.0] = 0.0
            df /= self.shape[0]

        else:  # Faster high memory approach
            for key, at in atom_pairs.items():
                dist_mat = self.get_dist_mat(atom_subset=at)
                df[key] = get_rdf(dist_mat, **kwarg)

        return df

    def get_dist_mat(self, mol_subset: MolSubset = None,
                     atom_subset: Tuple[AtomSubset] = (None, None)) -> np.ndarray:
        """Create and return a distance matrix for all molecules and atoms in this instance.

        Parameters
        ----------
        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        Returns
        -------
        :math:`m*n*k` |np.ndarray|_ [|np.float64|_]:
            A 3D distance matrix of :math:`m` molecules, created out of two sets of :math:`n`
            and :math:`k` atoms.

        """
        # Define array slices
        m_subset = self._get_mol_subset(mol_subset)
        i = m_subset, self._get_atom_subset(atom_subset[0])
        j = m_subset, self._get_atom_subset(atom_subset[1])

        # Slice the XYZ array
        A = self[i]
        B = self[j]

        # Create, fill and return the distance matrix
        if A.ndim == 2:
            return cdist(A, B)[None, ...]
        shape = A.shape[0], A.shape[1], B.shape[1]
        ret = np.empty(shape)
        for k, (a, b) in enumerate(zip(A, B)):
            ret[k] = cdist(a, b)
        return ret

    @staticmethod
    def get_pair_dict(atom_subset: AtomSubset,
                      r: int = 2) -> Dict[str, str]:
        """Take a subset of atoms and return a dictionary.

        Parameters
        ----------
        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        r : int
            The length of the to-be returned subsets.

        """
        values = list(combinations_with_replacement(atom_subset, r))

        if not isinstance(atom_subset[0], str):
            str_ = 'series' + ''.join(' {:d}' for _ in values[0])
            return {str_.format(*[i.index(j) for j in i]): i for i in values}

        else:
            str_ = ''.join(' {}' for _ in values[0])[1:]
            return {str_.format(*i): i for i in values}

    """############################  Angular Distribution Functions  ######################### """

    def init_power_spectrum(self, mol_subset: MolSubset = None,
                            atom_subset: AtomSubset = None,
                            freq_max: int = 4000) -> pd.DataFrame:
        """Calculate and return the power spectrum associated with this instance.

        Parameters
        ----------
        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        freq_max : |int|_
            The maximum to be returned wavenumber (cm**-1).

        Returns
        -------
        |pd.DataFrame|_
            A DataFrame containing the power spectrum for each set of atoms in
            **atom_subset**.

        """
        # Get atomic velocities
        v = self.get_velocity(1e-15, mol_subset=mol_subset, atom_subset=atom_subset)  # A / s

        # Create the to-be returned DataFrame
        freq_max = int(freq_max) + 1
        idx = pd.RangeIndex(0, freq_max, name='Wavenumber / cm**-1')
        df = pd.DataFrame(index=idx)

        # Construct the velocity autocorrelation function
        vacf = signal.fftconvolve(v, v[::-1], axes=0)[len(v)-1:]
        dv = v - v.mean(axis=0)
        vacf /= np.einsum('ij,ij->j', dv, dv)

        # Construct power spectra intensities
        n = int(1 / (constants.c * 1e-13))
        power_complex = fft(vacf, n, axis=0) / len(vacf)
        power_abs = np.abs(power_complex)
        for at, idx in self.atoms.items():
            slice_ = power_abs[:, idx]
            df[at] = np.einsum('ij,ij->i', slice_, slice_)[:freq_max]

    def init_adf(self, mol_subset: MolSubset = None,
                 atom_subset: AtomSubset = None,
                 r_max: Union[float, str] = 8.0,
                 distance_weighted: bool = True) -> pd.DataFrame:
        r"""Initialize the calculation of distance-weighted angular distribution functions (ADFs).

        ADFs are calculated for all possible atom-pairs in **atom_subset** and returned as a
        dataframe.

        .. _DASK: https://dask.org/

        Parameters
        ----------
        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        r_max : |float|_ or |str|_
            The maximum inter-atomic distance (in Angstrom) for which angles are constructed.
            The distance cuttoff can be disabled by settings this value to ``np.inf``, ``"np.inf"``
            or ``"inf"``.

        distance_weighted : |bool|_
            Return the distance-weighted angular distribution function.
            Every angle the to-be constructed angle matrix, :math:`\phi_{ijk}`, is weighted by the
            distance according to the weighting factor :math:`v_{ijk}`:

            .. math::

                v_{ijk} = \Biggl \lbrace
                {
                    e^{-r_{ji}}, \quad r_{ji} \; \gt \; r_{jk}
                    \atop
                    e^{-r_{jk}}, \quad r_{ji} \; \lt \; r_{jk}
                }

        Returns
        -------
        |pd.DataFrame|_
            A dataframe of angular distribution functions, averaged over all conformations in
            this instance.

        Note
        ----
        Disabling the distance cuttoff is strongly recommended (*i.e.* it is faster) for large
        values of **r_max**. As a rough guideline, ``r_max="inf"`` is roughly as fast as
        ``r_max=15.0`` (though this is, of course, system dependant).

        Note
        ----
        The ADF construction will be conducted in parralel if the DASK_ package is installed.
        DASK can be installed, via anaconda, with the following command:
        ``conda install -n FOX -y -c conda-forge dask``.

        """
        # Identify the maximum to-be considered radius
        if r_max in (np.inf, 'np.inf', 'inf'):
            r_max = 0.0
        n = int((1 + r_max / 4.0)**3)

        # Identify atom and molecule subsets
        m_subset = self._get_mol_subset(mol_subset)
        at_subset = np.asarray(self._get_atom_subset(atom_subset, as_sequence=True))

        # Construct a dictionary with atom counts and unique atom-pair identifiers
        atom_pairs = self.get_pair_dict(atom_subset or list(self.atoms), r=3)
        atnum_dict = {}
        get_atnum = PeriodicTable.get_atomic_number  # method alias
        for at1, at2, at3 in atom_pairs.values():
            key = (get_atnum(at1) + get_atnum(at3)) * get_atnum(at2)
            atnum_dict[key] = len(self.atoms[at2])

        # Slice this MultiMolecule instance based on **atom_subset** and **mol_subset**
        del_atom = np.arange(0, self.shape[1])[~at_subset]
        mol = self.delete_atoms(del_atom)[m_subset]
        atnum = mol.atnum

        # Construct the angular distribution function
        # Perform the task in parallel (with dask) if possible
        if DASK and r_max:
            func = dask.delayed(MultiMolecule._adf_inner_cdktree)
            jobs = [func(m, n, r_max, atnum_dict, atnum, distance_weighted) for m in mol]
            results = dask.compute(*jobs)
        elif DASK and not r_max:
            func = dask.delayed(MultiMolecule._adf_inner)
            jobs = [func(m, atnum_dict, atnum, distance_weighted) for m in mol]
            results = dask.compute(*jobs)
        elif not DASK and r_max:
            func = MultiMolecule._adf_inner_cdktree
            results = [func(m, n, r_max, atnum_dict, atnum, distance_weighted) for m in mol]
        elif not DASK and not r_max:
            func = MultiMolecule._adf_inner
            results = [func(m, atnum_dict, atnum, distance_weighted) for m in mol]

        df = get_adf_df(atom_pairs)
        df.loc[:, :] = np.array(results).mean(axis=0)
        return df

    @staticmethod
    def _adf_inner_cdktree(m: MultiMolecule,
                           n: int,
                           r_max: float,
                           atnum_dict: Dict[int, int],
                           atnum: np.ndarray,
                           distance_weighted: bool) -> np.ndarray:
        """Perform the loop of :meth:`.init_adf` with a distance cutoff."""
        # Construct slices and a distance matrix
        tree = cKDTree(m)
        dist, idx = tree.query(m, n, distance_upper_bound=r_max, p=2)
        dist[dist == np.inf] = 0.0
        idx[idx == m.shape[0]] = 0

        # Slice the Cartesian coordinates
        coords1 = coords3 = m[idx]
        coords2 = m[..., None, :]

        # Construct (3D) angle- and distance-matrices
        with np.errstate(divide='ignore', invalid='ignore'):
            vec12 = ((coords1 - coords2) / dist[..., None])
            vec32 = ((coords3 - coords2) / dist[..., None])
            ang = np.arccos(np.einsum('jkl,jml->jkm', vec12, vec32))
            dist = np.exp(-np.maximum(dist[..., None], dist[..., None, :]))
        ang[np.isnan(ang)] = 0.0

        # Create an array of (unique) atom-pair identifiers
        atnum_ar = atnum[:, None, None] * (atnum[idx][..., None] + atnum[idx][..., None, :])

        # Construct and return the ADF
        return get_adf(ang, dist, atnum_ar, atnum_dict, distance_weighted)

    @staticmethod
    def _adf_inner(m: MultiMolecule,
                   atnum_dict: Dict[int, int],
                   atnum: np.ndarray,
                   distance_weighted: bool) -> np.ndarray:
        """Perform the loop of :meth:`.init_adf` without a distance cutoff."""
        # Construct a distance matrix
        dist = cdist(m, m)

        # Slice the Cartesian coordinates
        coords1 = coords3 = m
        coords2 = m[..., None, :]

        # Construct (3D) angle- and distance-matrices
        with np.errstate(divide='ignore', invalid='ignore'):
            vec12 = ((coords1 - coords2) / dist[..., None])
            vec32 = ((coords3 - coords2) / dist[..., None])
            ang = np.arccos(np.einsum('jkl,jml->jkm', vec12, vec32))
            dist = np.exp(-np.maximum(dist[..., None], dist[..., None, :]))
        ang[np.isnan(ang)] = 0.0

        # Create an array of (unique) atom-pair identifiers
        atnum_ar = atnum[:, None, None] * (atnum[..., None] + atnum[..., None, :])

        # Construct and return the ADF
        return get_adf(ang, dist, atnum_ar, atnum_dict, distance_weighted)

    def get_angle_mat(self, mol_subset: MolSubset = 0,
                      atom_subset: Tuple[AtomSubset, AtomSubset, AtomSubset] = (None, None, None),
                      get_r_max: bool = False) -> np.ndarray:
        """Create and return an angle matrix for all molecules and atoms in this instance.

        Parameters
        ----------
        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        get_r_max : bool
            Whether or not the maximum distance should be returned or not.

        Returns
        -------
        :math:`m*n*k*l` |np.ndarray|_ [|np.float64|_] and (optionally) |float|_
            A 4D angle matrix of :math:`m` molecules, created out of three sets of :math:`n`,
            :math:`k` and :math:`l` atoms.
            If **get_r_max** = ``True``, also return the maximum distance.

        """
        # Define array slices
        m_subset = self._get_mol_subset(mol_subset)
        i = self._get_atom_subset(atom_subset[0])
        j = self._get_atom_subset(atom_subset[1])
        k = self._get_atom_subset(atom_subset[2])

        # Slice and broadcast the XYZ array
        A = self[m_subset][:, i][..., None, :]
        B = self[m_subset][:, j][:, None, ...]
        C = self[m_subset][:, k][:, None, ...]

        # Temporary ignore RuntimeWarnings related to dividing by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            # Prepare the unit vectors
            kwarg1 = {'atom_subset': [atom_subset[0], atom_subset[1]], 'mol_subset': m_subset}
            kwarg2 = {'atom_subset': [atom_subset[0], atom_subset[2]], 'mol_subset': m_subset}
            dist_mat1 = self.get_dist_mat(**kwarg1)[..., None]
            dist_mat2 = self.get_dist_mat(**kwarg2)[..., None]
            r_max = max(dist_mat1.max(), dist_mat2.max())
            unit_vec1 = (B - A) / dist_mat1
            unit_vec2 = (C - A) / dist_mat2

            # Create and return the angle matrix
            if get_r_max:
                return np.arccos(np.einsum('ijkl,ijml->ijkm', unit_vec1, unit_vec2)), r_max
            return np.arccos(np.einsum('ijkl,ijml->ijkm', unit_vec1, unit_vec2))

    def _get_atom_subset(self, atom_subset: AtomSubset,
                         as_sequence: bool = False) -> Union[slice, range, Sequence[int]]:
        """Sanitize the **_get_atom_subset** argument.

        Accepts the following objects:

            * ``None``
            * ``range`` or ``slice`` instances
            * Integers
            * Strings (*i.e.* atom types; see the **MultiMolecule.atoms** attribute)
            * Sequence of integers (*e.g.* lists, tuples or arrays)
            * Sequence of strings
            * Nested sequence of integers

        Notes
        -----
        Supports object suitable for both fancy and non-fancy array indexing.

        Parameters
        ----------
        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        as_sequence : bool
            Ensure the subset is returned as a sequence.

        Returns
        -------
        |list|_, |np.ndarray|_, |slice|_ or |range|_:
            An object suitable for array slicing.

        Raises
        ------
        IndexError
            Raised if an object unsuitable for array slicing is provided.

        """
        if as_sequence:
            if atom_subset is None:
                return np.arange(0, self.shape[1])
            elif isinstance(atom_subset, (range, slice)):
                return np.arange(atom_subset.start, atom_subset.stop, atom_subset.step)
        else:
            if atom_subset is None:
                return slice(0, None)
            elif isinstance(atom_subset, (range, slice)):
                return atom_subset

        if isinstance(atom_subset, (int, np.integer)):
            return [atom_subset]
        elif isinstance(atom_subset[0], (int, np.integer)):
            return atom_subset
        elif isinstance(atom_subset, str):
            return self.atoms[atom_subset]
        elif isinstance(atom_subset[0], str):
            return list(chain.from_iterable(self.atoms[i] for i in atom_subset))
        elif isinstance(atom_subset[0][0], (int, np.integer)):
            return list(chain.from_iterable(atom_subset))

        err = "'{}' of type '{}' is an invalid argument for 'atom_subset'"
        raise IndexError(err.format(str(atom_subset), atom_subset.__class__.__name__))

    def _get_mol_subset(self, mol_subset: MolSubset) -> slice:
        """Sanitize the **mol_subset** argument.

        Accepts the following objects:

            * ``None``
            * ``range`` or ``slice`` instances
            * Integers

        Notes
        -----
        Objects suitable for fancy array indexing are *not* supported.

        Parameters
        ----------
        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        as_sequence : bool
            Ensure the subset is returned as a sequence.

        Returns
        -------
        |slice|_:
            An object suitable for array slicing.

        Raises
        ------
        IndexError
            Raised if an object unsuitable for array slicing is provided.

        """
        if mol_subset is None:
            return slice(0, None)
        elif isinstance(mol_subset, slice):
            return mol_subset

        elif isinstance(mol_subset, (int, np.integer)):
            if mol_subset >= 0:
                return slice(mol_subset, mol_subset+1)
            else:
                j = len(self) + 1 - mol_subset
                return slice(mol_subset, j)

        elif len(mol_subset) == 1 and isinstance(mol_subset[0], (int, np.integer)):
            i = mol_subset[0]
            if i >= 0:
                return slice(i, i+1)
            else:
                j = len(self) + 1 - i
                return slice(i, j)

        err = "'{}' of type '{}' is an invalid argument for 'mol_subset'"
        raise IndexError(err.format(str(mol_subset), mol_subset.__class__.__name__))

    """#################################  Type conversion  ################################### """

    def _mol_to_file(self, filename: str,
                     outputformat: Optional[str] = None,
                     mol_subset: Optional[MolSubset] = 0) -> None:
        """Create files using the plams.Molecule.write_ method.

        .. _plams.Molecule.write: https://www.scm.com/doc/plams/components/mol_api.html#scm.plams.mol.molecule.Molecule.write  # noqa

        Parameters
        ----------
        filename : str
            The path+filename (including extension) of the to be created file.

        outputformat : str
            The outputformat.
            Accepated values are ``"mol"``, ``"mol2"``, ``"pdb"`` or ``"xyz"``.

        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        """
        _m_subset = self._get_mol_subset(mol_subset)
        m_subset = range(_m_subset.start, _m_subset.stop, _m_subset.step)
        outputformat = outputformat or filename.rsplit('.', 1)[-1]
        plams_mol = self.as_Molecule(mol_subset=0)[0]

        if len(m_subset) != 1 or (m_subset.stop - m_subset.start) // m_subset.step != 1:
            name_list = filename.rsplit('.', 1)
            name_list.insert(-1, '.{:d}.')
            name = ''.join(name_list)
        else:
            name = filename

        for i, j in enumerate(m_subset, 1):
            plams_mol.from_array(self[j])
            plams_mol.write(name.format(i), outputformat=outputformat)

    def as_pdb(self, filename: str,
               mol_subset: Optional[MolSubset] = 0) -> None:
        """Convert a *MultiMolecule* object into one or more Protein DataBank files (.pdb).

        Utilizes the plams.Molecule.write_ method.

        .. _plams.Molecule.write: https://www.scm.com/doc/plams/components/mol_api.html#scm.plams.mol.molecule.Molecule.write  # noqa

        Parameters
        ----------
        filename : str
            The path+filename (including extension) of the to be created file.

        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        """
        self._mol_to_file(filename, 'pdb', mol_subset)

    def as_mol2(self, filename: str,
                mol_subset: Optional[MolSubset] = 0) -> None:
        """Convert a *MultiMolecule* object into one or more .mol2 files.

        Utilizes the plams.Molecule.write_ method.

        .. _plams.Molecule.write: https://www.scm.com/doc/plams/components/mol_api.html#scm.plams.mol.molecule.Molecule.write  # noqa

        Parameters
        ----------
        filename : str
            The path+filename (including extension) of the to be created file.

        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        """
        self._mol_to_file(filename, 'mol2', mol_subset)

    def as_mol(self, filename: str,
               mol_subset: Optional[MolSubset] = 0) -> None:
        """Convert a *MultiMolecule* object into one or more .mol files.

        Utilizes the plams.Molecule.write_ method.

        .. _plams.Molecule.write: https://www.scm.com/doc/plams/components/mol_api.html#scm.plams.mol.molecule.Molecule.write  # noqa

        Parameters
        ----------
        filename : str
            The path+filename (including extension) of the to be created file.

        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        """
        self._mol_to_file(filename, 'mol', mol_subset)

    def as_xyz(self, filename: str,
               mol_subset: Optional[MolSubset] = None) -> None:
        """Create an .xyz file out of this instance.

        Comments will be constructed by iteration through ``MultiMolecule.properties["comments"]``
        if the following two conditions are fulfilled:

        * The ``"comments"`` key is actually present in ``MultiMolecule.properties``.
        * ``MultiMolecule.properties["comments"]`` is an iterable.

        Parameters
        ----------
        filename : str
            The path+filename (including extension) of the to be created file.

        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        """
        # Define constants and variables
        m_subset = self[self._get_mol_subset(mol_subset)]
        at = self.symbol[:, None]
        header = '{:d}\n'.format(len(at))
        kwarg = {'fmt': ['%-10.10s', '%-15s', '%-15s', '%-15s'],
                 'delimiter': '     ',
                 'comments': ''}

        # Alter variables depending on the presence or absence of self.properties.comments
        if 'comments' in self.properties and isinstance(self.properties.comments, abc.Iterable):
            header += '{}'
            iterator = islice(zip_longest(self.properties.comments, m_subset), len(m_subset))
        else:
            header += 'Frame {:d}'
            iterator = enumerate(m_subset, 1)

        # Create the .xyz file
        with open(filename, 'wb') as file:
            for i, xyz in iterator:
                np.savetxt(file, np.hstack((at, xyz)), header=header.format(i), **kwarg)

    def as_mass_weighted(self, mol_subset: MolSubset = None,
                         atom_subset: AtomSubset = None,
                         inplace: bool = False) -> Optional[MultiMolecule]:
        """Transform the Cartesian of this instance into mass-weighted Cartesian coordinates.

        Parameters
        ----------
        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        inplace : bool
            Instead of returning the new coordinates, perform an inplace update of this instance.

        Returns
        -------
        :math:`m*n*3` |np.ndarray|_ [|np.float64|_] or |None|_:
            if **inplace** = ``False`` return a new :class:`.MultiMolecule` instance with the
            mass-weighted Cartesian coordinates of :math:`m` molecules with :math:`n` atoms.

        """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Create an array of mass-weighted Cartesian coordinates
        if inplace:
            self[i, j, :] *= self.mass[None, j, None]
            return None
        else:
            return self[i, j, :] * self.mass[None, j, None]

    def from_mass_weighted(self, mol_subset: MolSubset = None,
                           atom_subset: AtomSubset = None) -> None:
        """Transform this instance from mass-weighted Cartesian into Cartesian coordinates.

        Performs an inplace update of this instance.

        Parameters
        ----------
        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Update this instance
        self[i, j, :] /= self.mass[None, j, None]

    def as_Molecule(self, mol_subset: MolSubset = None,
                    atom_subset: AtomSubset = None) -> List[Molecule]:
        """Convert this instance into a *list* of *plams.Molecule*.

        Parameters
        ----------
        mol_subset : slice
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if ``None``.

        atom_subset : |Sequence|_
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if ``None``.

        Returns
        -------
        :math:`m` |list|_ [|plams.Molecule|_]:
            A list of :math:`m` PLAMS molecules constructed from this instance.

        """
        m_subset = self._get_mol_subset(mol_subset)
        at_subset = np.asarray(sorted(self._get_atom_subset(atom_subset, as_sequence=True)))
        at_symbols = self.symbol

        # Construct a template molecule and fill it with atoms
        assert self.atoms is not None
        mol_template = Molecule()
        mol_template.properties = self.properties.copy()
        for i in at_subset:
            atom = Atom(symbol=at_symbols[i])
            mol_template.add_atom(atom)

        # Fill the template molecule with bonds
        if self.bonds.any():
            bond_idx = np.ones(len(self))
            bond_idx[at_subset] += np.arange(len(at_subset))
            for i, j, order in self.bonds:
                if i in at_subset and j in at_subset:
                    at1 = mol_template[int(bond_idx[i])]
                    at2 = mol_template[int(bond_idx[j])]
                    mol_template.add_bond(Bond(atom1=at1, atom2=at2, order=order/10.0))

        # Create copies of the template molecule; update their cartesian coordinates
        ret = []
        for i, xyz in enumerate(self[m_subset]):
            mol = mol_template.copy()
            mol.from_array(xyz[at_subset])
            mol.properties.frame = i
            ret.append(mol)

        return ret

    @classmethod
    def from_Molecule(cls, mol_list: Union[Molecule, Iterable[Molecule]],
                      subset: Sequence[str] = 'atoms') -> MultiMolecule:
        """Construct a :class:`.MultiMolecule` instance from one or more PLAMS molecules.

        Parameters
        ----------
        mol_list : |plams.Molecule|_ or |list|_ [|plams.Molecule|_]
            A PLAMS molecule or list of PLAMS molecules.

        subset : |Sequence|_ [str]
            Transfer a subset of *plams.Molecule* attributes to this instance.
            If ``None``, transfer all attributes.
            Accepts one or more of the following values as strings:
            ``"properties"``, ``"atoms"`` and/or ``"bonds"``.

        Returns
        -------
        |FOX.MultiMolecule|_:
            A :class:`.MultiMolecule` instance constructed from **mol_list**.

        """
        if isinstance(mol_list, Molecule):
            plams_mol = mol_list
            mol_list = [mol_list]
        else:
            plams_mol = mol_list[0]
        subset = subset or ('atoms', 'bonds', 'properties')

        # Convert coordinates
        coords = np.array([mol.as_array() for mol in mol_list])
        kwarg: dict = {}

        # Convert atoms
        if 'atoms' in subset:
            kwarg['atoms'] = {}
            for i, at in enumerate(plams_mol.atoms):
                try:
                    kwarg['atoms'][at.symbol].append(i)
                except KeyError:
                    kwarg['atoms'][at.symbol] = [i]

        # Convert properties
        if 'properties' in subset:
            kwarg['properties'] = plams_mol.properties.copy()

        # Convert bonds
        if 'bonds' in subset:
            plams_mol.set_atoms_id(start=0)
            kwarg['bonds'] = np.array([(bond.atom1.id, bond.atom2.id, bond.order * 10) for
                                       bond in plams_mol.bonds], dtype=int)
            plams_mol.unset_atoms_id()

        return cls(coords, **kwarg)

    @classmethod
    def from_xyz(cls, filename: str,
                 bonds: Optional[np.ndarray] = None,
                 properties: Optional[dict] = None) -> MultiMolecule:
        """Construct a :class:`.MultiMolecule` instance from a (multi) .xyz file.

        Comment lines extracted from the .xyz file are stored, as array, under
        ``MultiMolecule.properties["comments"]``.

        Parameters
        ----------
        filename : str
            The path+filename of an .xyz file.

        bonds : :math:`k*3` |np.ndarray|_ [|np.int64|_]
            An optional 2D array with indices of the atoms defining all :math:`k` bonds
            (columns 1 & 2) and their respective bond orders multiplied by 10 (column 3).
            Stored in the **MultieMolecule.bonds** attribute.

        properties : dict
            A Settings object (subclass of dictionary) intended for storing
            miscellaneous user-defined (meta-)data. Is devoid of keys by default.
            Stored in the **MultiMolecule.properties** attribute.

        Returns
        -------
        |FOX.MultiMolecule|_:
            A :class:`.MultiMolecule` instance constructed from **filename**.

        """
        coords, atoms, comments = read_multi_xyz(filename)
        return cls(coords, atoms, bonds, {'comments': comments})

    @classmethod
    def from_kf(cls, filename: str,
                bonds: Optional[np.ndarray] = None,
                properties: Optional[dict] = None) -> MultiMolecule:
        """Construct a :class:`.MultiMolecule` instance from a KF binary file.

        Parameters
        ----------
        filename : str
            The path+filename of an KF binary file.

        bonds : :math:`k*3` |np.ndarray|_ [|np.int64|_]
            An optional 2D array with indices of the atoms defining all :math:`k` bonds
            (columns 1 & 2) and their respective bond orders multiplied by 10 (column 3).
            Stored in the **MultieMolecule.bonds** attribute.

        properties : dict
            A Settings object (subclass of dictionary) intended for storing
            miscellaneous user-defined (meta-)data. Is devoid of keys by default.
            Stored in the **MultiMolecule.properties** attribute.

        Returns
        -------
        |FOX.MultiMolecule|_:
            A :class:`.MultiMolecule` instance constructed from **filename**.

        """
        return cls(*read_kf(filename), bonds, properties)
