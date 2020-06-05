"""A module with functions related to manipulating atomic charges.

Index
-----
.. currentmodule:: FOX.functions.charge_utils
.. autosummary::
    update_charge

API
---
.. autofunction:: update_charge

"""

from functools import partial
from types import MappingProxyType
from typing import (
    Hashable, Optional, Collection, Mapping, Container, Dict, Union, Iterable, Tuple, Any,
    SupportsFloat, Generator, Iterator, Set, TypeVar, overload, ItemsView, ValuesView, KeysView,
    FrozenSet, Type, Generic
)

import numpy as np
import pandas as pd

from nanoutils import TypedDict

from ..type_hints import ArrayLike

__all__ = ['update_charge']

T = TypeVar('T')
KT = TypeVar('KT', bound=Hashable)
ST = TypeVar('ST', bound='ChargeError')


class _StateDict(TypedDict):
    """A dictionary representing the keyword-only arguments of :exc:`ChargeError`."""

    reference: Optional[float]
    value: Optional[float]
    tol: Optional[float]


class ChargeError(ValueError, Generic[T]):
    """A :exc:`ValueError` subclass for charge-related errors."""

    __slots__ = ('__weakref__', 'reference', 'value', 'tol')

    reference: Optional[float]
    value: Optional[float]
    tol: Optional[float]
    args: Tuple[T, ...]

    def __init__(self, *args: T, reference: Optional[SupportsFloat] = None,
                 value: Optional[SupportsFloat] = None,
                 tol: Optional[SupportsFloat] = 0.001) -> None:
        """Initialize an instance."""
        super().__init__(*args)
        self.reference = float(reference) if reference is not None else None
        self.value = float(value) if value is not None else None
        self.tol = float(tol) if tol is not None else None

    def __reduce__(self: ST) -> Tuple[Type[ST], Tuple[T, ...], _StateDict]:
        """Helper for :mod:`pickle`."""
        cls = type(self)
        kwargs = _StateDict(reference=self.reference, value=self.value, tol=self.tol)
        return cls, self.args, kwargs

    def __setstate__(self, state: _StateDict) -> None:
        """Helper for :meth:`pickle`; handles the setting of keyword arguments."""
        for k, v in state.items():
            setattr(self, k, v)


def get_net_charge(param: pd.Series, count: pd.Series,
                   index: Optional[Collection] = None) -> float:
    """Calculate the total charge in **df**.

    Returns the (summed) product of the ``"param"`` and ``"count"`` columns in **df**.

    Parameters
    ----------
    df : |pd.DataFrame|_
        A dataframe with atomic charges.
        Charges should be stored in the ``"param"`` column and atom counts
        in the ``"count"`` column (see **key**).

    index : slice
        An object for slicing the index of **df**.

    columns : |Tuple|_ [|Hashable|_]
        The name of the columns holding the atomic charges and number of atoms (per atom type).

    Returns
    -------
    |float|_:
        The total charge in **df**.

    """
    index_ = slice(None) if index is None else index
    ret = param[index_] * count[index_]
    return ret.sum()


def update_charge(atom: KT, value: float, param: pd.Series, count: pd.Series,
                  constrain_dict: Optional[Mapping[KT, partial]] = None,
                  prm_min: Optional[ArrayLike] = None,
                  prm_max: Optional[ArrayLike] = None,
                  net_charge: Optional[float] = None) -> Optional[ChargeError]:
    """Set the atomic charge of **at** equal to **charge**.

    The atomic charges in **df** are furthermore exposed to the following constraints:

        * The total charge remains constant.
        * Optional constraints specified in **constrain_dict**
          (see :func:`.update_constrained_charge`).

    Performs an inplace update of the *param* column in **df**.

    Examples
    --------
    .. code:: python

        >>> print(df)
            param  count
        Br   -1.0    240
        Cs    1.0    112
        Pb    2.0     64

    Parameters
    ----------
    atom : str
        An atom type such as ``"Se"``, ``"Cd"`` or ``"OG2D2"``.

    charge : float
        The new charge associated with **at**.

    df : |pd.DataFrame|_
        A dataframe with atomic charges.
        Charges should be stored in the *param* column and atom counts in the *count* column.

    constrain_dict : dict
        A dictionary with charge constrains.

    """
    param_backup = param.copy()
    param[atom] = value

    if constrain_dict is None or atom in constrain_dict:
        exclude = constrained_update(atom, param, constrain_dict)
    else:
        exclude = {atom}

    if net_charge is not None:
        try:
            unconstrained_update(net_charge, param, count,
                                 prm_min=prm_min,
                                 prm_max=prm_max,
                                 exclude=exclude)
        except ChargeError as ex:
            param[:] = param_backup
            return ex
    return None


def constrained_update(at1: KT, param: pd.Series,
                       constrain_dict: Optional[Mapping[KT, partial]] = None) -> Set[KT]:
    """Perform a constrained update of atomic charges.

    Performs an inplace update of the ``"param"`` column in **df**.

    Parameters
    ----------
    at1 : str
        An atom type such as ``"Se"``, ``"Cd"`` or ``"OG2D2"``.

    df : |pd.DataFrame|_
        A dataframe with atomic charges.

    constrain_dict : dict
        A dictionary with charge constrains (see :func:`.get_charge_constraints`).

    Returns
    -------
    |list|_ [|str|_]:
        A list of atom types with updated atomic charges.

    """
    charge = param[at1]
    exclude = {at1}
    if constrain_dict is None:
        return exclude

    # Perform a constrained charge update
    func1 = invert_partial_ufunc(constrain_dict[at1])
    for at2, func2 in constrain_dict.items():
        if at2 == at1:
            continue
        exclude.add(at2)

        # Update the charges
        param[at2] = func2(func1(charge))
    return exclude


def _constrained_update(atom: KT, value: float, param: pd.Series, count: pd.Series,
                        atom_coefs: Collection[pd.Series],
                        param_min: pd.Series, param_max: pd.Series) -> None:
    """Pass."""
    value_old = pd.Series[atom]
    x = value / value_old

    # Scale all parameters (correct them afterwards)
    param *= x
    param.clip(param_min, param_max, inplace=True)
    param[atom] = value

    # Identify the charge of the moved atoms set of constraints
    for ref_coef in atom_coefs:
        if atom in ref_coef.index:
            break
    else:
        raise ValueError(repr(atom))
    idx = ref_coef.index
    net_charge = (param.loc[idx] * ref_coef.loc[idx]).sum()

    # Update the charge of all other charge-constraint blocks
    coef_iterator = (coef for coef in atom_coefs if coef is not ref_coef)
    for coef in coef_iterator:
        idx = coef.index
        _prm = param.loc[idx] * coef
        _min = param_min.loc[idx] * coef
        _max = param_max.loc[idx] * coef

        unconstrained_update(net_charge, _prm, count.loc[idx], prm_min=_min, prm_max=_max)
        param.loc[idx] = _prm / coef


def unconstrained_update(net_charge: float, param: pd.Series, count: pd.Series,
                         prm_min: Optional[ArrayLike] = None,
                         prm_max: Optional[ArrayLike] = None,
                         exclude: Optional[Container[Hashable]] = None) -> None:
    """Perform an unconstrained update of atomic charges."""
    if exclude is None:
        include = param.astype(bool, copy=True)
        include.loc[:] = True
    else:
        include = pd.Series([i not in exclude for i in param.keys()], index=param.index)
    if not include.any():
        return

    # Identify the multplicative factor that yields a net-neutral charge
    i = net_charge - get_net_charge(param, count, ~include)
    i /= get_net_charge(param, count, include)

    # Define the minimum and maximum values
    s_min = prm_min if prm_min is not None else -np.inf
    s_max = prm_max if prm_max is not None else np.inf

    # Identify which parameters are closest to their extreme values
    s = param * i
    s_clip = np.clip(s, s_min, s_max).loc[include]
    s_delta = abs(s_clip - s.loc[include])
    s_delta.sort_values(ascending=False, inplace=True)

    start = -len(s_delta) + 1
    for j, atom in enumerate(s_delta.index, start=start):
        param[atom] = s_clip[atom]
        include[atom] = False

        if j and s_clip[atom] != s[atom]:
            i = net_charge - get_net_charge(param, count, ~include)
            i /= get_net_charge(param, count, include)

            s = param * i
            s_clip = np.clip(s, s_min, s_max).loc[include]

    _check_net_charge(param, count, net_charge)


def _check_net_charge(param: pd.Series, count: pd.Series, net_charge: float,
                      tolerance: float = 0.001) -> None:
    """Check if the net charge is actually conserved."""
    net_charge_new = get_net_charge(param, count)
    condition = abs(net_charge - net_charge_new) > tolerance

    if not condition:
        return

    raise ChargeError(
        f"Failed to conserve the net charge: ref = {net_charge:.4f}); {net_charge_new:.4f} != ref",
        reference=net_charge, value=net_charge_new, tol=tolerance
    )


def invert_partial_ufunc(ufunc: partial) -> partial:
    """Invert a NumPy universal function embedded within a :class:`partial` instance."""
    func = ufunc.func
    x2 = ufunc.args[0]
    return partial(func, x2**-1)


ExtremiteDict = Dict[Tuple[str, str], float]


def assign_constraints(constraints: Union[str, Iterable[str]]
                       ) -> Tuple[ExtremiteDict, Optional[ConstrainDict]]:
    operator_set = {'>', '<', '*', '=='}

    # Parse integers and floats
    if isinstance(constraints, str):
        constraints = [constraints]

    constrain_list = []
    for item in constraints:
        for i in operator_set:  # Sanitize all operators; ensure they are surrounded by spaces
            item = item.replace(i, f'~{i}~')

        item_list = [i.strip().rstrip() for i in item.split('~')]
        if len(item_list) == 1:
            continue

        for i, j in enumerate(item_list):  # Convert strings to floats where possible
            try:
                float_j = float(j)
            except ValueError:
                pass
            else:
                item_list[i] = float_j

        constrain_list.append(item_list)

    # Set values in **param**
    extremite_dict: ExtremiteDict = {}
    constraints_ = None
    for constrain in constrain_list:
        if '==' in constrain:
            constraints_ = _eq_constraints(constrain)
        else:
            extremite_dict.update(_gt_lt_constraints(constrain))
    return extremite_dict, constraints_


#: Map ``"min"`` to ``"max"`` and *vice versa*.
_INVERT = MappingProxyType({'max': 'min', 'min': 'max'})

#: Map :math:`>`, :math:`<`, :math:`\ge` and :math:`\le` to either ``"min"`` or ``"max"``.
_OPPERATOR_MAPPING = MappingProxyType({'<': 'min', '<=': 'min', '>': 'max', '>=': 'max'})


def _gt_lt_constraints(constrain: list) -> Generator[Tuple[Tuple[str, str], float], None, None]:
    r"""Parse :math:`>`, :math:`<`, :math:`\ge` and :math:`\le`-type constraints."""
    for i, j in enumerate(constrain):
        if j not in _OPPERATOR_MAPPING:
            continue

        operator, value, atom = _OPPERATOR_MAPPING[j], constrain[i-1], constrain[i+1]
        if isinstance(atom, float):
            atom, value = value, atom
            operator = _INVERT[operator]
        yield (atom, operator), value


def _find_float(iterable: Tuple[str, str]) -> Tuple[str, float]:
    """Take an iterable of 2 strings and identify which element can be converted into a float."""
    try:
        i, j = iterable
    except ValueError:
        return iterable[0], 1.0

    try:
        return j, float(i)
    except ValueError:
        return i, float(j)


def _eq_constraints(constrain_: list) -> Dict[str, partial]:
    """Parse :math:`a = i * b`-type constraints."""
    constrain_dict: Dict[str, partial] = {}
    constrain = ''.join(str(i) for i in constrain_).split('==')
    iterator: Iterator[str] = iter(constrain)

    # Set the first item; remove any prefactor and compensate al other items if required
    item_ = next(iterator).split('*')
    if len(item_) == 1:
        atom = item_[0]
        multiplier = 1.0
    elif len(item_) == 2:
        atom, multiplier = _find_float(item_)
        multiplier **= -1
    constrain_dict[atom] = partial(np.multiply, 1.0)

    # Assign all other constraints
    for item in iterator:
        item_ = item.split('*')
        atom, i = _find_float(item_)
        i *= multiplier
        constrain_dict[atom] = partial(np.multiply, i)
    return constrain_dict
