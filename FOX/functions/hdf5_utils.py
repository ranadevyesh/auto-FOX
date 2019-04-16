""" Functions for storing Monte Carlo results in hdf5 format. """

__all__ = ['create_hdf5', 'index_to_hdf5', 'to_hdf5']

import os
from os.path import join

import numpy as np
import pandas as pd

from scm.plams import Settings

try:
    import h5py
    H5PY_ERROR = False
except ImportError:
    __all__ = []
    H5PY_ERROR = "Use of the FOX.{} function requires the 'h5py' package.\
                  \n\t'h5py' can be installed via anaconda with the following command:\
                  \n\tconda install --name FOX -y -c conda-forge h5py"

from ..functions.utils import get_shape


def create_hdf5(mc_kwarg, name='MC.hdf5'):
    """ Create a hdf5 file to hold all addaptive rate Mone Carlo results (:class:`FOX.ARMC`).
    Datasets are created to hold a number of results following results over the course of the
    MC optimization:

    * The acceptance rate (dataset: *acceptance*)
    * The parameters (dataset: *param*)
    * User-specified PES descriptors (dataset(s): user-specified name(s))
    * The *index*, *columns* and/or *name* attributes above-mentioned results

    :parameter mc_kwarg: An ARMC object.
    :type mc_kwarg: |FOX.ARMC|_
    :parameter str name: The name (including extension) of the hdf5 file.
    """
    path = mc_kwarg.job.path
    filename = join(path, name)

    # Create a Settings object with the shape and dtype of all to-be stored data
    shape_dict = Settings()
    shape_dict.param.shape = mc_kwarg.armc.iter_len, len(mc_kwarg.param)
    shape_dict.param.dtype = np.float64
    shape_dict.acceptance.shape = (mc_kwarg.armc.iter_len, )
    shape_dict.acceptance.dtype = bool
    for key, value in mc_kwarg.pes.items():
        shape_dict[key].shape = (mc_kwarg.armc.iter_len, ) + get_shape(value.ref)
        shape_dict[key].dtype = np.float64

    # Create a hdf5 file with *n* datasets
    with h5py.File(filename, 'w-') as f:
        f.iteration = -1
        f.subiteration = -1
        for key, value in shape_dict.items():
            shape = value.shape
            dtype = value.dtype
            f.create_dataset(name=key, shape=shape, maxshape=shape, dtype=dtype)

    # Store the *index*, *column* and *name* attributes of dataframes/series in the hdf5 file
    pd_dict = {'param': mc_kwarg.param}
    for key, value in mc_kwarg.pes.items():
        pd_dict[key] = value.ref
    index_to_hdf5(pd_dict, path)


def index_to_hdf5(pd_dict, path=None, name='MC.hdf5'):
    """ Export the *index* and *columns* / *name* attributes of a Pandas dataframe/series to a
    pre-existing hdf5 file.
    Attributes are exported for all dataframes/series in **pd_dict** and skipped otherwise.
    The keys in **pd_dict**, together with the attribute names, are used for naming the datasets:

    .. code-block:: python

        >>> pd_dict = {}
        >>> pd_dict['df'] = pd.DataFrame(np.random.rand(10, 10))
        >>> pd_dict['series'] = pd.Series(np.random.rand(10))
        >>> index_to_hdf5(pd_dict, name='my_file.hdf5')

        >>> with h5py.File('my_file.hdf5', 'r') as f:
        >>>     tuple(f.keys())
        ('df.columns', 'df.index', 'series.index', 'series.name')

    :parameter pd_dict: A dictionary with dataset names as keys and matching array-like objects
        as values.
    :parameter str path: The path where the the hdf5 file is stored.
    :parameter str name: The name (including extension) of the hdf5 file.
    """
    path = path or os.getcwd()
    filename = join(path, name)
    attr_tup = ('index', 'columns', 'name')

    with h5py.File(filename, 'r+') as f:
        for key, value in pd_dict.items():
            key += '.{}'
            for attr_name in attr_tup:
                if hasattr(value, attr_name):
                    attr = getattr(value, attr_name)
                    i, j = key.format(attr_name), _attr_to_array(attr)
                    f.create_dataset(i, data=j)


def _attr_to_array(item):
    """ Convert an attribute value, retrieved from :func:`FOX.index_to_hdf5`, into a NumPy array.

    .. code-block:: python

        >>> item = 'name'
        >>> _attr_to_array(item)
        array([b'name'], dtype='|S4')

        >>> item = pd.Index(np.arange(5))
        >>> _attr_to_array(item)
        array([0, 1, 2, 3, 4, 5])

    :parameter object item: An object that may or may not belong to the pd.Index class.
    :return: An array created fron **item**.
    :rtype: |np.ndarray|_
    """
    # If **idx** does not belong to the pd.Index class or one of its subclass
    if not isinstance(item, pd.Index):  # **item** belongs to the *name* attribute of pd.Series
        return np.array(item, dtype='S', ndmin=1)

    # Convert **item** into an array
    ret = np.array(item.to_list())
    if 'U' in ret.dtype.str:  # h5py does not support unicode strings
        return ret.astype('S', copy=False)  # Convert to byte strings
    return ret


def to_hdf5(dict_, i, j, path=None, name='MC.hdf5'):
    """ Export results from **dict_** to the hdf5 file **name**.

    :parameter dict dict_: A dictionary with dataset names as keys and matching array-like objects
        as values.
    :parameter int i: The iteration in the outer loop of :meth:`ARMC.init_armc`.
    :parameter int j: The subiteration in the inner loop of :meth:`ARMC.init_armc`.
    :parameter str path: The path where the the hdf5 file is stored.
    :parameter str name: The name (including extension) of the hdf5 file
    """
    path = path or os.getcwd()
    filename = join(path, name)
    k = j + i * j

    with h5py.File(filename, 'r+') as f:
        f.iteration = i
        f.subiteration = j
        for key, value in dict_.items():
            f[key][k] = value


# If hdf5 is not installed
if H5PY_ERROR:
    _doc1 = create_hdf5.__doc__
    def create_hdf5(mc_kwarg, name='MC.hdf5'):
        raise ModuleNotFoundError(H5PY_ERROR.format('create_hdf5'))


    _doc2 = index_to_hdf5.__doc__
    def index_to_hdf5(dict_, path=None, name='MC.hdf5'):
        raise ModuleNotFoundError(H5PY_ERROR.format('index_to_hdf5'))


    _doc3 = to_hdf5.__doc__
    def to_hdf5(dict_, i, j, path=None, name='MC.hdf5'):
        raise ModuleNotFoundError(H5PY_ERROR.format('to_hdf5'))


    to_hdf5.__doc__ = _doc1
    create_hdf5.__doc__ = _doc2
    index_to_hdf5.__doc__ = _doc3