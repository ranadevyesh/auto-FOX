import numpy as np

from os.path import join

from scm.plams import Settings

from ..classes.psf_dict import PSFDict
from ..functions.utils import (get_template, _get_move_range, dict_to_pandas, get_atom_count)
from ..functions.cp2k_utils import (set_keys, set_subsys_kind)
from ..armc_functions.schemas import (
    get_pes_schema, schema_armc, schema_move, schema_job, schema_param,
    schema_hdf5, schema_molecule, schema_psf
)

__all__ = ['init_armc_sanitization']


def init_armc_sanitization(dict_: dict) -> Settings:
    """Initialize the armc input settings sanitization.

    Parameters
    ----------
    dict_ : dict
        A dictionary containing all ARMC settings.

    Returns
    -------
    |plams.Settings|_:
        A Settings instance suitable for ARMC initialization.

    """
    # Load and apply the template
    s = get_template('armc_template.yaml')
    s.update(Settings(dict_))

    # Validate, post-process and return
    s_ret = validate(s)
    reshape_settings(s_ret)
    return s_ret


def validate(s: Settings) -> Settings:
    """Validate all settings in **s** using schema_.

    Preset schemas are stored in :mod:`.schemas`.

    .. _schema: https://github.com/keleshev/schema

    Parameters
    ----------
    s : |plams.Settings|_
        A Settings instance containing all ARMC settings.

    Returns
    -------
    |plams.Settings|_:
        A validated Settings instance.

    """
    # Flatten the Settings instance
    job_settings = s.job.pop('settings')
    pes_settings = s.pop('pes')
    s_flat = Settings()
    for k, v in s.items():
        try:
            s_flat[k] = v.flatten(flatten_list=False)
        except AttributeError:
            s_flat[k] = v

    # Validate the Settings instance
    s_flat.psf = schema_psf.validate(s_flat.psf)
    s_flat.job = schema_job.validate(s_flat.job)
    s_flat.hdf5_file = schema_hdf5.validate(s_flat.hdf5_file)
    s_flat.armc = schema_armc.validate(s_flat.armc)
    s_flat.move = schema_move.validate(s_flat.move)
    s_flat.param = schema_param.validate(s_flat.param)
    s_flat.molecule = schema_molecule.validate(s_flat.molecule)
    for k, v in pes_settings.items():
        schema_pes = get_pes_schema(k)
        pes_settings[k] = schema_pes.validate(v)

    # Unflatten and return
    s_ret = Settings()
    for k, v in s_flat.items():
        try:
            s_ret[k] = v.unflatten()
        except AttributeError:
            s_ret[k] = v

    s_ret.job.settings = job_settings
    s_ret.pes = pes_settings
    return s_ret


def reshape_settings(s: Settings) -> None:
    """Reshape and post-process the validated ARMC settings.

    Parameters
    ----------
    s : |plams.Settings|_
        A Settings instance containing all ARMC settings.

    """
    s.job.molecule = s.pop('molecule')

    for v in s.pes.values():
        v.ref = v.func(s.job.molecule, *v.arg, **v.kwarg)

    s.move.range = _get_move_range(**s.move.range)
    if s.move.charge_constraints is None:
        s.move.charge_constraints = Settings()

    if s.param.prm_file is None:
        s.job.settings.input.force_eval.mm.forcefield.parmtype = 'OFF'
        del s.param.prm_file
    else:
        s.job.settings.input.force_eval.mm.forcefield.conn_file_name = s.param.pop('prm_file')

    s.param = dict_to_pandas(s.param, 'param')
    s.param['param old'] = np.nan
    s.param['key'] = set_keys(s.job.settings, s.param)
    s.param['count'] = get_atom_count(s.param.index, s.job.molecule)

    s.phi.phi = s.armc.pop('phi')
    s.phi.arg = []
    s.phi.kwarg = Settings()
    s.phi.func = np.add

    s.job.psf = generate_psf(s.pop('psf'), s.param, s.job)
    set_subsys_kind(s.job.settings, s.job.psf['atoms'])


def generate_psf(psf: Settings,
                 param: Settings,
                 job: Settings) -> Settings:
    """Generate the job.psf block.

    Parameters
    ----------
    psf : |plams.Settings|_
        The psf block for the ARMC input settings.

    param : |pd.DataFrame|_
        A DataFrame with all to-be optimized parameters.

    job : |plams.Settings|_
        The job block of the ARMC input settings.

    Returns
    -------
    |plams.Settings|_:
        The updated psf block

    """
    psf_file = join(job.path, 'mol.psf')
    mol = job.molecule

    if all(i is None for i in psf.values()):
        psf_dict: PSFDict = PSFDict.from_multi_mol(mol)
        psf_dict.filename = np.array([False])
    else:
        mol.guess_bonds(atom_subset=psf.ligand_atoms)
        psf_dict: PSFDict = PSFDict.from_multi_mol(mol)
        psf_dict.filename = psf_file
        psf_dict.update_atom_type(psf.str_file)
        job.settings.input.force_eval.subsys.topology.conn_file_name = psf_file
        job.settings.input.force_eval.subsys.topology.conn_file_format = 'PSF'

    for at, charge in param.loc['charge', 'param'].items():
        psf_dict.update_atom_charge(at, charge)

    return psf_dict
