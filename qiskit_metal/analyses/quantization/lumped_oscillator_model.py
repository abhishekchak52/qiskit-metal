# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import pandas as pd
from pint import UnitRegistry

from pyEPR.calcs.convert import Convert

from qiskit_metal.designs import QDesign  # pylint: disable=unused-import
from qiskit_metal.analyses.core import QAnalysis
from qiskit_metal.analyses.simulation import LumpedElementsSim
from qiskit_metal import Dict, config

if not config.is_building_docs():
    from .lumped_capacitive import extract_transmon_coupled_Noscillator


# TODO: eliminate every reference to "renderer" in this file
class LOManalysis(QAnalysis):
    """Performs Lumped Oscillator Model analysis on a simulated or user-provided capacitance matrix.

    Default Setup:
        * junctions (Dict)
            * Lj (float): Junction inductance (in nH)
            * Cj (float): Junction capacitance (in fF)
        * freq_readout (float): Coupling readout frequency (in GHz).
        * freq_bus (Union[list, float]): Coupling bus frequencies (in GHz).
            * freq_bus can be a list with the order they appear in the capMatrix.

    Data Labels:
        * lumped_oscillator (pd.DataFrame): Lumped oscillator result at the last simulation pass
        * lumped_oscillator_all (dict): of pd.DataFrame. Lumped oscillator resulting
            at every pass of the simulation

    """
    default_setup = Dict(junctions=Dict(Lj=12, Cj=2),
                         freq_readout=7.0,
                         freq_bus=[6.0, 6.2])
    """Default setup."""

    # supported labels for data generated from the simulation
    data_labels = ['lumped_oscillator', 'lumped_oscillator_all']
    """Default data labels."""

    def __init__(self, design: 'QDesign' = None, renderer_name: str = None):
        """Initialize the Lumped Oscillator Model analysis.

        Args:
            design (QDesign): Pointer to the main qiskit-metal design.
                Used to access the QRenderer. Defaults to None.
            renderer_name (str, optional): Which renderer to use. Valid entries: 'q3d'.
                Defaults to None.
        """
        # QAnalysis are expected to either run simulation or use pre-saved sim outputs
        # we use a Dict() to store the sim outputs previously saved. Its key names need
        # to match those found in the correspondent simulation class.
        self.sim = Dict() if renderer_name is None else LumpedElementsSim(
            design, renderer_name)
        super().__init__()

    @property
    def lumped_oscillator(self) -> dict:
        """Getter

        Returns:
            dict: Lumped oscillator result at the last simulation pass.
        """
        return self.get_data('lumped_oscillator')

    @lumped_oscillator.setter
    def lumped_oscillator(self, data: dict):
        """Setter

        Args:
            data (dict): Lumped oscillator result at the last simulation pass.
        """
        if not isinstance(data, dict):
            self.logger.warning(
                'Unsupported type %s. Only accepts dict. Please try again.',
                {type(data)})
            return
        self.set_data('lumped_oscillator', data)

    @property
    def lumped_oscillator_all(self) -> pd.DataFrame:
        """Getter

        Returns:
            pd.DataFrame: each line corresponds to a simulation pass number
                and the remainder of the data is the respective lump oscillator information.
        """
        return self.get_data('lumped_oscillator_all')

    @lumped_oscillator_all.setter
    def lumped_oscillator_all(self, data: pd.DataFrame):
        """Setter

        Args:
            data (pd.DataFrame): each line corresponds to a simulation pass number
                and the remainder of the data is the respective lump oscillator information.
        """
        if not isinstance(data, pd.DataFrame):
            self.logger.warning(
                'Unsupported type %s. Only accepts pd.DataFrame. Please try again.',
                {type(data)})
            return
        self.set_data('lumped_oscillator_all', data)

    def run(self, *args, **kwargs):
        """Executes sequentially the system capacitance simulation (if a renderer was provided
        at creation of this object) and lom extraction by executing the methods
        LumpedElementsSim.run_sim(`*args`, `**kwargs`) and LOManalysis.run_lom().
        For input parameter, see documentation for LumpedElementsSim.run_sim().

        Returns:
            (dict): Pass numbers (keys) and respective lump oscillator information (values).
        """
        if isinstance(self.sim, LumpedElementsSim):
            self.sim.run(*args, **kwargs)
        return self.run_lom()

    def run_lom(self):
        """Executes the lumped oscillator extraction from the capacitance matrix,
        and based on the setup values.

        Returns:
            dict: Pass numbers (keys) and their respective capacitance matrices (values).
        """
        # wipe data from the previous run (if any)
        self.clear_data()

        s = self.setup

        if not isinstance(self.sim.capacitance_matrix, pd.DataFrame):
            if self.sim.capacitance_matrix == {} or self.sim.capacitance_matrix is None:
                self.logger.warning(
                    'Please initialize the capacitance_matrix before executing this method.'
                    '`self.sim.capacitance_matrix = pd.DataFrame(...)`')
                return

        ureg_local = UnitRegistry()
        
        # If capacitance_all_passes is empty (e.g., when using PyAEDT backend which
        # doesn't support per-pass exports), use the main capacitance_matrix instead
        if not self.sim.capacitance_all_passes:
            self.logger.info(
                'Per-pass capacitance matrices not available. Using final capacitance matrix.'
            )
            # Convert capacitance_matrix to farads if it's a DataFrame
            if isinstance(self.sim.capacitance_matrix, pd.DataFrame):
                c_units = ureg_local(self.sim.units).to('farads').magnitude
                self.sim.capacitance_all_passes[1] = self.sim.capacitance_matrix.values * c_units
            else:
                self.sim.capacitance_all_passes[1] = self.sim.capacitance_matrix
        
        # Filter out invalid (all-zero) matrices
        # These can occur when the simulation converged before max_passes
        valid_passes = {}
        for pass_num, matrix in self.sim.capacitance_all_passes.items():
            # Convert to numpy array if needed
            if isinstance(matrix, pd.DataFrame):
                matrix_values = matrix.values
            else:
                matrix_values = np.asarray(matrix)
            
            # Check if matrix is valid (not all zeros)
            # Use an absolute tolerance for very small capacitances (femtofarads = 1e-15)
            max_val = np.max(np.abs(matrix_values))
            # Threshold: if max value is less than 1e-20 F (0.01 aF), consider it zero
            if max_val > 1e-20:
                valid_passes[pass_num] = matrix
            else:
                self.logger.debug(
                    f"Pass {pass_num}: Matrix is all zeros (max value: {max_val}), skipping."
                )
        
        if not valid_passes:
            self.logger.error(
                'No valid capacitance matrices found. All matrices appear to be zeros.'
            )
            return
        
        if len(valid_passes) < len(self.sim.capacitance_all_passes):
            self.logger.info(
                f"Filtered {len(self.sim.capacitance_all_passes) - len(valid_passes)} invalid "
                f"passes (all-zero matrices). Using {len(valid_passes)} valid passes."
            )
        
        # Replace capacitance_all_passes with only valid passes
        self.sim.capacitance_all_passes = valid_passes
        
        ic_amps = Convert.Ic_from_Lj(s.junctions.Lj, 'nH', 'A')
        cj = ureg_local(f'{s.junctions.Cj} fF').to('farad').magnitude
        fread = ureg_local(f'{s.freq_readout} GHz').to('GHz').magnitude
        fbus = [ureg_local(f'{freq} GHz').to('GHz').magnitude for freq in s.freq_bus]

        # derive number of coupling pads from setup
        num_cpads = 2
        if isinstance(fread, list):
            num_cpads += len(fread) - 1
        if isinstance(fbus, list):
            num_cpads += len(fbus) - 1

        # get the LOM for every pass
        all_res = {}
        for idx_cmat, df_cmat in self.sim.capacitance_all_passes.items():
            # Validate matrix size before calling extraction
            matrix_size = len(df_cmat)
            expected_size = num_cpads + 3
            
            if matrix_size != expected_size:
                # Try to auto-derive num_cpads from matrix size
                # Matrix structure: bus1...busN-1, ground, Qubit_pad1, Qubit_pad2, readout
                # So matrix_size = N + 3, where N = num_cpads
                derived_cpads = matrix_size - 3
                
                if derived_cpads > 0:
                    self.logger.warning(
                        f"Capacitance matrix size ({matrix_size}) doesn't match expected size "
                        f"({expected_size}) based on freq_bus/freq_readout settings. "
                        f"Auto-adjusting num_cpads from {num_cpads} to {derived_cpads}."
                    )
                    num_cpads = derived_cpads
                    
                    # Also adjust fbus if needed
                    if isinstance(fbus, list) and len(fbus) != num_cpads - 1:
                        self.logger.warning(
                            f"Number of bus frequencies ({len(fbus)}) doesn't match "
                            f"derived coupling pads ({num_cpads - 1} buses). "
                            f"Using first frequency for all buses."
                        )
                        fbus = [fbus[0]] * (num_cpads - 1) if num_cpads > 1 else []
                else:
                    self.logger.error(
                        f"Capacitance matrix size ({matrix_size}) is too small. "
                        f"Expected at least 4 (N + 3 where N >= 1)."
                    )
                    continue
            
            res = extract_transmon_coupled_Noscillator(
                df_cmat,
                ic_amps,
                cj,
                num_cpads,
                fbus,
                fread,
                g_scale=1,
                print_info=bool(
                    idx_cmat == len(self.sim.capacitance_all_passes)))
            all_res[idx_cmat] = res
        
        # Get the last (or only) result
        if all_res:
            last_key = max(all_res.keys())
            self.lumped_oscillator = all_res[last_key]
        else:
            self.logger.error('No LOM results could be calculated.')
            return
        all_res = pd.DataFrame(all_res).transpose()
        all_res['χr MHz'] = abs(all_res['chi_in_MHz'].apply(lambda x: x[0]))
        all_res['gr MHz'] = abs(all_res['gbus'].apply(lambda x: x[0]))
        self.lumped_oscillator_all = all_res
        return self.lumped_oscillator_all

    def plot_convergence(self, *args, **kwargs):
        """Plots alpha and frequency versus pass number, as well as convergence of delta (in %).

        It accepts the same inputs as run_lom(), to allow regenerating the LOM
        results before plotting them.
        """
        if self.lumped_oscillator_all is None or args or kwargs:
            self.run_lom(*args, **kwargs)
        # TODO: copy plot_convergence_main() from pyEPR and move it here
        self.sim.renderer.plot_convergence_main(self.lumped_oscillator_all)

    def plot_convergence_chi(self, *args, **kwargs):
        """Plot convergence of chi and g, both in MHz, as a function of pass number.

        It accepts the same inputs as run_lom(), to allow regenerating the LOM
        results before plotting them.
        """
        if self.lumped_oscillator_all is None or args or kwargs:
            self.run_lom(*args, **kwargs)
        # TODO: copy plot_convergence_main() from pyEPR and move it here
        self.sim.renderer.plot_convergence_chi(self.lumped_oscillator_all)

    def load_simulation_data(self, data_name: str, data):
        """Load simulation data for the following analysis. This will override any data found

        Args:
            data_name (str): name of the variable
            data (Any): simulation output
        """
        self.sim[data_name] = data
