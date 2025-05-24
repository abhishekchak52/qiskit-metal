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

# pylint: disable=invalid-name
# pylint: disable=unused-import
"""File contains some config definitions.

Mostly internal.
"""

from .toolbox_python.attr_dict import Dict
from ._defaults import DefaultMetalOptions, DefaultOptionsRenderer
import os

renderers_to_load = Dict(
    hfss=Dict(path_name='qiskit_metal.renderers.renderer_ansys.hfss_renderer',
              class_name='QHFSSRenderer'),
    q3d=Dict(path_name='qiskit_metal.renderers.renderer_ansys.q3d_renderer',
             class_name='QQ3DRenderer'),
    gds=Dict(path_name='qiskit_metal.renderers.renderer_gds.gds_renderer',
             class_name='QGDSRenderer'),
    gmsh=Dict(path_name='qiskit_metal.renderers.renderer_gmsh.gmsh_renderer',
              class_name='QGmshRenderer'),
    elmer=Dict(path_name='qiskit_metal.renderers.renderer_elmer.elmer_renderer',
               class_name='QElmerRenderer'),
    aedt_q3d=Dict(
        path_name=
        'qiskit_metal.renderers.renderer_ansys_pyaedt.q3d_renderer_aedt',
        class_name='QQ3DPyaedt'),
    aedt_hfss=Dict(
        path_name=
        'qiskit_metal.renderers.renderer_ansys_pyaedt.hfss_renderer_aedt',
        class_name='QHFSSPyaedt'))
"""
Define the renderes to load. Just provide the module names here.
"""

# Headless-compatible renderers that don't require Qt
headless_compatible_renderers = Dict(
    gds=Dict(path_name='qiskit_metal.renderers.renderer_gds.gds_renderer',
             class_name='QGDSRenderer'),
    gmsh=Dict(path_name='qiskit_metal.renderers.renderer_gmsh.gmsh_renderer',
              class_name='QGmshRenderer'),
    elmer=Dict(path_name='qiskit_metal.renderers.renderer_elmer.elmer_renderer',
               class_name='QElmerRenderer'))
"""
Define renderers that work in headless environments without Qt dependencies.
"""

GUI_CONFIG = Dict(
    load_metal_modules=Dict(Qubits='qiskit_metal.qlibrary.qubits',
                            TLines='qiskit_metal.qlibrary.tlines',
                            Terminations='qiskit_metal.qlibrary.terminations'),
    exclude_metal_classes=['Metal_Qubit'],
    tips=[
        'Right clicking the tree elements allows you to do neat things.',
        'You can show all connector names on the plot by clicking the connector '
        'icon in the plot toolbar.',
        'The gui and the Python code work synchronously. If you modify something '
        'in the gui, it will be reflected in your Python interpreter and vice versa. '
        'Note that the gui does not automatically refresh on all events if you update '
        'variables from the Python interpreter.',
        'Changed some object parameters? Click the <b>Remake</b> button in the main '
        'toolbar to recreate the polygons.',
        """<b>Log widget:</b> Right click the logger window to be able to change the log level and
        the loggers that are shown / hidden.""",
        """<b>All component widget:</b> Double click a component to zoom into it!""",
    ],
    logger=Dict(
        style=
        ".DEBUG {color: green;}\n.WARNING,.ERROR,.CRITICAL {color: red;}\n.'\
                'ERROR,.CRITICAL {font-weight: bold;}\n",
        num_lines=500,
        level='DEBUG',
        stream_to_std=False,  # stream to jupyter notebook
    ),
    main_window=Dict(
        title='Qiskit Metal — The Quantum Builder',
        auto_size=False,  # Autosize on creation of window
    ))
"""
GUI_CONFIG

**load_metal_modules**

---------------------------
Name of class folders that contain modules that will be available to be
created in the GUI

Conventions:
Assumes that the module file name is the same as the class name contained in it.
For example, provided `qiskit_metal.qubits` has `Metal_Transmon_Pocket.py`, the
gui will do
    `from qiskit_metal.qubits.Metal_Transmon_Pocket import Metal_Transmon_Pocket`


**tips**

---------------------------
Tips that the user can define to show in the gui. These rotate each time the gui is started.


**logger**

---------------------------
Logger settings


**main_window**

---------------------------
Main window defaults
"""

log = Dict(format='%(asctime)s %(levelname)s [%(funcName)s]: %(message)s',
           datefmt='%I:%M%p %Ss')
"""
A dictionary containing the log format for standard text and date/time
"""


def is_using_ipython():
    """Check if we're in IPython.

    Returns:
        bool -- True if ran in IPython
    """
    return 'JPY_PARENT_PID' in os.environ


def is_building_docs():
    """Checks for the existance of the .buildingdocs file which is only present
    when building the docs.

    Returns:
        bool: True if .buildingdocs exists
    """
    from pathlib import Path  # pylint: disable=import-outside-toplevel
    build_docs_file = Path(__file__).parent.parent / "docs" / ".buildingdocs"
    return Path.exists(build_docs_file)


def is_headless():
    """Comprehensive check for headless environment.
    
    Returns:
        bool: True if running in headless mode, False otherwise
    """
    # Check explicit headless environment variable
    if os.getenv('QISKIT_METAL_HEADLESS', '').lower() in ('true', '1', 'yes'):
        return True
    
    # Check if building docs
    if is_building_docs():
        return True
    
    # Check for remote/headless environments
    if is_remote_environment():
        return True
    
    # Check for CI/testing environments
    if is_ci_environment():
        return True
    
    return False


def is_remote_environment():
    """Check if running in a remote environment (SSH, etc.).
    
    Returns:
        bool: True if in remote environment
    """
    # Check for SSH connection
    if os.getenv('SSH_CLIENT') or os.getenv('SSH_TTY') or os.getenv('SSH_CONNECTION'):
        return True
    
    # Check for missing DISPLAY variable on Unix systems
    if os.name != 'nt' and not os.getenv('DISPLAY'):
        return True
    
    # Check for remote Jupyter environments
    # Remote Jupyter often doesn't have proper X11 forwarding
    if is_using_ipython():
        # Additional checks for remote Jupyter
        if not os.getenv('DISPLAY') and os.name != 'nt':
            return True
        # Check for common remote Jupyter indicators
        if os.getenv('JUPYTER_SERVER_ROOT') or os.getenv('JUPYTERHUB_SERVICE_PREFIX'):
            return True
    
    return False


def is_ci_environment():
    """Check if running in a CI/testing environment.
    
    Returns:
        bool: True if in CI environment
    """
    ci_indicators = [
        'CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS', 'TRAVIS', 
        'CIRCLECI', 'JENKINS_URL', 'GITLAB_CI', 'BUILDKITE',
        'TF_BUILD'  # Azure DevOps
    ]
    
    return any(os.getenv(indicator) for indicator in ci_indicators)


def should_enable_gui():
    """Determine if GUI components should be enabled.
    
    Returns:
        bool: True if GUI should be enabled, False for headless mode
    """
    return not is_headless()


_ipython = is_using_ipython()

####################################################################################
# USER CONFIG
