# Qiskit Metal Headless Mode

This document describes the headless mode capabilities added to Qiskit Metal, which allows you to use the quantum device design framework without GUI dependencies.

## Overview

Headless mode solves the problem of using Qiskit Metal in environments where GUI components are not available or desired, such as:

- Remote Jupyter notebooks
- SSH sessions
- CI/CD pipelines
- Docker containers
- Server environments
- Automated scripting

## Quick Start

### Explicit Headless Mode

Set the environment variable to explicitly enable headless mode:

```bash
export QISKIT_METAL_HEADLESS=true
python your_script.py
```

Or in Python:
```python
import os
os.environ['QISKIT_METAL_HEADLESS'] = 'true'
import qiskit_metal
```

### Automatic Detection

Qiskit Metal automatically detects headless environments in many scenarios:

- No DISPLAY variable on Unix systems
- SSH connections
- CI environments (GitHub Actions, Travis, etc.)
- Remote Jupyter environments

## Usage Examples

### Basic Headless Workflow

```python
import qiskit_metal

# Create design (automatically loads only headless-compatible renderers)
design = qiskit_metal.designs.DesignPlanar()
print(f"Available renderers: {list(design.renderers.keys())}")
# Output: ['gds', 'gmsh', 'elmer']

# Create components normally
from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
qubit = TransmonPocket(design, 'Q1')

# GUI components are blocked in headless mode
try:
    gui = qiskit_metal.MetalGUI()
except ImportError as e:
    print("GUI not available in headless mode (expected)")
```

### Jupyter Notebook Example

```python
# In a remote Jupyter notebook
import qiskit_metal

# Check headless status
print(f"Headless mode: {qiskit_metal.config.is_headless()}")
print(f"Remote environment: {qiskit_metal.config.is_remote_environment()}")

# Design works normally
design = qiskit_metal.designs.DesignPlanar()

# Add your quantum components...
```

## Key Features

### Automatic Environment Detection

The system automatically detects headless environments based on:

- `QISKIT_METAL_HEADLESS` environment variable
- Absence of DISPLAY variable (Unix)
- SSH connection indicators
- CI environment variables
- Remote Jupyter indicators

### Headless-Compatible Renderers

In headless mode, only renderers without GUI dependencies are loaded:

- **GDS Renderer**: For GDSII file export
- **GMSH Renderer**: For finite element mesh generation  
- **Elmer Renderer**: For finite element analysis

### Matplotlib Backend Management

- **GUI Mode**: Uses QtAgg backend for interactive plots
- **Headless Mode**: Uses Agg backend for file output only

### Error Handling

Attempting to use GUI components in headless mode provides helpful error messages:

```python
qiskit_metal.MetalGUI()
# ImportError: MetalGUI is not available in headless mode. 
# GUI components require PySide6/Qt which is not available in headless environments.
```

## Configuration

### Environment Variables

- `QISKIT_METAL_HEADLESS=true`: Force headless mode
- `QISKIT_METAL_HEADLESS=false`: Force GUI mode (if available)

### Programmatic Control

```python
import qiskit_metal.config as config

# Check current mode
print(f"Headless: {config.is_headless()}")
print(f"Should enable GUI: {config.should_enable_gui()}")

# Check detection details
print(f"IPython: {config.is_using_ipython()}")
print(f"Remote: {config.is_remote_environment()}")
print(f"CI: {config.is_ci_environment()}")
```

### QDesign Renderer Control

```python
# Explicit control over renderer loading
design = qiskit_metal.designs.DesignPlanar(enable_renderers=False)  # No renderers
design = qiskit_metal.designs.DesignPlanar(enable_renderers=True)   # All available renderers
design = qiskit_metal.designs.DesignPlanar()  # Auto-detect based on environment
```

## Advanced Usage

### Custom Renderer Sets

You can customize which renderers are available in headless mode by modifying the configuration:

```python
import qiskit_metal.config as config

# View current headless renderers
print(config.headless_compatible_renderers)

# Add custom renderer (advanced users)
config.headless_compatible_renderers['my_renderer'] = {
    'path_name': 'my_package.my_renderer',
    'class_name': 'MyRenderer'
}
```

## Troubleshooting

### Common Issues

1. **Qt still being imported**: Ensure `QISKIT_METAL_HEADLESS=true` is set before importing qiskit_metal

2. **Some renderers not available**: This is expected - only headless-compatible renderers load in headless mode

3. **Matplotlib backend issues**: The system automatically selects appropriate backends, but you can manually set:
   ```python
   import matplotlib
   matplotlib.use('Agg')  # For headless
   ```

### Debugging

Check your environment:

```python
import qiskit_metal
qiskit_metal.config.is_headless()  # Should return True in headless mode

# Detailed environment check
import os
print(f"QISKIT_METAL_HEADLESS: {os.getenv('QISKIT_METAL_HEADLESS')}")
print(f"DISPLAY: {os.getenv('DISPLAY')}")
print(f"SSH_CLIENT: {os.getenv('SSH_CLIENT')}")
```

## Migration Guide

### Existing Code

Most existing code will work without changes:

```python
# This still works
import qiskit_metal
design = qiskit_metal.designs.DesignPlanar()
# Components work the same...
```

### GUI-Dependent Code

Replace GUI-dependent code:

```python
# Before (GUI required)
import qiskit_metal
gui = qiskit_metal.MetalGUI(design)
gui.rebuild()

# After (headless compatible)
import qiskit_metal
design = qiskit_metal.designs.DesignPlanar()
design.rebuild()  # Direct design manipulation
```
