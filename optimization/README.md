# LW Integrator Optimization Module

This module provides tools for optimizing laser-wakefield integrator parameters to maximize energy gain and other performance metrics.

## Overview

The optimization module enables systematic exploration and optimization of simulation parameters including:

- **Aperture sizes** (from 1 nm to 1 mm)
- **Initial energies** (from 1 MeV to 500 GeV)
- **Transverse momentum and position spread**
- **Starting positions** (10 mm to 150 m)
- **Timestep refinement**
- **Particle and subparticle counts**

## Module Structure

```
optimization/
├── __init__.py           # Main API exports
├── metrics.py            # Performance metric calculations
├── parameter_sweep.py    # Grid-based parameter exploration
├── optimizer.py          # Gradient-free optimization algorithms
├── visualization.py      # Heatmaps and analysis plots
└── README.md            # This file
```

## Quick Start

### 1. Parameter Sweep (Grid Search)

Run a systematic sweep over aperture sizes and energies:

```python
from optimization import create_energy_aperture_grid, run_parameter_sweep
from optimization.visualization import plot_energy_heatmap

# Create base configuration
base_config = {
    "rest_energy_mev": 0.511,  # Electron
    "particle_count": 10,
    "timestep": 0.01,
    # ... other parameters
}

# Define parameter grid
grid = create_energy_aperture_grid(
    aperture_sizes_mm=[0.001, 0.01, 0.1, 1.0],  # 1 μm to 1 mm
    energies_gev=[1.0, 10.0, 100.0]              # 1 to 100 GeV
)

# Run sweep
results = run_parameter_sweep(
    base_config=base_config,
    parameter_grid=grid,
    output_dir="results/sweep_001"
)

# Visualize results
plot_energy_heatmap(
    aperture_sizes=results['arrays']['aperture_radius'],
    energies=results['arrays']['initial_energy_gev'],
    metric_values=results['arrays']['max_energy_gain_gev'],
    save_path='results/heatmap.png'
)
```

### 2. Optimization (Finding Optima)

Use optimization algorithms to find best parameters:

```python
from optimization import optimize_parameters

result = optimize_parameters(
    config_template=base_config,
    parameter_names=['aperture_radius', 'initial_energy_gev'],
    parameter_bounds=[(1e-6, 1.0), (1.0, 100.0)],
    metric_name='max_energy_gain_gev',
    method='differential_evolution',
    maximize=True,
    maxiter=50
)

print(f"Best parameters: {result.best_params_dict}")
print(f"Best energy gain: {result.objective_function.best_value} GeV")
```

### 3. Dual Energy Plotting

Plot both total ΔE (from Δγ) and longitudinal ΔE_z (from Δβ_z):

```python
from examples.validation.core_vs_legacy_benchmark import compute_delta_energy_components
from optimization.visualization import plot_dual_energy_curves

# Compute both energy components from trajectory
delta_e_total, delta_e_z, z_series = compute_delta_energy_components(
    states=trajectory,
    initial_state=initial_state,
    rest_energy_mev=0.511
)

# Create dual plot
plot_dual_energy_curves(
    z_positions=z_series,
    delta_e_total=delta_e_total,
    delta_e_z=delta_e_z,
    aperture_z=0.0,
    save_path='dual_energy.png'
)
```

## Key Features

### Metrics (`metrics.py`)

Compute various performance metrics from trajectories:

- **Max energy gain**: Peak energy increase along trajectory
- **Near-aperture gain**: Energy gain within specified range of aperture
- **Relative gain**: ΔE/E₀ for comparing different initial energies
- **Transverse deflection detection**: Identify energy jumps followed by dips
- **Comprehensive trajectory metrics**: Full analysis including deflections

```python
from optimization.metrics import compute_trajectory_metrics

metrics = compute_trajectory_metrics(
    trajectory=states,
    initial_state=initial_state,
    rest_energy_mev=0.511,
    aperture_z=0.0
)

print(f"Max energy gain: {metrics['max_energy_gain_gev']} GeV")
print(f"Max relative gain: {metrics['max_relative_gain']*100:.2f}%")
print(f"Deflection events: {metrics['num_deflection_events']}")
```

### Parameter Sweeps (`parameter_sweep.py`)

Systematic exploration of parameter space:

- **ParameterGrid**: Define multi-dimensional parameter grids
- **run_parameter_sweep**: Execute simulations over entire grid
- **Automatic saving**: Results saved as JSON and NumPy arrays
- **Custom metrics**: Define your own metric functions

```python
from optimization import ParameterGrid, run_parameter_sweep

# Custom parameter grid
grid = ParameterGrid({
    'aperture_radius': [0.01, 0.05, 0.1],
    'start_z': [-10.0, -15.0, -20.0],
    'transverse_momentum': [0.0, 0.01, 0.02]
})

# Run with custom metric function
def custom_metric(trajectory, config):
    # Your custom analysis
    return {"my_metric": some_value}

results = run_parameter_sweep(
    base_config=config,
    parameter_grid=grid,
    metric_function=custom_metric,
    save_trajectories=True  # Save full trajectories
)
```

### Optimization Algorithms (`optimizer.py`)

Find optimal configurations using gradient-free methods:

- **Differential Evolution**: Global optimization for multi-parameter problems
- **Nelder-Mead, Powell, etc.**: Local optimization methods
- **Multi-start optimization**: Multiple random starts for robust global optimization
- **Adaptive grid search**: Coarse-to-fine grid refinement

```python
from optimization import multi_start_optimize

# Run optimization with multiple random starts
result = multi_start_optimize(
    config_template=base_config,
    parameter_names=['aperture_radius', 'initial_energy_gev', 'start_z'],
    parameter_bounds=[(1e-6, 1.0), (1.0, 100.0), (-150.0, -10.0)],
    n_starts=5,
    method='nelder_mead',
    maxiter=50
)

# Access results from all starts
for i, start_result in enumerate(result.all_starts):
    print(f"Start {i}: {start_result.objective_function.best_value}")
```

### Visualization (`visualization.py`)

Create publication-quality plots and analysis figures:

- **Energy heatmaps**: 2D plots of metric vs two parameters
- **Dual energy curves**: Total ΔE and ΔE_z on same plot
- **Parameter slices**: 1D cuts through parameter space
- **Optimization summaries**: Multi-panel summary figures
- **Interactive plots**: HTML plots using Plotly (optional)

```python
from optimization.visualization import plot_optimization_summary

# Create comprehensive summary figure
plot_optimization_summary(
    results=sweep_results,
    primary_metric='max_energy_gain_gev',
    figsize=(16, 12),
    save_path='summary.png'
)
```

## Recommended Workflow

### Phase 1: Initial Exploration (Coarse Sweep)

1. **Wide parameter ranges** with coarse spacing
2. **Low particle counts** (10-20) for speed
3. **Disabled self-consistency** for faster runs
4. **Larger timesteps** (0.01-0.1 mm/c)

Goal: Identify promising regions in parameter space.

### Phase 2: Focused Optimization

1. **Narrow parameter ranges** around promising regions
2. **Run optimization algorithms** (differential evolution or multi-start)
3. **Moderate particle counts** (20-50)
4. **Medium timesteps** (0.001-0.01 mm/c)

Goal: Find local optima in promising regions.

### Phase 3: Fine-Tuning

1. **Small variations** around optimum
2. **High particle counts** (50-100+)
3. **Small timesteps** (0.0001-0.001 mm/c)
4. **More subparticles** (8-16)
5. **Enabled self-consistency** (3-5 iterations)
6. **Multiple random seeds** for robustness

Goal: Refine optimum and verify stability.

## Important Considerations

### Energy Jumps and Transverse Deflections

Energy jumps followed by large dips often indicate **transverse deflections** rather than true acceleration:

```python
from optimization.metrics import detect_transverse_deflection

deflections = detect_transverse_deflection(
    trajectory,
    energy_jump_threshold=0.1,   # 10% jump
    energy_dip_threshold=0.05    # 5% dip
)

for step, event_type, magnitude in deflections:
    if event_type == "deflection":
        print(f"Deflection at step {step}: magnitude {magnitude*100:.1f}%")
```

**Solutions:**
- Adjust transverse momentum spread
- Modify starting transverse position
- Change starting z position
- Try different aperture sizes

### Computational Considerations

**Speed vs. Accuracy Trade-offs:**

| Parameter | Fast (Sweep) | Medium (Optimize) | Slow (Fine-tune) |
|-----------|--------------|-------------------|------------------|
| Particle count | 10 | 20-50 | 50-100 |
| Timestep | 0.01-0.1 | 0.001-0.01 | 0.0001-0.001 |
| Subparticles | 4 | 4-8 | 8-16 |
| SC iterations | 1 | 1-3 | 3-5 |
| Run time/config | ~seconds | ~minutes | ~tens of minutes |

### Position Ranges

Keep starting positions within **realistic lab scales**:
- **Minimum**: 10 mm (1 cm)
- **Typical**: 10-50 mm
- **Maximum**: 150 m (if you see strong trends)

Interaction typically occurs **10-20 mm in front of aperture**.

## Example Scripts

See `examples/optimization_example.py` for complete workflows:

```bash
# Quick test (3x3 grid, ~1 minute)
python examples/optimization_example.py --mode quick

# Full sweep (15x15 grid, hours)
python examples/optimization_example.py --mode sweep

# Run optimization
python examples/optimization_example.py --mode optimize

# Fine-tune around optimum
python examples/optimization_example.py --mode finetune

# Generate example plots
python examples/optimization_example.py --mode plot
```

## Advanced Usage

### Custom Objective Functions

Create custom optimization objectives:

```python
from optimization.optimizer import ObjectiveFunction

class MyObjective(ObjectiveFunction):
    def _compute_metrics(self, trajectory, config):
        # Custom metric computation
        # e.g., minimize emittance while maximizing energy
        energy_gain = compute_max_energy_gain(...)
        emittance = compute_emittance(...)
        
        # Combined metric: energy/emittance ratio
        return {
            'energy_per_emittance': energy_gain / emittance
        }

# Use in optimization
objective = MyObjective(...)
result = scipy.optimize.differential_evolution(objective, bounds=...)
```

### Parallel Execution

For large sweeps, enable parallel execution:

```python
results = run_parameter_sweep(
    base_config=config,
    parameter_grid=grid,
    max_workers=4  # Use 4 CPU cores
)
```

**Note**: Currently sequential (max_workers=1), parallel support coming soon.

### Loading and Analyzing Previous Results

```python
from optimization.parameter_sweep import load_sweep_results

# Load previous sweep
results = load_sweep_results('results/sweep_001')

# Access arrays
energy_gains = results['arrays']['max_energy_gain_gev']

# Find best configuration
best_idx = np.argmax(energy_gains.flatten())
best_params = results['parameters'][best_idx]
print(f"Best config: {best_params}")
```

## Tips and Best Practices

1. **Start with quick sweeps** to understand parameter sensitivities
2. **Use log-spacing** for parameters spanning multiple orders of magnitude
3. **Save intermediate results** frequently
4. **Validate results** by checking for transverse deflections
5. **Test multiple random seeds** when fine-tuning
6. **Monitor trajectories** near interaction points
7. **Check for numerical artifacts** when using very small timesteps
8. **Balance computation time** vs. accuracy based on your goals

## Troubleshooting

### "NaN or Inf in results"
- Timestep may be too large or too small
- Try different starting positions
- Check for numerical instabilities near aperture

### "Optimization stuck at boundary"
- Expand parameter bounds
- Try different optimization method
- Use multi-start optimization

### "Energy jumps everywhere"
- Increase timestep initially, then refine
- Check transverse parameters
- Verify aperture configuration

### "Very slow"
- Reduce particle count for sweeps
- Disable self-consistency for initial exploration
- Use larger timesteps initially
- Consider parallel execution (when available)

## Integration with GUI

The optimization module can be integrated as a GUI plugin. The modular design allows:

- **Config generation** from GUI widgets
- **Progress callbacks** for real-time updates
- **Result visualization** in GUI windows
- **Interactive parameter exploration**

GUI integration example coming soon in `lw_integrator/gui_optimization_plugin.py`.

## Citation

If you use this optimization module in published work, please cite the LW Integrator project.

## Contributing

Contributions welcome! Areas for improvement:
- Parallel execution support
- Additional optimization algorithms
- Machine learning-based optimization
- Real-time visualization during sweeps
- GPU acceleration for metrics computation

## License

Same as main LW_integrator project.