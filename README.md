# Six-Mode Truncated-Wigner Simulation

This project simulates cavity-mediated atom-pair creation using a six-mode truncated-Wigner approximation (TWA). It contains two related models:

- a **single-pump model** based on the experiment introduced by Finger *et al.* in *Spin- and Momentum-Correlated Atom Pairs Mediated by Photon Exchange and Seeded by Vacuum Fluctuations*,
- a **two-pump / two-colour extension** for pair transfer between the $F=1$ and $F=2$ hyperfine manifolds of Rb-87, developed in the Master's thesis of **Kemal Önen** at ETH Zurich in the Quantum Optics Group.

The code samples initial conditions from the Wigner distribution of the initial quantum state, propagates each realization with `scipy.integrate.solve_ivp`, stores the complete trajectory ensemble, and evaluates (Weyl-corrected) observables in a separate analysis notebook.

## Project structure

```text
project_root/
├── run_simulation.py          # Select, run and save a simulation
├── run_phase_noise_sweep.sh   # Run a two-pump phase-noise parameter sweep
├── analysis.ipynb             # Load results and create analysis figures
├── requirements.txt           # Python dependencies
├── data/                      # Generated NPZ files
├── plots/                     # Exported figures
└── twa/
    ├── parameters.py          # Central user inputs, unit conversion and derived couplings
    ├── sampling.py            # Initial-state sampling considering vacuum fluctuations and technical noise
    ├── eoms.py                # Single- and two-pump equations of motion
    └── observables.py         # Populations, correlations and squeezing metrics```
```


## Modules and capabilities

### `parameters.py`

Contains the physics and simulation inputs. User settings are collected at the bottom in four blocks:

```python
COMMON
SINGLE_PUMP
TWO_PUMP
SIMULATION
```

The module also handles unit conversion and calculates the effective coefficients / rates used in the equations of motion.

### `sampling.py`

Generates the initial state for each trajectory. It includes:

- atom-number fluctuations,
- Wigner noise in occupied and empty modes,
- an optional classical seed in the pair modes,
- relative pump-phase noise for the two-pump model.

The implemented phase noise is shot-to-shot noise, i.e., one phase is sampled and then held fixed during each trajectory.

### `eoms.py`

Contains the equations of motions as solver-compatible functions:

```python
eoms_single_pump(...)
eoms_two_pump(...)
```

Both propagate six complex amplitudes in the same fixed mode order. Modifications to the existing models, as well as the implementation of additional models, are made in this module.

### `run_simulation.py`

Selects the model, samples and integrates the trajectory ensemble, reports progress, and stores the results as a compressed `.npz` file. Besides `parameters.py`, this is the second place for user input. Here, the user chooses which model to run and may adjust output-related settings such as filenames or storage paths.

### `observables.py`

Evaluates (Weyl-corrected) ensemble observables. For each pair channel, the module currently returns mode populations, pair number, population difference, imbalance variance, relative-number squeezing and the corresponding metrological gain, pair coherence, and the Cauchy–Schwarz ratio.

Additional observables can be implemented in this module without changing the propagation code.

### `analysis.ipynb`

Provides the post-processing workflow used for the project. It loads saved simulations and produces figures for:

- six-mode population dynamics,
- pair number and population imbalance,
- relative-number squeezing and metrological gain,
- Cauchy-Schwarz correlations,
- pair-coherence magnitude and phase,
- relative-phase-noise sweeps.

The notebook can analyse the `plus` channel, the `minus` channel, or both channels independently.

## How to use the code

### 1. Install the dependencies

It is recommended to install the dependencies in a virtual environment.

```bash
pip install -r requirements.txt
```

### 2. Choose the physical and numerical parameters

Edit the four input blocks at the bottom of `twa/parameters.py`.

Typical simulation parameters are

```python
SIMULATION = SimulationParameters(
    time_start_ms=0.0,
    time_stop_ms=0.2,
    number_time_points=2000,
    number_trajectories=1000,
    initial_pair_seed=0.0,
    random_seed=12345,
    solver_method="DOP853",
    absolute_tolerance=1e-10,
    relative_tolerance=1e-8,
)
```

Use fewer trajectories for quick tests and more trajectories for converged ensemble observables.

### 3. Select the model

At the top of `run_simulation.py`, set

```python
MODEL = "single_pump"
```

or

```python
MODEL = "two_pump"
```

Then run

```bash
python run_simulation.py
```

The result is written to `data/` together with its physical parameters, numerical settings, mode names, sampled atom numbers and, for the two-pump model, sampled relative phases.

### 4. Select one or both pair channels

After loading the trajectory array, calculate the observables with

```python
from twa.observables import compute_observables

observables = compute_observables(trajectories)
```

Inspect one channel with

```python
plus_channel = observables.pair_channels["plus"]
minus_channel = observables.pair_channels["minus"]
```

Thus the same simulation can be used to study:

- only the `plus` pair channel;
- only the `minus` pair channel;
- both channels and their relative behaviour.

For example,

```python
pair_number = minus_channel.pair_number.mean
population_difference = minus_channel.population_difference.mean
metrological_gain = minus_channel.metrological_gain
pair_coherence = minus_channel.pair_coherence.mean
```

### 5. Analyse and plot the results

Open

```bash
jupyter notebook analysis.ipynb
```

Set the data path in the loading cell, run the observable calculation, and select `"plus"` or `"minus"` in the relevant plotting cells. The notebook also contains the loader and grid plots used for the phase-noise sweep.

### 6. Run a phase-noise sweep

The supplied shell script runs the two-pump model for the mean relative phases $\Delta\phi = 0,\ \frac{\pi}{2},\ \pi$ and ten phase-noise widths between $0$ and $\pi$.

Before running the sweep, ensure that the two-pump model is selected in `run_simulation.py`, make the script executable and run

```bash
chmod +x run_phase_noise_sweep.sh
./run_phase_noise_sweep.sh
```

## Physical mode basis

| Index | Single-pump model | Two-pump model |
|---:|---|---|
| 0 | $(0,0)$ condensate | $(0,F=1,m_F=0)$ condensate |
| 1 | $(+k,+1)$ | $(+k,F=2,m_F=+1)$ |
| 2 | $(-k,-1)$ | $(-k,F=2,m_F=-1)$ |
| 3 | $(+k,-1)$ | $(+k,F=2,m_F=-1)$ |
| 4 | $(-k,+1)$ | $(-k,F=2,m_F=+1)$ |
| 5 | higher-order $(\pm 2k,0)$ condensate mode | higher-order $(\pm 2k,F=1,m_F=0)$ condensate mode |

The mode ordering must remain consistent across `sampling.py`, `eoms.py`, `run_simulation.py`, `observables.py`, and the analysis notebook.

## Current status and possible extensions

The code covers the full simulation workflow, from initial-state sampling and time evolution to data storage, observable calculation and plotting.

The single-pump model follows the established six-mode description and can simulate either pair-production channel separately or both at the same time.

The two-pump model can be used for exploratory studies, including phase-noise simulations. Its equations of motion should still be checked carefully, and particular care is required when choosing parameters for comparison with the single-pump model.

Possible future additions include further parameter-sweep scripts, parallelisation, additional modes, simpler toy models and new observables.


## Author

**Kemal Önen**  
ETH Zurich, Quantum Optics Group, Cavity Team

The two-pump extension and its effective many-body description were developed as part of the Master's thesis:

*Photon-Mediated Atom Pair Creation via Two-Colour Raman Processes in a Bose-Einstein Condensate*.

## References

1. F. Finger, R. Rosa-Medina, N. Reiter, P. Christodoulou, T. Donner, and T. Esslinger, “Spin- and Momentum-Correlated Atom Pairs Mediated by Photon Exchange and Seeded by Vacuum Fluctuations,” *Physical Review Letters* **132**, 093402 (2024). DOI: `10.1103/PhysRevLett.132.093402`.

2. K. Önen, *Photon-Mediated Atom Pair Creation via Two-Colour Raman Processes in a Bose-Einstein Condensate*, Master's thesis, ETH Zurich, Quantum Optics Group.
