# Generic OLG Model with Finance-Informed Neural Networks (FINNs)

A computational framework for solving overlapping generations (OLG) models with capital accumulation and aggregate productivity shocks using Finance-Informed Neural Networks (FINN) and policy iteration.

## Overview

This repository implements a generic stochastic OLG model using **Finance-Informed Neural Networks (FINN)**:
- **Agents** are homogeneous and live for `I` periods
- **Capital** accumulates through savings decisions
- **Production** uses Cobb-Douglas technology with TFP shocks
- **Policy function** (savings rule) is approximated using a feedforward neural network
- **Training** uses policy iteration with finance-informed loss penalties
- **FINN approach**: Economic constraints (Euler equations, feasibility, capital dynamics) directly inform the loss function

The FINN framework satisfies equilibrium conditions and prevents pathological solutions by penalizing economically unrealistic behavior (e.g., zero capital adjustment). This approach has been successfully applied in:
- Mott (2025) "Student Loan (Forgiveness) in General Equilibrium"
- Mott (2025) "Real and Asset Pricing Effects of Employer Retirement Matching"

Suggested citations for these papers are available below.

The model is designed to be easily adapted for specific research questions by replacing the income calibration in `datawork.py`, model parameters in `parameters.py`, data creation in `dataset.py`, and equilibrium allocations in `econ_sim.py`.

## Key Features

- **Finance-Informed Neural Networks (FINN)**: Equilibrium constraints and economic penalties directly inform training
- **Capital adjustment penalty**: Prevents pathological constant-capital solutions
- **Real-time diagnostics**: Comprehensive dashboard generated every 100 episodes during training
- **GPU acceleration**: Automatically uses CUDA if available
- **Policy iteration**: Re-simulates economy each episode for stable convergence
- **Fisher-Burmeister**: Handles complementarity constraints (savings ≥ 0)
- **Gauss-Hermite quadrature**: Accurate integration of expectations (15 nodes)
- **Automatic plotting**: All results save to `./results/` with overwrites (no clutter)
- **Checkpoint saving**: Model continuously saved during training
- **Normalization + Tanh activation**: Inputs normalized by steady-state values to keep values inside the gradient zone for tanh activation, which is an excellent choice for continuous policy functions (smooth, bounded derivatives)
- **Ergodic distribution**: Training data comes from simulated ergodic distribution (grid-free approach)

## Model Structure

### Economic Environment
- **Demographics**: `I=60` age cohorts (homogeneous agents)
- **Preferences**: CRRA utility with risk aversion `γ=3`, discount factor `β=0.95^(60/I)`
- **Technology**: `Y = Z * K^α * L^(1-α)` with `α=0.35`
- **Depreciation**: Constant rate `δ=0.10` (annual)
- **TFP Process**: AR(1) with `ρ=0.9`, `σ=0.025` (annual)

### State Space
- **Aggregate state**: `[k, z]` where
  - `k`: Capital holdings by age cohort (vector of length `I-1`)
  - `z`: Total factor productivity (scalar)

### Neural Network Policy
- **Input**: Aggregate state `[k, z]` (dimension: `I-1+1 = 60`)
- **Architecture**: 2 hidden layers (150 neurons each) with Tanh activation
- **Output**: Next period capital `k'` (dimension: `I-1 = 59`)
- **Activation**: Softplus on output (ensures non-negative savings)

## Repository Structure

```
.
├── dashboard.py                      # Main training script
├── training.py                       # Policy iteration training loop
├── dataset.py                        # Forward simulation for training data generation
├── econ_sim.py                       # Equilibrium computation and Euler equation errors
├── nn.py                             # Neural network architecture
├── parameters.py                     # Economic parameters and utility functions
├── datawork.py                       # Income profile generation (REPLACE THIS)
├── plot_stationary.py                # Real-time diagnostic plotting during training
├── analysis.py                       # Post-training analysis and plots
├── packages.py                       # Common imports
├── modelsave/                        # Model checkpoints
│   └── trained_model_params.pt       # Trained neural network weights
└── results/                          # All outputs (plots, tables, losses)
    ├── dashboard.png                 # Real-time training dashboard (updated every 100 episodes)
    ├── losses.csv                    # Training loss history
    ├── training_losses.png           # Training loss plot
    ├── training_losses.csv           # Training loss data
    ├── calibration.tex               # LaTeX parameter table
    ├── income_profile.png            # Life-cycle income profile
    ├── consumption_profile.png       # Life-cycle consumption profile
    ├── capital_profile.png           # Life-cycle capital profile
    └── euler_errors_profile.png      # Euler equation errors by age
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kpmott/mott_olg_nn.git
   cd mott_olg_nn
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   python3 -m pip install --upgrade pip
   python3 -m pip install torch numpy pandas matplotlib scipy tqdm torchinfo torchsummary
   ```

## Usage

### Basic Training

Run the main training script:
```bash
python dashboard.py
```

This will:
1. Initialize the neural network policy function
2. Train for 20,000 episodes with adaptive learning rates
3. Generate comprehensive diagnostic dashboards every 100 episodes
4. Save all outputs to `./results/` (plots, losses, calibration table)
5. Save model checkpoint to `./modelsave/trained_model_params.pt`

**Real-time monitoring**: During training, the dashboard plot (`./results/dashboard.png`) updates every 100 episodes, showing:
- Training loss convergence
- Equilibrium error decomposition
- Life-cycle profiles (consumption, assets, Euler errors)
- Policy functions (savings behavior)
- Time series of macro aggregates

### Post-Training Analysis

To regenerate detailed life-cycle plots after training:
```bash
python analysis.py
```

This creates individual high-resolution plots:
- `results/consumption_profile.png` - Life-cycle consumption
- `results/capital_profile.png` - Life-cycle capital holdings
- `results/euler_errors_profile.png` - Euler equation errors by age
- `results/income_profile.png` - Labor income efficiency units
- `results/calibration.tex` - LaTeX table with parameters and moments

### Customizing Training

Edit `dashboard.py` to adjust hyperparameters:

```python
train = TRAIN(model)
losses = train.train(
    episodes=20000,
    batchsize=500,
    lr=1e-6
)
```

Parameters:
- `episodes`: Total number of policy iterations
- `batchsize`: Mini-batch size for gradient descent
- `lr`: Learning rate for Adam optimizer

### Customizing Neural Network Architecture

Edit `nn.py` to adjust the network architecture. This is where the neural network structure is defined and can be made more or less flexible depending on your problem complexity:

**Current architecture choices (justified)**:
- **Input normalization**: All inputs are normalized as `x / (1 + xbar)` where `xbar` are steady-state values. Dividing by `(1 + xbar)` rather than `xbar` alone keeps the formula well-defined and agnostic to cases where `xbar = 0`. This normalization is critical for keeping values roughly in the [-3, 3] range, which is the "sweet spot" for tanh activation where gradients are strong. Without normalization, capital values in the tens (e.g., K=24) would push tanh into saturation regions where ∂tanh/∂x ≈ 0, crippling gradient-based learning. **Crucially, the normalization happens inside the `nn.py` module** - you pass in raw state variables `x` and normalization occurs within the forward pass. This means PyTorch's autodifferentiation automatically captures the normalization in the computation graph; no manual gradient adjustments needed.
- **Hidden layers**: `[150, 150]` - Moderate size provides good approximation power without excessive parameters. Increase for more complex models (e.g., heterogeneous agents), decrease for simpler models or faster training.
- **Activation function (hidden layers)**: `Tanh` - Excellent choice for continuous policy functions. Provides smooth, bounded derivatives with good gradient flow when combined with proper normalization. Tanh is symmetric around zero and smooth, making it ideal for approximating economic decision rules.
- **Output activation**: `Softplus` - Ensures non-negative capital holdings (k ≥ 0) without hard constraints. Smooth alternative to ReLU that provides everywhere-differentiable policy functions. **Other activations can be substituted** depending on economic constraints. 

## Training Algorithm

The FINN model uses **policy iteration** with automatic diagnostic plotting. The training loop is implemented in `training.py`, while equilibrium calculations that map state variables and budget constraints into the loss function are in `econ_sim.py`:

```
for episode in episodes:
    1. Simulate economy forward T periods with current policy (dataset.py)
    2. For each state in simulated data:
       a. Compute equilibrium: prices, consumption, investment (econ_sim.py)
       b. Evaluate Euler equation errors and feasibility violations (econ_sim.py)
       c. Compute capital adjustment penalty: 1e-3/σ(K') (econ_sim.py)
    3. Update policy via gradient descent to minimize FINN loss (training.py)
    4. Every 100 episodes: Generate diagnostic dashboard (plot_stationary.py)
```

**Key implementation files**:
- `training.py`: Main policy iteration loop, optimizer, batching
- `econ_sim.py`: Equilibrium calculations mapping state variables (K, Z) through budget constraints into consumption, then computing Euler equation errors for the loss function
- `dataset.py`: Forward simulation generating training data
- `plot_stationary.py`: Diagnostic dashboard generation

### FINN Loss Function

The FINN training minimizes:
```
Loss = log(1 + ||Euler errors||² + ||Feasibility penalty||²)
```

where:
- **Euler errors**: Violations of `u'(c_t) = β * E[u'(c_{t+1}) * (1 + R_{t+1} - δ)]`
- **Feasibility penalty**: `100 * (C + I - Y)/Y + 1e-3/σ(K')`

The **capital adjustment penalty** `1e-3/σ(K')` is critical for preventing pathological solutions. Neural networks can exploit the flexibility of the model by finding "solutions" where the capital stock never adjusts (constant K across all states). While this trivially satisfies equilibrium conditions, it's economically meaningless. The penalty term forces the network to learn policies with realistic capital dynamics by penalizing low variance in aggregate capital holdings.

## Model Results

The baseline calibration produces the following equilibrium outcomes:

### Calibrated Parameters
| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Periods of life | I | 60 | 60 periods |
| Discount factor (annual) | β | 0.950 | Time preference |
| Risk aversion | γ | 3 | CRRA utility |
| Capital share | α | 0.35 | Cobb-Douglas |
| Depreciation (annual) | δ | 0.10 | Capital depreciation |
| TFP persistence (annual) | ρ_Z | 0.90 | AR(1) process |
| TFP std dev (annual) | σ_Z | 2.5% | Productivity shocks |

### Equilibrium Moments
| Moment | Value | Notes |
|--------|-------|-------|
| Mean return (annual) | 9.68% | Net of depreciation |
| Return volatility (annual) | 0.50% | Annualized std dev |
| Capital-output ratio | 1.78 | K/Y |
| Mean output | 13.65 | Aggregate production |
| Mean capital | 24.30 | Aggregate capital stock |
| Mean wage | 8.87 | Wage rate |
| Training loss (final) | 1.30e-06 | Converged value |

### Life-Cycle Behavior
- **Consumption**: Monotonically increasing over the life cycle (from 0.09 to 0.28)
- **Capital accumulation**: Hump-shaped profile peaking around age 45 (max ≈ 1.0)
- **Euler errors**: Extremely small throughout (< 0.0014), indicating high accuracy

See `./results/dashboard.png` for comprehensive visualizations.



## Adapting for Your Research

This is a **generic template** on an unrealistically simple model. To adapt it for your research:

1. **Data calibration and estimation**: Edit `datawork.py` to estimate/calibrate any necessary data (income profiles, wealth distributions, etc.)
   - Current implementation uses a hump-shaped income profile peaking at age 30
   - **Best practice**: Precompute and store calibrated data as CSVs, then load from disk during training
   - This significantly speeds up training and fine-tuning by avoiding repeated estimation/calibration
   - Example: Estimate income profiles from PSID/CPS once, save to `./data/income_profile.csv`, then load in `datawork.py`

2. **Modify state variables**: To add or change state variables (e.g., government debt, additional shocks):
   - Update dimensions and slices in `parameters.py` (`self.input`, `self.output`, state slices)
   - Place new variables in correct position when building states in `dataset.py`
   - Carefully construct the forward forecast in `econ_sim.py` (inside the `with torch.no_grad()` block)

3. **Adjust equilibrium calculations**: Modify `econ_sim.py` to match your model:
   - Budget constraints (consumption calculation)
   - First-order conditions (Euler equations)
   - Market clearing conditions
   - Any policy functions (taxes, transfers, etc.)

4. **Update parameters**: Change economic parameters in `parameters.py` as needed (`β`, `γ`, `α`, `δ`, shock processes, etc.)

5. **Customize plotting**: Edit `plot_stationary.py` to add/remove diagnostic plots
   - Current dashboard includes losses, profiles, policies, and time series
   - Dark theme with cyan/red color scheme

## Output Files

All outputs save to `./results/` and overwrite on each run (no accumulation of old files).

### Training Outputs
- **Model checkpoint**: `./modelsave/trained_model_params.pt`
  - Contains: `model_state` (neural network weights), `xbar` (normalization)
- **Training dashboard**: `./results/dashboard.png` (updated every 100 episodes)
- **Loss history**: `./results/losses.csv` and `./results/training_losses.csv`

### Analysis Outputs
- **Life-cycle plots**: Individual high-resolution PNGs
  - `income_profile.png` - Labor income efficiency units
  - `consumption_profile.png` - Consumption over the life cycle
  - `capital_profile.png` - Asset holdings over the life cycle
  - `euler_errors_profile.png` - Policy function accuracy by age
- **Calibration table**: `calibration.tex` (LaTeX table with parameters and moments)

## Performance Benchmark

Training time for 20,000 episodes with diagnostic plotting every 100 episodes:

**Duration**: 39 minutes, 17 seconds

**Hardware**:
- **CPU**: Intel Core i9-12900KF (24 cores) @ 5.1 GHz
- **GPU**: NVIDIA GeForce RTX 3080 (LHR)
- **RAM**: 32 GB
- **OS**: Ubuntu 24.04.3 LTS

Expected training times will scale roughly with:
- GPU memory bandwidth (critical for large batch processing)
- CPU single-thread performance (for simulation between episodes)
- Number of episodes and diagnostic plotting frequency

Systems without CUDA-capable GPUs will run significantly slower (CPU-only mode supported but not recommended for production use).

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mott_olg_finn,
  author = {Mott, Kevin P.},
  title = {Generic OLG Model with Finance-Informed Neural Networks (FINN)},
  year = {2025},
  url = {https://github.com/kpmott/mott_olg_nn}
}
```

### Papers Using This Framework

If you're interested in applications of the FINN framework to specific economic questions:

```bibtex
@article{mott2025studentloan,
  author = {Mott, Kevin P.},
  title = {Student Loan (Forgiveness) in General Equilibrium},
  year = {2025},
  note = {Working Paper}
}

@article{mott2025retirement,
  author = {Mott, Kevin P.},
  title = {Real and Asset Pricing Effects of Employer Retirement Matching},
  year = {2025},
  note = {Working Paper}
}
```

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

**You are free to:**
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

**Under the following terms:**
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made
- **NonCommercial** — You may not use the material for commercial purposes

This license allows academics and researchers to freely use, modify, and extend this code for non-commercial research purposes, provided appropriate citation is given. For commercial use inquiries, please contact the author.

## Contact

Kevin P. Mott

[Professional website](https://kevinpmott.com/)

For bug reports or questions about the code, please open an issue on the GitHub repository.
