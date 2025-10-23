# Generic OLG Model with Neural Network Policy Iteration

A computational framework for solving overlapping generations (OLG) models with capital accumulation and aggregate productivity shocks using neural network policy iteration.

## Overview

This repository implements a generic OLG model where:
- **Agents** are homogeneous and live for `I` periods
- **Capital** accumulates through savings decisions
- **Production** uses Cobb-Douglas technology with TFP shocks
- **Policy function** (savings rule) is approximated using a feedforward neural network
- **Training** uses policy iteration: simulate economy → train on Euler errors → repeat

The model is designed to be easily adapted for specific research questions by replacing the income calibration in `datawork.py`.

## Model Structure

### Economic Environment
- **Demographics**: `I=60` age cohorts (homogeneous agents)
- **Preferences**: CRRA utility with risk aversion `γ=3`, discount factor `β=0.95^(60/I)`
- **Technology**: `Y = Z * K^α * L^(1-α)` with `α=0.35`
- **Depreciation**: Constant rate `δ=0.1` (annual)
- **TFP Process**: AR(1) with `ρ=0.9`, `σ=0.035` (annual)

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
├── dashboard.py       # Main training script
├── training.py        # Policy iteration training loop
├── dataset.py         # Forward simulation for training data generation
├── econ_sim.py        # Equilibrium computation and Euler equation errors
├── nn.py              # Neural network architecture
├── parameters.py      # Economic parameters and utility functions
├── datawork.py        # Income profile generation (REPLACE THIS)
├── analysis.py        # Post-training analysis and plots
├── packages.py        # Common imports
├── modelsave/         # Model checkpoints (created automatically)
└── results/           # Analysis outputs (created automatically)
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kpmott/mott_olg_nn.git
   cd mott_olg_nn
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch numpy pandas matplotlib scipy tqdm torchinfo torchsummary
   ```

## Usage

### Basic Training

Run the main training script:
```bash
python dashboard.py
```

This will:
1. Initialize the neural network policy function
2. Train for 10,000 episodes across 2 regimes (5,000 each)
3. Save checkpoints after each regime to `./modelsave/trained_model_params.pt`

### Post-Training Analysis

**Note**: This repository includes pre-generated results from the baseline model in `./results/` and `./modelsave/`. You can view these immediately, or regenerate them after making modifications.

To regenerate results after training or modifying parameters:
```bash
python analysis.py
```

This will:
1. Load the trained model from `./modelsave/trained_model_params.pt`
2. Simulate the equilibrium (100k periods)
3. Generate life-cycle plots (consumption, capital, Euler errors)
4. Create a calibration table with parameters and model moments
5. Overwrite existing outputs in `./results/`

**Outputs** (included in this repo, can be regenerated):
- `results/consumption_profile.png` - Life-cycle consumption
- `results/capital_profile.png` - Life-cycle capital holdings
- `results/euler_errors_profile.png` - Euler equation errors by age
- `results/calibration.tex` - LaTeX table with parameters and moments

### Customizing Training

Edit `dashboard.py` to adjust hyperparameters:

```python
num_regimes = 2                              # Number of training regimes
episodes = [range(5000), range(5000)]        # Episodes per regime
batches = [200, 500]                         # Batch sizes
lrs = [1e-6, 1e-7]                          # Learning rates
weight_decay = [0, 0]                        # L2 regularization
```

### Customizing the Model

1. **Income profiles**: Edit `datawork.py` to load your own calibrated income data
2. **Parameters**: Modify `parameters.py` to change:
   - Demographics (`I`, number of periods)
   - Preferences (`β`, `γ`)
   - Technology (`α`, `δ`)
   - TFP process (`ρ_z`, `σ_z`)
3. **Network architecture**: Adjust layer sizes in `nn.py`

## Training Algorithm

The model uses **policy iteration**:

```
for episode in episodes:
    1. Simulate economy forward T periods with current policy
    2. For each state in simulated data:
       a. Compute equilibrium (prices, consumption, investment)
       b. Evaluate Euler equation errors
    3. Update policy via gradient descent to minimize errors
    4. Save checkpoint
```

### Loss Function

The training minimizes:
```
Loss = log(1 + ||Euler errors||² + ||Feasibility errors||²)
```

where:
- **Euler errors**: Violations of `u'(c_t) = β * E[u'(c_{t+1}) * (1 + R_{t+1} - δ)]`
- **Feasibility errors**: Violations of `C + I = Y` (resource constraint)

## Key Features

- **GPU acceleration**: Automatically uses CUDA if available
- **Policy iteration**: Re-simulates economy each episode for stable convergence
- **Fisher-Burmeister**: Handles complementarity constraints (savings ≥ 0)
- **Gauss-Hermite quadrature**: Accurate integration of expectations (15 nodes)
- **Progressive training**: Multiple regimes with decreasing learning rates
- **Checkpoint saving**: Model saved after each regime

## Adapting for Your Research

This is a **generic template**. To adapt it for your research:

1. **Replace income calibration**: Edit `datawork.py` to load your own income/productivity data

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

## Output Files

**Note**: Pre-generated outputs from the baseline model are included in this repository for reference.

### Training Outputs
- **Model checkpoint**: `./modelsave/trained_model_params.pt` *(included)*
  - Contains: `model_state` (neural network weights), `xbar` (normalization)
- **Loss trajectories**: Returned by `train()` method

### Analysis Outputs (in `./results/`, all included)
- **Life-cycle plots**: `./results/*.png`
  - `income_profile.png` - Labor income efficiency units (generated when loading model)
  - `consumption_profile.png` - Consumption over the life cycle
  - `capital_profile.png` - Asset holdings over the life cycle
  - `euler_errors_profile.png` - Policy function accuracy by age
- **Calibration table**: `./results/calibration.tex`
  - Model parameters (β, γ, α, δ, ρ, σ)
  - Simulated moments (returns, output, capital, wages)

To regenerate any outputs after modifications, simply run `python dashboard.py` (for training) or `python analysis.py` (for plots/tables).

## Technical Notes

- **Normalization**: Inputs normalized by steady-state values for training stability
- **Boundary conditions**: Youngest agents born with zero assets, oldest die with zero assets
- **Ergodic distribution**: Training data comes from simulated ergodic distribution
- **No borrowing**: Complementarity constraint enforces `k' ≥ 0`

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mott_olg_nn,
  author = {Mott, Kevin},
  title = {Generic OLG Model with Neural Network Policy Iteration},
  year = {2025},
  url = {https://github.com/kpmott/mott_olg_nn}
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
