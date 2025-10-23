"""
Analysis script for generic OLG model.

This script loads a trained model and generates:
1. Life-cycle profile plots (consumption, capital, Euler errors)
2. Calibration table

Usage:
    python analysis.py
"""

from packages import *
from dataset import DATASET
from econ_sim import ECONOMY_SIM
from nn import MODEL
from training import TRAIN

#===============================================================================
# SETUP
#===============================================================================

print('Loading trained model...')

#===============================================================================
# LOAD TRAINED MODEL
#===============================================================================

model = MODEL()
checkpoint = torch.load(
    model.par.modelSavePath + 'trained_model_params.pt',
)
model.load_state_dict(checkpoint['model_state'])
model.par.xbar = checkpoint.get('xbar', model.par.xbar)
model = model.to('cpu')
model.par.device = 'cpu'
model.par.traindevice = 'cpu'

print('Model loaded successfully')

#===============================================================================
# SIMULATE EQUILIBRIUM
#===============================================================================

print('Simulating equilibrium...')

sim = ECONOMY_SIM()
with torch.no_grad():
    # Long simulation to capture ergodic distribution
    model.par.T = 100000 + model.par.burn
    model.par.n = 1
    ds = DATASET(model)
    sim.economic_loss(model, ds.X)

print('Simulation complete')

#===============================================================================
# EXTRACT LIFE-CYCLE PROFILES
#===============================================================================

print('Extracting life-cycle profiles...')

# Transform from time series to cohort profiles
C_profile = sim.profiles(sim.C).squeeze().detach().cpu()  # (n_cohorts, I)
K_profile = sim.profiles(model.par.padAssets(sim.K, side=0)).squeeze().detach().cpu()
hhError_profile = sim.profiles(
    model.par.padAssets(sim.hhError, side=0)
).squeeze().detach().cpu()

# Average across cohorts for plotting
C_mean = C_profile.mean(0)
K_mean = K_profile.mean(0)
hhError_mean = hhError_profile.abs().mean(0)

print('Profiles extracted')

#===============================================================================
# LIFE-CYCLE PLOTS
#===============================================================================

print('Generating life-cycle plots...')

# Use results path from model parameters
resultsPath = model.par.resultsPath

# Plot consumption
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(C_mean, 'b-', linewidth=2)
ax.set_xlabel('Age')
ax.set_ylabel('Consumption')
ax.set_title('Life-Cycle Consumption Profile')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(resultsPath + 'consumption_profile.png', dpi=300)
plt.close()
print(f'  Saved: {resultsPath}consumption_profile.png')

# Plot capital
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(K_mean, 'r-', linewidth=2)
ax.set_xlabel('Age')
ax.set_ylabel('Capital Holdings')
ax.set_title('Life-Cycle Capital Profile')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(resultsPath + 'capital_profile.png', dpi=300)
plt.close()
print(f'  Saved: {resultsPath}capital_profile.png')

# Plot Euler errors
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(hhError_mean, 'g-', linewidth=2)
ax.set_xlabel('Age')
ax.set_ylabel('Euler Equation Error')
ax.set_title('Life-Cycle Euler Equation Errors')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(resultsPath + 'euler_errors_profile.png', dpi=300)
plt.close()
print(f'  Saved: {resultsPath}euler_errors_profile.png')

#===============================================================================
# CALIBRATION TABLE
#===============================================================================

print('Generating calibration table...')

# Extract parameters
I_val = model.par.I
β_val = model.par.β ** (model.par.I / 60)  # Annualized
γ_val = model.par.γ
α_val = model.par.α
ρ_val = model.par.ρ_annual
σ_annual_val = model.par.σ_annual
δ_val = model.par.δannual

# Extract aggregate statistics
R_mean = (sim.R - model.par.δ).mean().item()
R_std = (sim.R - model.par.δ).std().item()
Prod_mean = sim.Prod.mean().item()
K_mean = sim.Ksum.mean().item()
W_mean = sim.W.mean().item()

# Generate LaTeX table
latex_calibration = f"""\\begin{{table}}[ht]
\\centering
\\caption{{Model Calibration and Moments}}
\\label{{tab:calibration}}
\\begin{{tabular}}{{l c c l}}
\\toprule
\\textbf{{Description}} & \\textbf{{Symbol}} & \\textbf{{Value}} & \\textbf{{Notes}} \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\textbf{{Parameters}}}} \\\\
Periods of life                     & $I$              & ${I_val}$           & 60 periods \\\\
Discount factor (annual)            & $\\beta$          & ${β_val:.3f}$       & -- \\\\
Relative risk aversion              & $\\gamma$         & ${γ_val}$           & CRRA utility \\\\
Capital share                       & $\\alpha$         & ${α_val:.2f}$       & Cobb-Douglas \\\\
Depreciation rate (annual)          & $\\delta$         & ${δ_val:.2f}$       & -- \\\\
TFP persistence (annual)            & $\\rho_Z$         & ${ρ_val:.2f}$       & AR(1) \\\\
TFP std dev (annual)                & $\\sigma_Z$       & ${σ_annual_val*100:.1f}\\%$ & -- \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\textbf{{Model Moments}}}} \\\\
Mean return (annual)                & $E[R-\\delta]$    & ${R_mean * (model.par.I/60)*100:.2f}\\%$  & Net of depreciation \\\\
Std dev return (annual)             & $\\sigma(R)$      & ${R_std * np.sqrt(model.par.I/60)*100:.2f}\\%$   & Annualized \\\\
Mean output                         & $E[Y]$           & ${Prod_mean:.3f}$   & -- \\\\
Mean capital                        & $E[K]$           & ${K_mean:.3f}$      & -- \\\\
Mean wage                           & $E[w]$           & ${W_mean:.3f}$      & -- \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

# Save to file
with open(resultsPath + 'calibration.tex', 'w') as f:
    f.write(latex_calibration)

print(f'  Saved: {resultsPath}calibration.tex')

#===============================================================================
# SUMMARY STATISTICS
#===============================================================================

print('\\n' + '='*60)
print('SUMMARY STATISTICS')
print('='*60)
print(f'Mean consumption:        {C_mean.mean():.3f}')
print(f'Mean capital:            {K_mean:.3f}')
print(f'Mean Euler error:        {hhError_mean.mean():.3e}')
print(f'Max abs Euler error:     {hhError_mean.abs().max():.3e}')
print(f'Mean return (annual):    {R_mean * (model.par.I/60):.2%}')
print(f'Std return (annual):     {R_std * np.sqrt(model.par.I/60):.2%}')
print(f'Mean output:             {Prod_mean:.3f}')
print(f'Mean wage:               {W_mean:.3f}')
print('='*60)

#===============================================================================
# REGENERATE TRAINING LOSS PLOT
#===============================================================================

print('\\nRegenerating training loss plot...')

# Load training losses from CSV
losses_file = resultsPath + 'training_losses.csv'
if os.path.exists(losses_file):
    losses = np.loadtxt(losses_file, delimiter=',', skiprows=1)

    # Create trainer instance and plot losses
    trainer = TRAIN(model)
    trainer.plot_losses(losses)
    print(f'  Saved: {resultsPath}training_losses.png')
else:
    print(f'  Warning: {losses_file} not found. Skipping loss plot.')

print('\\nAnalysis complete!')
print(f'Results saved to: {resultsPath}')
