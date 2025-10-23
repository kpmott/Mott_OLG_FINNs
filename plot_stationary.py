#!/home/kpmott/Git/olg_nn_studentdebt/stdebt/bin/python3
"""
Diagnostic plotting for stationary equilibrium analysis.

This module generates comprehensive diagnostic dashboards showing:
- Training loss curves and equilibrium error decomposition
- Life-cycle profiles (consumption, assets, Euler errors, consumption volatility)
- Policy functions (savings as a function of aggregate state)
- Time series of key aggregates (production, capital, returns, taxes)

The dashboard provides a visual snapshot of model performance and economic
outcomes at any point during training. Used for monitoring convergence and
validating equilibrium properties.

Output: Multi-panel PNG dashboard saved to ./results/
"""

from packages import *
from nn import MODEL
from dataset import DATASET
from econ_sim import ECONOMY_SIM

class PLOTS_STAT(ECONOMY_SIM):
    """
    Generate diagnostic plots for stationary equilibrium.

    Inherits from ECONOMY_SIM to access equilibrium computation methods.
    Creates a comprehensive dashboard with 4 main sections:
    1. Loss curves (training progress)
    2. Life-cycle profiles (age-specific behavior)
    3. Policy functions (aggregate savings rules)
    4. Time series (macro aggregates)
    """

    def __init__(self,model,episode=None,losses=[]):
        """
        Initialize plotting class with model and training history.

        Args:
            model: Trained MODEL instance with policy function
            episode (int, optional): Current episode number for labeling.
                If None, inferred from length of losses. Defaults to None.
            losses (list, optional): Training loss history. Defaults to [].
        """
        super().__init__()
        self.model = model
        self.losses = losses
        if episode == None:
            self.episode = len(losses)
        else:
            self.episode = episode

    def Plots(self):
        """
        Generate comprehensive diagnostic dashboard.

        Creates a multi-panel figure with:
        - Top left: Training loss curve (log scale) and error decomposition bar chart
        - Bottom left: Life-cycle profiles (C, K, Euler errors, C volatility)
                       Policy functions (K' vs K, K' vs Z scatter plots)
        - Right side: Time series of macro aggregates (Z, Prod, K, returns, taxes, etc.)

        Saves:
        - Model checkpoint (for resuming training)
        - Loss history CSV
        - Dashboard PNG to results/ directory (overwrites each time)

        The dashboard uses dark theme with color coding for agent types.
        """

        # Save model checkpoint
        torch.save(
            {
                'model_state': self.model.state_dict(),
                'xbar': self.model.par.xbar
            },
            self.model.par.modelSavePath+'trained_model_params.pt'
        )

        #-----------------------------------------------------------------------
        # PLOT CONFIGURATION
        # Style parameters and output paths
        #-----------------------------------------------------------------------

        plottime = slice(-100, -1, 1)  # Last 100 periods for time series

        # Dark theme
        plt.style.use('dark_background')
        plt.rcParams['axes.prop_cycle'] = cycler(color=['r', 'b', 'w'])

        # Output directory (use results path directly)
        resultsdir = self.model.par.resultsPath

        # Save loss history to CSV
        with open(resultsdir+'losses.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(self.losses)

        #-----------------------------------------------------------------------
        # DATA PREPARATION
        # Extract life-cycle profiles and compute summary statistics
        #-----------------------------------------------------------------------

        # Life-cycle profiles (cohort dimension): shape (cohorts, age)
        self.Clife = self.profiles(self.C)  # Consumption
        # Annualized consumption volatility (coefficient of variation)
        self.Cvol = self.Clife.var(0,True).sqrt() * np.sqrt(self.model.par.I/60) / self.Clife.mean(0,True)
        self.Klife = self.profiles(self.K)  # Assets
        self.hhEulerlife = self.profiles(self.hhError).abs()  # Euler errors

        # Variables to plot in life-cycle panel
        lifecycle_vars = [
            ('C', self.Clife),
            ('Cvol', self.Cvol),
            ('K', self.Klife),
            ('hhError', self.hhEulerlife)
        ]

        # Policy function scatter plots (savings vs aggregate state)
        policy_vars = ['K_vs_Ksum', 'Z_vs_Ksum']

        # Loss decomposition labels and values
        lossLabels = [
            'Capital',      # Euler equation errors
            'Penalties'     # Feasibility violations
        ]
        losses_bar = torch.concat([
            self.hhError.pow(2).mean().sqrt()[None],  # RMS Euler error
            self.cpen.pow(2).mean().sqrt()[None]      # RMS feasibility error
        ], 0)

        # Time series variables to plot
        time_series_vars = [
            'Z',        # TFP
            'Prod',     # Output
            'Csum',    # Aggregate consumption
            'Kpsum',    # Aggregate capital
            'I',        # Investment
            'rets'     # Returns (equity & bond)
        ]

        #-----------------------------------------------------------------------
        # FIGURE LAYOUT
        # Create multi-panel dashboard with nested GridSpec
        #-----------------------------------------------------------------------

        # Master grid: 2 columns (left=losses+profiles+policies, right=time series)
        fig = plt.figure(figsize=(22, 15), constrained_layout=True)
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2], figure=fig)

        # Left side: Split into losses (top) and profiles+policies (bottom)
        gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], height_ratios=[0.75, 2.25])

        # Top left: Loss curve and error bar chart side-by-side
        gs_top_left = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_left[0], width_ratios=[2, 1])
        ax_loss_curve = fig.add_subplot(gs_top_left[0])
        ax_error_bar = fig.add_subplot(gs_top_left[1])

        # Bottom left: Profiles (top) and policy functions (bottom)
        gs_bottom_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_left[1], height_ratios=[3, 2])

        # Life-cycle profiles: 2x2 grid
        num_profiles = len(lifecycle_vars)
        profile_cols = 2
        profile_rows = (num_profiles + 1) // profile_cols
        gs_profiles = gridspec.GridSpecFromSubplotSpec(profile_rows, profile_cols, subplot_spec=gs_bottom_left[0])
        axs_profiles = [fig.add_subplot(gs_profiles[i, j]) for i in range(profile_rows) for j in range(profile_cols)][:num_profiles]

        # Policy functions: 2x1 grid (K' vs K, K' vs Z)
        num_policies = len(policy_vars)
        gs_policies = gridspec.GridSpecFromSubplotSpec(num_policies, 1, subplot_spec=gs_bottom_left[1])
        axs_policies = [fig.add_subplot(gs_policies[i, 0]) for i in range(num_policies)]

        # Right side: Time series in dynamic grid
        num_time_series = len(time_series_vars)
        num_columns = 2
        num_rows = (num_time_series + 1) // num_columns
        gs_time_series = gridspec.GridSpecFromSubplotSpec(num_rows, num_columns, subplot_spec=gs[1])
        axs_time_series = [fig.add_subplot(gs_time_series[i, j]) for i in range(num_rows) for j in range(num_columns)][:num_time_series]

        #-----------------------------------------------------------------------
        # PLOT 1: TRAINING LOSSES
        # Loss curve (left) and error decomposition (right)
        #-----------------------------------------------------------------------

        # Loss trajectory over training episodes (log scale)
        ax_loss_curve.plot(self.losses)
        ax_loss_curve.set_title(f'Training Loss: {self.losses[-1]:.2e}')
        ax_loss_curve.set_yscale('log')
        ax_loss_curve.set_xlabel('Episode')

        # Bar chart of equilibrium error components (Capital=Euler, Penalties=Feasibility)
        x = np.arange(len(lossLabels))
        ax_error_bar.bar(x, losses_bar.cpu().detach())
        ax_error_bar.set_xticks(x)
        ax_error_bar.set_xticklabels(lossLabels)
        ax_error_bar.set_title('Equilibrium Errors')

        #-----------------------------------------------------------------------
        # PLOT 2: LIFE-CYCLE PROFILES
        # Consumption, assets, Euler errors, consumption volatility by age
        #-----------------------------------------------------------------------

        # Plot each variable by age (averaged over cohorts)
        for ax, (label, data) in zip(axs_profiles, lifecycle_vars):
            ax.plot(
                data.mean(0).detach().cpu(),  # Average over cohorts, shape (age,)
                linestyle='-',
                color='cyan'
            )
            ax.set_xlabel('Age')
            # Format consumption volatility as percentage
            if label == 'Cvol':
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            ax.set_title(label)

        #-----------------------------------------------------------------------
        # PLOT 3: POLICY FUNCTIONS
        # Aggregate savings as function of aggregate capital and TFP
        #-----------------------------------------------------------------------

        # Extract data for scatter plots
        x_K = self.Ksum.cpu().numpy()     # Current aggregate capital
        x_Z = self.Z.cpu().numpy()        # TFP shock
        y_Ksum = self.Kpsum.cpu().numpy() # Next period aggregate capital

        # Plot 1: K' vs K (colored by Z)
        sc1 = axs_policies[0].scatter(x_K, y_Ksum, c=x_Z, cmap='coolwarm', s=10, alpha=0.8)
        axs_policies[0].set_xlabel(r'$K$')
        axs_policies[0].set_title(r'$K^\prime$')

        # Plot 2: K' vs Z (colored by Z)
        sc2 = axs_policies[1].scatter(x_Z, y_Ksum, c=x_Z, cmap='coolwarm', s=10, alpha=0.8)
        axs_policies[1].set_xlabel(r'$Z$')
        axs_policies[1].set_title(r'$K^\prime$')

        scatters = [sc1, sc2]

        # Shared colorbar for TFP (red=high productivity, blue=low productivity)
        cb = fig.colorbar(scatters[0], ax=axs_policies, orientation='vertical', shrink=0.8, pad=0.01)
        cb.set_label(r'$Z$', rotation=0, labelpad=15)
        scatters[0].set_cmap('coolwarm')

        #-----------------------------------------------------------------------
        # PLOT 4: TIME SERIES
        # Macro aggregates over last 100 periods
        #-----------------------------------------------------------------------

        for ax, var in zip(axs_time_series, time_series_vars):
            # Special handling for capital (show K/Y ratio in title)
            if var == 'Kpsum':
                ax.plot(self.Kpsum[plottime].cpu())
                ax.set_title(
                    r'Forward Capital (K/Y = '+f'{
                        (
                            (self.Ksum/(self.Prod*self.model.par.I/60)).mean().cpu().item()
                        ):.2f
                    }'+')'
                )
            # Special handling for returns (plot equity and bond, show volatility)
            elif var == 'rets':
                Re = self.model.par.annualize(self.R - self.model.par.δ)[1:]  # Equity return
                σe = Re.std()*np.sqrt(self.model.par.I/60)          # Annualized volatility
                # Rb = self.model.par.annualize(1/self.Q - 1)[:-1]    # Risk-free rate
                ax.plot(Re[plottime].detach().cpu(),'r-',label='Equity')
                # ax.plot(Rb[plottime].detach().cpu(),'b-',label='Bond')
                ax.set_title(
                    r'Returns: $\sigma_e=$'+f'{σe.mean().item():.2%}'
                )
                # ax.legend()
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            else:
                ax.plot(getattr(self, var)[plottime].cpu())
                ax.set_title(var)
            ax.set_xticks([])
            ax.set_xlabel(r'$t$')

        #-----------------------------------------------------------------------
        # SAVE FIGURE
        # Save to results directory (overwrites each time)
        #-----------------------------------------------------------------------

        plotloc = resultsdir+'dashboard.png'
        fig.savefig(plotloc)
        plt.close(fig)
        plt.clf()

    def PrintPlots(self):
        """
        Wrapper to simulate economy and generate plots.

        This method:
        1. Simulates the economy forward using current policy (DATASET)
        2. Computes equilibrium given simulated data (economic_loss)
        3. Generates diagnostic dashboard (Plots)

        All operations performed without gradient tracking (inference mode).
        """
        with torch.no_grad():
                # Simulate economy on CPU
                self.model = self.model.to(self.model.par.device)
                ds = DATASET(self.model)

                # Compute equilibrium on GPU
                self.model = self.model.to(self.model.par.traindevice)
                self.X = ds.X.to(self.model.par.traindevice)
                self.economic_loss(self.model,self.X)

                # Generate plots
                self.Plots()