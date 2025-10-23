u"""
Economic simulation and equilibrium computation for generic OLG model.

This module defines the ECONOMY_SIM class which simulates the equilibrium dynamics
of the overlapping generations economy. It computes:
- Aggregate production and factor prices (wages, interest rates)
- Household consumption and savings
- Government taxes and Social Security transfers (if enabled)
- Euler equation residuals and feasibility constraints
- Loss function for neural network training
"""

from packages import *

class ECONOMY_SIM():
    """
    Simulate OLG economy equilibrium and compute training loss.

    This class takes a neural network policy function (capital savings)
    and simulates the resulting equilibrium. It computes allocations,
    prices, and Euler equation errors, which form the loss function
    for training the neural network.
    """

    def economic_loss(self,model,x,y=None):
        """
        Compute equilibrium allocations, prices, and training loss.

        This is the main method that:
        1. Takes aggregate state x (capital distribution, depreciation, TFP)
        2. Predicts next period capital using neural network policy
        3. Computes current period equilibrium (production, wages, consumption)
        4. Computes one-period-ahead forecasts for Euler equation expectations
        5. Evaluates Euler equation errors and feasibility violations
        6. Returns loss function for neural network training

        Args:
            model: Neural network model with policy function and parameters
            x (torch.Tensor): Current aggregate state, shape (batch, input_dim)
            y (torch.Tensor, optional): Pre-computed predictions (for efficiency).
                If None, model(x) is called. Defaults to None.

        The method stores all computed quantities as attributes (self.C, self.K, etc.)
        and computes self.loss as the training objective.
        """

        # Initialize and validate inputs
        self.simdevice = x.device
        self.model = model
        assert x.device == next(model.parameters()).device, \
            "Device mismatch in econ_sim!"

        # Get policy predictions (savings decisions)
        if y == None:
            self.Y = self.model(x)
        else:
            self.Y = y
        self.yLen = self.Y.shape[0]

        #-----------------------------------------------------------------------
        # CURRENT PERIOD EQUILIBRIUM
        # Compute allocations and prices given current state and policy
        #-----------------------------------------------------------------------

        # Extract aggregate state variables
        self.Z = x[...,self.model.par.aggstate]  # TFP shock

        # Capital stock and investment
        self.K = x[...,self.model.par.k]  # Current capital by cohort
        self.Ksum = self.K.sum(-1,True)  # Aggregate capital

        self.Kp = self.Y[...,self.model.par.kp]  # Next period capital (policy)
        self.Kpsum = self.Kp.sum(-1,True)  # Aggregate savings
        self.I = self.Kpsum - (1-self.model.par.δ)*self.Ksum  # Investment

        # Firm production and factor prices (Cobb-Douglas production function)
        self.Prod = self.Z*self.Ksum.pow(self.model.par.α)*self.model.par.L.pow(
            1-self.model.par.α
        )  # Y = Z * K^α * L^(1-α)
        self.R = self.model.par.α*self.Prod/self.Ksum  # Interest rate (MPK)
        self.W = (1-self.model.par.α)*self.Prod/self.model.par.L  # Wage (MPL)
        self.Inc = self.W*self.model.par.εW.flatten(-2).to(self.simdevice)  # Labor income by age

        # Consumption from budget constraint
        ϵc = 1e-5  # Minimum consumption floor (prevent numerical issues)

        self.Chat = self.Inc + self.model.par.padAssets(
            self.K*(1 + self.R - self.model.par.δ),  # Asset income
            side=0
        ) - self.model.par.padAssets(
            self.Kp,  # Savings
            side=1
        )
        self.C = self.Chat.clip(min=ϵc)  # Enforce consumption floor
        self.Csum = self.C.sum(-1,True)

        # Feasibility constraint: C + I = Production
        self.feas = (self.C.sum(-1,True) + self.I) / self.Prod - 1

        #-----------------------------------------------------------------------
        # ONE-PERIOD-AHEAD FORECAST
        # Compute expectations for Euler equations (integrate over future shocks)
        #-----------------------------------------------------------------------

        with torch.no_grad():
            # Repeat current savings across quadrature nodes
            endog = self.Kp[None].repeat(self.model.par.num_quadnodes,1,1)

            # Future TFP shocks (AR(1) process + quadrature nodes)
            Zf = (
                self.model.par.ρ_z*(self.Z/self.model.par.TFP).log() +\
                    self.model.par.σ_z*self.model.par.quad_nodes.to(
                        self.simdevice
                    )
            ).exp().permute(1,0)[...,None]*self.model.par.TFP

            # Build next period state vector [K', Z']
            self.Xf = torch.concat(
                [endog, Zf],
                -1
            ).float()

            # Predict next period savings using policy function
            self.Yf = self.model(self.Xf)

            # Extract next period state variables
            self.Zf = self.Xf[...,self.model.par.aggstate]

            # Next period capital and investment
            self.Kf = self.Xf[...,self.model.par.k]
            self.Kfsum = self.Kf.sum(-1,True)

            self.Kpf = self.Yf[...,self.model.par.kp]
            self.Kpfsum = self.Kpf.sum(-1,True)
            self.If = self.Kpfsum - (1-self.model.par.δ)*self.Kfsum

            # Next period production and factor prices
            self.Prodf = self.Zf*self.Kfsum.pow(
                self.model.par.α
            )*self.model.par.L.pow(1-self.model.par.α)
            self.Rf = self.model.par.α*self.Prodf/self.Kfsum  # Next period interest rate
            self.Wf = (1-self.model.par.α)*self.Prodf/self.model.par.L  # Next period wage
            self.Incf = self.Wf*self.model.par.εW.flatten(-2).to(self.simdevice)

            # Next period consumption
            ϵc = 1e-5  # Minimum consumption floor

            self.Cfhat = self.Incf + self.model.par.padAssets(
                self.Kf*(1 + self.Rf - self.model.par.δ),
                side=0
            ) - self.model.par.padAssets(
                self.Kpf,
                side=1
            )
            self.Cf = self.Cfhat.clip(min=ϵc)

        #-----------------------------------------------------------------------
        # LOSS FUNCTION ASSEMBLY
        # Combine Euler equation errors and feasibility violations
        #-----------------------------------------------------------------------

        # Feasibility penalty: enforce resource constraint and prevent capital variance collapse
        self.cpen = self.feas*100 + 1e-3/self.Kpsum.std()

        # Euler equation: u'(c_t) = β * E[u'(c_{t+1}) * (1 + R_{t+1} - δ)]
        # Expressed as fractional error: [u'^{-1}(β * E[...]) / c_t] - 1
        self.Euler = self.model.par.upinv(
            self.model.par.β*self.Exp(
                self.model.par.up(
                    self.Cf[...,self.model.par.isNotYoungest]  # Next period consumption
                )*(1 + self.Rf - self.model.par.δ)  # Gross return on capital
            )
        )/(
            self.C[...,self.model.par.isNotOldest]  # Current consumption
        ) - 1

        # Complementarity slackness: Euler error = 0 OR savings = 0 (borrowing constraint)
        self.hhError = self.model.par.FB(
            self.Euler,
            self.Kp
        )

        # Combined loss: log(1 + Euler errors + feasibility penalties)
        # Log transformation smooths outliers during training
        self.lossT = (1 +
                self.hhError.pow(2).mean(-1,True) + \
                self.cpen.pow(2).mean(-1,True)
        ).log()
        self.loss = self.lossT.mean()

    def Exp(self,x):
        """
        Compute expectation using Gauss-Hermite quadrature.

        Integrates x over the discrete shock distribution using quadrature
        weights. Used to evaluate expectations in Euler equations.

        Args:
            x (torch.Tensor): Values at quadrature nodes, shape (num_nodes, ...)

        Returns:
            torch.Tensor: Expected value E[x], shape (...)
        """
        return (
            x * self.model.par.quad_weights[...,None,None].to(self.simdevice)
        ).sum(0,False)

    def profiles(self,x):
        """
        Extract life-cycle profiles from panel data.

        Converts time-series panel data into cohort-based life-cycle profiles
        by tracking each cohort from young to old. Useful for plotting
        age profiles of consumption, assets, etc.

        Args:
            x (torch.Tensor): Panel data, shape (T, I)

        Returns:
            torch.Tensor: Cohort profiles, shape (cohort_count, I)
                where cohort_count = T - I + 1

        Example:
            If T=100, I=60, we can extract 41 complete life-cycle cohorts.
        """
        T, I = x.shape

        assert T >= I, "Not enough data"

        cohort_count = T - I + 1
        # Advanced indexing to extract diagonal cohort paths
        row_idx = torch.arange(I).view(1,-1) + torch.arange(
            cohort_count
        ).view(-1,1)  # shape: (cohort_count, I)

        col_idx = torch.arange(I).expand(
            cohort_count, -1
        )  # shape: (cohort_count, I)

        return x[row_idx, col_idx]  # shape: (cohort_count, I)