"""
Economic parameters and utility functions for generic OLG model.

This module defines the PARAMS class which encapsulates all economic parameters,
simulation settings, and utility functions for a standard overlapping generations (OLG)
model with capital accumulation and aggregate productivity shocks.

The model features:
- Life-cycle consumption and savings decisions
- Capital accumulation
- Social Security and retirement (optional)
- Aggregate productivity (TFP) shocks
"""

from packages import *
from datawork import DATA

class PARAMS():
    """
    Container for all economic parameters, simulation settings, and utility functions.

    This class initializes and stores all parameters needed for the OLG model simulation,
    including demographic structure, preferences, technology, stochastic processes,
    and neural network architecture specifications.
    """

    def __init__(self):
        """
        Initialize parameters for OLG model.
        """

        # Device configuration for PyTorch computations
        self.device = torch.device('cpu:0')  # Default device for simulations
        self.traindevice = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu:0'  # Use GPU if available for training
        )

        #-----------------------------------------------------------------------
        # ECONOMIC PARAMETERS
        # All tensors indexed as [t,i] where:
        #   t = time period
        #   i = age cohort
        #-----------------------------------------------------------------------

        # Demographics: Agent lifespan
        self.I = 60  # Number of age cohorts (periods in a lifetime)

        # Preferences: Discount factor (annual β=0.95 converted to model periods)
        self.β = .95**(60/self.I)

        # Risk aversion coefficient for CRRA utility function
        self.γ = 3

        # Production technology: Cobb-Douglas production function parameters
        self.α = 0.35  # Capital share in production

        # Capital depreciation
        self.δannual = 0.1  # Annual depreciation rate (10%)
        self.δ = 1 - (1 - self.δannual)**(60/self.I)  # Per-period depreciation rate

        # Total Factor Productivity (normalized)
        self.TFP = 10**(1-self.α)

        # Initial aggregate capital stock
        self.Kbar = 10
        # Initial capital distribution (uniform across all age cohorts)
        self.kbar = self.Kbar * torch.ones(self.I - 1) / (self.I - 1)
        # Initial aggregate state vector [capital by cohort, TFP]
        self.xbar = torch.cat(
            [self.kbar, self.TFP * torch.ones(1)],
            -1
        )

        #-----------------------------------------------------------------------
        # STOCHASTIC PROCESSES
        # AR(1) process for aggregate productivity shocks
        #-----------------------------------------------------------------------

        # Persistence parameter for TFP (annual autocorrelation = 0.9)
        self.ρ_annual = 0.9
        self.ρ_z = self.ρ_annual ** (60/self.I)  # Converted to model periods

        # Volatility of TFP shocks (calibrated to match business cycle moments)
        self.σ_annual = 0.025*np.sqrt(60/self.I)
        self.σ_z = np.sqrt(
            np.log(
                (1+np.sqrt(1+4*self.σ_annual**2))/2
            )*(1-self.ρ_z**2)
        )

        # Gauss-Hermite quadrature for integrating over future shock realizations
        self.num_quadnodes = 15  # Number of quadrature nodes
        quad_nodes, quad_weights = hermgauss(self.num_quadnodes)
        self.quad_nodes = torch.tensor(quad_nodes, dtype=torch.float32)
        self.quad_weights = torch.tensor(
            quad_weights, dtype=torch.float32
        ) / np.sqrt(np.pi)  # Normalize weights for standard normal

        #-----------------------------------------------------------------------
        # INDICATOR FLAGS
        # Boolean tensors for age boundaries
        #-----------------------------------------------------------------------

        # Age cohort indicators (used for boundary conditions in simulation)
        self.isNotYoungest = torch.tensor(
            [i != 0 for i in range(self.I)], dtype=torch.bool
        )
        self.isNotOldest = torch.tensor(
            [i != self.I - 1 for i in range(self.I)], dtype=torch.bool
        )

        #-----------------------------------------------------------------------
        # DATA PROCESSING
        # Generate theoretical income profiles
        #-----------------------------------------------------------------------

        # Initialize data processor
        data = DATA(self.I)

        # Generate hump-shaped income profile
        εW = data.generate_income_profile()  # Shape: (1, I)

        # Normalize labor supply to Lnorm=1
        self.Lnorm = 1
        self.scale = εW.sum() / self.Lnorm  # Scaling factor for all quantities
        self.εW = εW[None] / self.scale  # Scaled earnings efficiency units, shape: (1, 1, I)
        self.L = self.εW.sum()  # Total labor supply

        #-----------------------------------------------------------------------
        # SIMULATION SETTINGS
        # Time periods and training configuration
        #-----------------------------------------------------------------------

        self.burn = 10  # Burn-in periods to discard (allow convergence)
        self.T = 100 + self.burn  # Total simulation periods
        self.n = 100  # Number of simulation paths for stochastic shocks

        # Training data configuration
        self.train = int(self.T - self.burn)  # Number of periods used for training
        self.time = slice(self.burn,self.T,1)  # Slice indexing training periods

        #-----------------------------------------------------------------------
        # NEURAL NETWORK ARCHITECTURE
        # Input/output dimensions and state vector slicing
        #-----------------------------------------------------------------------

        # Input dimension: capital by cohort (I-1) + TFP (1)
        self.input = (self.I - 1) + 1

        # Slices for accessing components of input state vector
        self.k = slice(0, self.I - 1, 1)  # Capital distribution across cohorts
        self.aggstate = slice(-1, self.input, 1)  # Aggregate TFP shock

        # Output dimension: next period capital by cohort (policy function)
        self.output = self.I - 1

        # Slices for accessing components of output vector
        self.kp = slice(0, self.I - 1, 1)  # Next period capital (savings policy)

        #-----------------------------------------------------------------------
        # FILE SYSTEM PATHS
        # Platform-independent directory paths for model saves and results
        #-----------------------------------------------------------------------

        if platform.system() == 'Windows':
            self.modelSavePath = '.\\modelsave\\'
            self.resultsPath = '.\\results\\'
        else:
            self.modelSavePath = './modelsave/'
            self.resultsPath = './results/'

        # Create directories if they don't exist
        if not os.path.exists(self.modelSavePath):
            os.makedirs(self.modelSavePath)
        if not os.path.exists(self.resultsPath):
            os.makedirs(self.resultsPath)

    #---------------------------------------------------------------------------
    # Economic helper functions
    #---------------------------------------------------------------------------

    def SHOCKS(self, T, n=1):
        """
        Generate realizations of AR(1) productivity shocks.

        Simulates TFP following: log(z_t) = ρ_z * log(z_{t-1}) + σ_z * ε_t
        where ε_t ~ N(0,1). Initial condition drawn from stationary distribution.

        Args:
            T (int): Number of time periods to simulate
            n (int, optional): Number of simulation paths. Defaults to 1.

        Returns:
            torch.Tensor: TFP levels, shape (n, T)
        """
        log_z = torch.zeros((n,T))
        # Initialize from stationary distribution
        log_z[:,0] = torch.randn(n) * (self.σ_z / np.sqrt(1 - self.ρ_z**2))
        ε = torch.randn(n,T-1)
        for t in range(1, T):
            log_z[:,t] = self.ρ_z * log_z[:,t-1] + self.σ_z * ε[:,t-1]

        zhist = torch.exp(log_z)*self.TFP  # Convert to levels
        return zhist

    def u(self,x):
        """
        CRRA utility function.

        Implements constant relative risk aversion utility with coefficient γ.
        Special case: γ=1 gives log utility.

        Args:
            x (torch.Tensor): Consumption level

        Returns:
            torch.Tensor: Utility value
        """
        if self.γ == 1:
            return x.log()
        else:
            return (x.pow(1-self.γ)-1)/(1-self.γ)

    def up(self,x):
        """
        Marginal utility (first derivative of utility function).

        For CRRA utility: u'(c) = c^(-γ)

        Args:
            x (torch.Tensor): Consumption level

        Returns:
            torch.Tensor: Marginal utility
        """
        return x.pow(-self.γ)

    def upinv(self,x):
        """
        Inverse of marginal utility function.

        Solves c = (u')^(-1)(m) where m is marginal utility.
        Used in Euler equation solutions.

        Args:
            x (torch.Tensor): Marginal utility value

        Returns:
            torch.Tensor: Consumption level
        """
        return x.pow(-1/self.γ)

    def annualize(self,r):
        """
        Convert period interest rate to annual equivalent.

        Args:
            r (torch.Tensor or float): Per-period interest rate

        Returns:
            torch.Tensor or float: Annualized interest rate
        """
        return (1+r)**(self.I/60)-1

    def periodize(self,r):
        """
        Convert annual interest rate to period equivalent.

        Args:
            r (torch.Tensor or float): Annual interest rate

        Returns:
            torch.Tensor or float: Per-period interest rate
        """
        return (1+r)**(60/self.I)-1

    def padAssets(self, x, side):
        """
        Pad asset vector to include youngest/oldest cohort boundaries.

        Inserts zeros at age boundaries (youngest or oldest).
        Used to handle boundary conditions in life-cycle problem.

        Args:
            x (torch.Tensor): Asset holdings by cohort, shape (..., I-1)
            side (int): Which side to pad (0=left/youngest, 1=right/oldest)

        Returns:
            torch.Tensor: Padded asset vector, shape (..., I)
        """
        if side == 0:
            return F.pad(x, (1, 0))  # Pad left (youngest cohort has 0 assets)
        elif side == 1:
            return F.pad(x, (0, 1))  # Pad right (oldest cohort dies with 0 assets)

    def FB(self,a,b):
        """
        Fisher-Burmeister smoothing function for complementarity constraints.

        Smooth approximation of the complementarity condition: a ≥ 0, b ≥ 0, a*b = 0.
        Used as an alternative to Karush-Kuhn-Tucker (KKT) conditions.

        Args:
            a (torch.Tensor): First argument (e.g., constraint slack)
            b (torch.Tensor): Second argument (e.g., multiplier)

        Returns:
            torch.Tensor: Smoothed complementarity residual (should be ≈ 0)
        """
        l = .8
        val = a.pow(2) + b.pow(2)
        return l*(
            a + b - val/(val+1e-6).sqrt()
        ) + (1-l)*a.clip(min=0)*b.clip(min=0)