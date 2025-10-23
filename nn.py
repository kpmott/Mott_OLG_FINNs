"""
Neural network architecture for approximating the OLG policy function.

This module defines the MODEL class, a feedforward neural network that learns
the aggregate savings policy function k'(k, z). The network maps from
the aggregate state (capital distribution + TFP shock) to next period
capital holdings for all age cohorts.

Architecture:
- Input: Aggregate state [capital by cohort, TFP]
- Hidden layers: 2 layers with 150 neurons each, Tanh activation
- Output: Next period capital by cohort, Softplus activation (ensures k' ≥ 0)
"""

from packages import *
from parameters import PARAMS

class MODEL(nn.Module):
    """
    Neural network model for OLG aggregate policy function.

    This class inherits from PyTorch's nn.Module and implements a feedforward
    network that approximates the equilibrium savings policy. The network is
    trained to satisfy Euler equations and market clearing conditions.
    """

    def __init__(self):
        """
        Initialize neural network and parameter structure.

        Constructs a feedforward network with:
        - 2 hidden layers (150 neurons each) with Tanh activation
        - Output layer with Softplus activation (guarantees positive savings)
        - Input normalization using steady-state values
        """
        super().__init__()
        self.par = PARAMS()  # Economic parameters

        # Network architecture: input -> 150 -> 150 -> output
        sizes = [self.par.input,150,150,self.par.output]

        # Build sequential network layers
        layers = []
        # Hidden layers with Tanh activation
        for layer in range(len(sizes)-2):
            layers.append(
                nn.Linear(in_features=sizes[layer],out_features=sizes[layer+1])
            )
            layers.append(nn.Tanh())

        # Output layer with Softplus activation (ensures k' ≥ 0)
        layers.append(nn.Linear(in_features=sizes[-2],out_features=sizes[-1]))
        layers.append(nn.Softplus())

        self.model = nn.Sequential(*layers).to(self.par.device)

    def forward(self, x):
        """
        Forward pass: compute savings policy k' = f(k, z).

        Normalizes input by steady-state values before passing through network.
        This improves training stability and convergence.

        Args:
            x (torch.Tensor): Aggregate state [k, z], shape (batch, input_dim)

        Returns:
            torch.Tensor: Next period capital k', shape (batch, output_dim)
        """
        # Normalize inputs by steady-state values (+1 to avoid division issues)
        x_norm = x.clone()/(self.par.xbar.to(x.device)+1)
        return self.model(x_norm)