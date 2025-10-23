"""
Training dataset generation for neural network policy iteration.

This module defines the DATASET class which generates training data by simulating
the economy forward using the current policy function. The simulated aggregate
states serve as input features for refining the policy function in the next
training iteration.

This implements a "policy iteration" approach where:
1. Simulate economy using current policy
2. Train policy to minimize Euler errors on simulated data
3. Repeat until convergence
"""

from packages import *

class DATASET(Dataset):
    """
    Generate training data through forward simulation.

    This PyTorch Dataset class simulates the economy for T periods across n
    stochastic paths, using the current neural network policy. The resulting
    aggregate states form the training data for the next policy iteration.

    The dataset automatically updates the steady-state normalization (xbar)
    based on the ergodic distribution of the simulated data.
    """

    def __init__(self,model):
        """
        Initialize dataset by simulating the economy forward.

        Simulates T periods across n paths using current policy, then
        discards the first 'burn' periods to remove transient dynamics.

        Args:
            model: Neural network model with policy function and parameters

        Side effects:
            Updates model.par.xbar to the mean of simulated states (improves
            normalization for next training iteration)
        """

        with torch.no_grad():

            # Simulation dimensions
            T = int(model.par.T)  # Total periods (including burn-in)
            n = int(model.par.n)  # Number of simulation paths

            # Generate stochastic TFP shock paths
            zhist = model.par.SHOCKS(T,n)

            # Allocate storage for states (X) and policy (Y)
            X = torch.zeros(n,T,model.par.input)
            Y = torch.zeros(n,T,model.par.output)

            # Initial condition: start from steady state with period-0 shocks
            X[:,0] = torch.concat(
                [
                    model.par.xbar[None,:-1].expand(n,-1),  # Steady-state capital
                    zhist[:,0,None]   # Initial TFP
                ],
                -1
            )
            Y[:,0] = model(X[:,0])  # Initial policy

            # Forward simulation: iterate law of motion
            for t in (range(1,T)):
                # Build state vector: [yesterday's savings, today's shock]
                X[:,t] = torch.concat(
                    [
                        Y[:,t-1,model.par.kp],  # Capital evolves from previous savings
                        zhist[:,t,None]          # Current TFP
                    ],
                    -1
                )
                Y[:,t] = model(X[:,t])  # Compute today's savings

            # Discard burn-in periods and flatten across (paths, time)
            self.X = X[:,model.par.burn:].reshape(
                n*(model.par.T-model.par.burn),model.par.input
            )

            # Update steady-state normalization using ergodic distribution
            model.par.xbar = self.X.mean(0)

    def __len__(self):
        """
        Return number of training samples.

        Required by PyTorch Dataset interface.

        Returns:
            int: Number of samples (n * train_periods)
        """
        return len(self.X)

    def __getitem__(self,idx):
        """
        Return training sample at given index.

        Required by PyTorch Dataset interface for DataLoader batching.

        Args:
            idx (int or tensor): Sample index or indices

        Returns:
            torch.Tensor: Aggregate state vector at index idx
        """
        return self.X[idx]