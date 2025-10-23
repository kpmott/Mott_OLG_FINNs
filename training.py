"""
Neural network training for OLG equilibrium policy functions.

This module implements the training loop for learning the aggregate savings
policy using a combination of:
- Policy iteration: Re-simulate economy with current policy to generate data
- Gradient descent: Minimize Euler equation errors on simulated data
- Progressive training: Multiple regimes with varying learning rates and batch sizes

The training alternates between:
1. Simulating the economy forward (dataset generation)
2. Training on economic loss (Euler errors + feasibility)
"""

from packages import *
from dataset import DATASET
from econ_sim import ECONOMY_SIM
from plot_stationary import PLOTS_STAT

class TRAIN():
    """
    Training coordinator for neural network policy iteration.

    Manages the training process including data generation, optimization,
    checkpointing, and visualization. Supports multi-regime training with
    different hyperparameters for each regime.
    """

    def __init__(self,model):
        """
        Initialize trainer with model and economic simulator.

        Args:
            model: Neural network model with policy function
        """
        self.model = model
        self.sim = ECONOMY_SIM()  # Economic equilibrium simulator

    def plot_losses(self, losses):
        """
        Plot training losses on log scale and save to file.

        Args:
            losses (list or array): Loss values across episodes
        """
        losses_array = np.array(losses)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(losses_array, linewidth=2)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Loss (log scale)', fontsize=12)
        ax.set_title('Training Loss Curve', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.model.par.resultsPath + 'training_losses.png', dpi=300)
        plt.close()

        # Save losses to CSV
        losses_array = np.array(losses)
        np.savetxt(
            self.model.par.resultsPath + 'training_losses.csv',
            losses_array,
            delimiter=',',
            header='loss',
            comments=''
        )

    def train_loop(self,episodes,batchsize,lr,losses=[]):
        """
        Single training regime with fixed hyperparameters.

        Implements policy iteration: for each episode, re-simulate the economy
        with the current policy, then train on the simulated data.

        Args:
            episodes (int): Number of episodes to train
            batchsize (int): Mini-batch size for gradient descent
            lr (float): Learning rate for Adam optimizer
            losses (list, optional): Running list of losses. Defaults to [].

        Returns:
            list: Updated loss history (one entry per episode)

        Training procedure per episode:
        1. Re-simulate economy forward using current policy (CPU)
        2. Move model and data to GPU
        3. Train on mini-batches via gradient descent
        """

        # Convert episodes to range if passed as int
        if isinstance(episodes, int):
            episodes = range(episodes)

        # Initialize optimizer
        optimizer = Adam(
            self.model.parameters(),
            lr=lr
        )

        for episode in (lossvalPrint := tqdm(episodes)):
            # Re-simulate economy for each episode (policy iteration step)
            self.model = self.model.to(self.model.par.device)
            with torch.no_grad():
                data = DATASET(self.model)

            # Transfer to GPU for training
            self.model = self.model.to(self.model.par.traindevice)
            X = data.X.to(self.model.par.traindevice)
            # Shuffle data
            perm = torch.randperm(X.shape[0],device=self.model.par.traindevice)
            X = X[perm]

            # Mini-batch gradient descent
            batchloss = []
            for batch in range(0,X.shape[0],batchsize):

                x_batch = X[
                    batch:batch+batchsize
                ].clone().detach().requires_grad_(True)

                # Zero gradients
                optimizer.zero_grad(set_to_none=True)

                # Forward pass and loss computation
                y_batch = self.model(x_batch)
                self.sim.economic_loss(self.model,x_batch,y_batch)

                # Backward pass with gradient clipping
                lossval = self.sim.loss
                lossval.backward()
                optimizer.step()

                # Track batch loss
                batchloss.append(lossval.item())

            # Record episode loss and update progress bar
            episodeloss = np.mean(batchloss)
            losses.append(episodeloss)
            lossvalPrint.set_description(
                'Loss %.2e'%episodeloss
            )

            # Generate diagnostic plots every 100 episodes
            if episode % 100 == 0:
                plotter = PLOTS_STAT(self.model, episode=episode, losses=losses)
                plotter.PrintPlots()

        return losses

    def train(self,episodes,batchsize,lr):
        """
        Train the model via policy iteration.

        Args:
            episodes (int): Number of policy iteration episodes
            batchsize (int): Batch size for gradient descent
            lr (float): Learning rate for Adam optimizer

        Returns:
            list: Loss history across all episodes

        Side effects:
            - Saves model checkpoint after training completes
        """

        # Run training loop
        losses = self.train_loop(
                episodes=episodes,
                batchsize=batchsize,
                lr=lr,
                losses=[]
        )

        # Save checkpoint after training
        torch.save(
            {
                'model_state': self.model.state_dict(),
                'xbar': self.model.par.xbar
            },
            self.model.par.modelSavePath+'trained_model_params.pt'
        )

        # Plot final losses
        self.plot_losses(losses)

        return losses