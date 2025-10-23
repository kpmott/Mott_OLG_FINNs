"""
Main training dashboard for OLG neural network policy iteration.

This script orchestrates the complete training pipeline for the generic OLG model
with capital accumulation and TFP shocks.

Training Strategy:
- Policy iteration: Simulate economy → Train on Euler errors → Repeat
- Continue for specified number of episodes with fixed hyperparameters

Usage:
    python dashboard.py

Outputs:
- Model checkpoints: ./modelsave/trained_model_params.pt
- Diagnostic plots: ./plots/ (generated every 25 episodes)
- Loss trajectories tracked across all episodes
"""

from packages import *
from nn import MODEL
from training import TRAIN

#-------------------------------------------------------------------------------
# TRAINING CONFIGURATION
#-------------------------------------------------------------------------------

# Initialize model
model = MODEL()

# Print model architecture
summary(model.to('cuda:0'), input_size=(model.par.input,))

#-------------------------------------------------------------------------------
# HYPERPARAMETERS
#-------------------------------------------------------------------------------

# Number of policy iteration episodes
episodes = 20000

# Batch size (number of states per gradient step)
batchsize = 200

# Learning rate
lr = 1e-6

# Initialize trainer
train = TRAIN(model)

#-------------------------------------------------------------------------------
# RUN TRAINING
#-------------------------------------------------------------------------------

train.train(
    episodes=episodes,
    batchsize=batchsize,
    lr=lr
)

print("Training completed!")
print(f"Checkpoint saved to: {model.par.modelSavePath}trained_model_params.pt")
