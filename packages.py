"""
Package imports for OLG Neural Network Student Debt project.

This module centralizes all external package imports used throughout the project,
including PyTorch for neural networks, NumPy/SciPy for numerical computation,
matplotlib for visualization, and pandas for data processing.

The module also configures global settings for PyTorch (default dtype),
multiprocessing (spawn method), and matplotlib (warning suppression).
"""

# ==============================================================================
# Operating system and utility imports
# ==============================================================================
import os                    # File and directory operations
import json                  # JSON serialization/deserialization
import sys                   # System-specific parameters and functions
import csv                   # CSV file reading and writing
from cycler import cycler    # Color cycling for matplotlib plots

# ==============================================================================
# PyTorch: Deep learning framework
# ==============================================================================
import torch
# Set default precision to float32 for performance/memory balance
torch.set_default_dtype(torch.float32)
# Optional CUDA device setting (commented out for CPU/GPU flexibility)
# if torch.cuda.is_available():
    # torch.set_default_device('cuda:0')

# Neural network building blocks
import torch.nn as nn

# Optimization algorithms for training neural networks
from torch.optim import Adam, AdamW, SGD, LBFGS, RMSprop

# Data loading utilities for mini-batch training
from torch.utils.data import DataLoader, Dataset

# Functional interface for neural network operations
import torch.nn.functional as F

# Model architecture summary tool
from torchsummary import summary

# Optional debugging and logging tools (commented out)
# from torch.utils.tensorboard import SummaryWriter
# torch.autograd.set_detect_anomaly(True)

# ==============================================================================
# NumPy: Numerical computing and array operations
# ==============================================================================
import numpy as np
# Hermite-Gauss quadrature for numerical integration (used in economic simulations)
from numpy.polynomial.hermite import hermgauss

# ==============================================================================
# SciPy: Scientific computing and optimization
# ==============================================================================
# Numerical solvers for nonlinear equations and optimization problems
from scipy.optimize import fsolve, minimize, root_scalar, root, bisect

# Signal processing filter for data smoothing
from scipy.signal import savgol_filter

# Optional statistical distributions (commented out)
# from scipy.stats import norm

# ==============================================================================
# Progress monitoring
# ==============================================================================
# Console progress bars for long-running computations
from tqdm import tqdm

# ==============================================================================
# Matplotlib: Plotting and visualization
# ==============================================================================
import matplotlib.pyplot as plt          # Main plotting interface
import matplotlib.patches as mpatches    # Patch objects (rectangles, circles, etc.)
import matplotlib.lines                  # Line objects for legends and annotations
from matplotlib.ticker import StrMethodFormatter  # Axis tick formatting
import matplotlib.ticker as mtick        # Additional tick utilities
import matplotlib.gridspec as gridspec   # Subplot layout management

# Suppress warning about too many open figures
plt.rcParams['figure.max_open_warning'] = 0

# ==============================================================================
# Command-line argument parsing
# ==============================================================================
import argparse  # Parse command-line options and arguments

# ==============================================================================
# Performance and timing utilities
# ==============================================================================
import time  # Time measurement for profiling and benchmarking

# ==============================================================================
# Pandas: Data manipulation and analysis
# ==============================================================================
import pandas as pd

# Suppress pandas warnings to keep output clean
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# ==============================================================================
# System information
# ==============================================================================
import platform  # Access to underlying platform's identifying data

# ==============================================================================
# Date and time handling
# ==============================================================================
from datetime import datetime  # Date and time manipulation