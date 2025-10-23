"""
Income profile generation for generic OLG model.

This module generates theoretical life-cycle income profiles for the OLG model.
For a standard OLG model with capital accumulation and TFP shocks, we use
a simple hump-shaped income profile.

The income profile is generated using a quadratic function that:
- Starts low at labor market entry
- Peaks at middle age
- Declines toward end of life

IMPORTANT: This is a placeholder for your model's calibration/estimation.
Replace this file with your own calibration procedure based on:
- Empirical data (e.g., PSID, CPS, administrative data)
- Estimated income processes from your research
- Any other model-specific parameters that need to be calibrated
"""

from packages import *

class DATA():
    """
    Generate theoretical life-cycle income profiles.

    This class creates a simple hump-shaped income profile for a generic
    OLG model without heterogeneous types.
    """

    def __init__(self, I=60):
        """
        Initialize income profile generator.

        Args:
            I (int, optional): Number of age cohorts. Defaults to 60.
        """
        self.I = I

    def generate_income_profile(self):
        """
        Generate hump-shaped income profile.

        Creates a theoretical income profile that:
        1. Follows a cubic (third-order) polynomial over the life cycle
        2. Starts above zero, peaks around age 30, decays but stays positive
        3. Provides realistic life-cycle earnings dynamics

        Returns:
            torch.Tensor: Income profile of shape (1, I) normalized to mean 1
        """
        # Generate hump-shaped profile using cubic polynomial
        # Designed to: start at ~0.6, peak at age 30 (~1.3), end at ~0.4
        peak_age = 30

        # Cubic polynomial coefficients (derived to match boundary conditions)
        # Form: income[i] = s_peak + c*(i - peak_age)^2 + d*(i - peak_age)^3
        s_peak = 1.3  # Height at peak
        c = -0.000926  # Quadratic coefficient (creates hump)
        d = -0.00000496  # Cubic coefficient (creates asymmetry)

        income = np.zeros(self.I)
        for i in range(self.I):
            # Cubic with peak at age 30
            age_deviation = i - peak_age
            income[i] = s_peak + c * age_deviation**2 + d * age_deviation**3

        # Normalize to mean 1
        income = income / income.mean()

        # Save profile plot
        resultsPath = './results/'
        if not os.path.exists(resultsPath):
            os.makedirs(resultsPath)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(income, 'b-', linewidth=2)
        ax.set_xlabel('Age')
        ax.set_ylabel('Labor Income Efficiency Units')
        ax.set_title('Life-Cycle Labor Income Profile')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(resultsPath + 'income_profile.png', dpi=300)
        plt.close()

        return torch.tensor(income, dtype=torch.float32).reshape(1, -1)