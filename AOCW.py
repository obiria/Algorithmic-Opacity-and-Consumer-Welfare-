"""
Publication-Quality Figure Generation for:
'Algorithmic Opacity and Consumer Welfare: A Discrete Choice Analysis of AI-Mediated Markets'

This script generates two main figures:
1. Figure 1: Equilibrium price, profit, and welfare under algorithmic confusion
2. Figure 2: Welfare-cost frontier and confusion reduction under transparency regulation

Requirements: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import matplotlib as mpl

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


class AIMarketModel:
    """
    Discrete choice model of algorithmic confusion in AI-mediated markets.
    
    Parameters:
    -----------
    a : float
        Demand intercept parameter
    b : float
        Demand slope parameter (price sensitivity)
    c : float
        Marginal cost
    eta_0 : float
        Baseline price elasticity (should be < -1)
    alpha : float
        Confusion sensitivity parameter (0 < alpha < 1)
    sigma_0 : float
        Initial confusion level
    beta : float
        Transparency policy effectiveness
    lambda_cost : float
        Enforcement cost parameter
    """
    
    def __init__(self, a=100, b=2, c=20, eta_0=-3, alpha=0.1, 
                 sigma_0=2, beta=2, lambda_cost=50):
        self.a = a
        self.b = b
        self.c = c
        self.eta_0 = eta_0
        self.alpha = alpha
        self.sigma_0 = sigma_0
        self.beta = beta
        self.lambda_cost = lambda_cost
    
    def elasticity(self, sigma):
        """Effective price elasticity as a function of confusion."""
        return self.eta_0 * (1 - self.alpha * sigma**2)
    
    def optimal_price(self, sigma):
        """Profit-maximizing price given confusion level."""
        eta = self.elasticity(sigma)
        return self.c / (1 - 1/eta)
    
    def demand(self, p, sigma):
        """Demand function (simplified linear form)."""
        return max(self.a - self.b * p, 0)
    
    def profit(self, sigma):
        """Equilibrium firm profit."""
        p_star = self.optimal_price(sigma)
        q_star = self.demand(p_star, sigma)
        return (p_star - self.c) * q_star
    
    def consumer_surplus(self, sigma):
        """Consumer surplus."""
        p_star = self.optimal_price(sigma)
        q_star = self.demand(p_star, sigma)
        return 0.5 * q_star**2 / self.b
    
    def welfare(self, sigma):
        """Total social welfare."""
        return self.consumer_surplus(sigma) + self.profit(sigma)
    
    def confusion_with_transparency(self, tau):
        """Effective confusion under transparency policy."""
        return self.sigma_0 * np.exp(-self.beta * tau)
    
    def enforcement_cost(self, tau):
        """Quadratic enforcement cost."""
        return 0.5 * self.lambda_cost * tau**2
    
    def net_welfare(self, tau):
        """Net social welfare accounting for enforcement costs."""
        sigma_tau = self.confusion_with_transparency(tau)
        return self.welfare(sigma_tau) - self.enforcement_cost(tau)
    
    def optimal_transparency(self):
        """Find optimal transparency level."""
        result = minimize_scalar(
            lambda tau: -self.net_welfare(tau),
            bounds=(0, 1),
            method='bounded'
        )
        return result.x


def generate_figure1(model, save_path='fig1_equilibrium.pdf'):
    """
    Generate Figure 1: Equilibrium outcomes under algorithmic confusion.
    
    Shows how price, profit, and welfare evolve with confusion intensity.
    """
    # Generate confusion intensity range
    sigma_range = np.linspace(0, 3, 200)
    
    # Calculate equilibrium outcomes
    prices = np.array([model.optimal_price(s) for s in sigma_range])
    profits = np.array([model.profit(s) for s in sigma_range])
    welfare = np.array([model.welfare(s) for s in sigma_range])
    
    # Normalize for visualization (optional, for better comparison)
    prices_norm = (prices - prices.min()) / (prices.max() - prices.min())
    profits_norm = (profits - profits.min()) / (profits.max() - profits.min())
    welfare_norm = (welfare - welfare.min()) / (welfare.max() - welfare.min())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot normalized values
    ax.plot(sigma_range, prices_norm, 'b-', label='Equilibrium Price $p^*(\sigma)$', linewidth=2.5)
    ax.plot(sigma_range, profits_norm, 'g--', label='Firm Profit $\Pi^*(\sigma)$', linewidth=2.5)
    ax.plot(sigma_range, 1 - welfare_norm, 'r-.', label='Welfare Loss (normalized)', linewidth=2.5)
    
    # Formatting
    ax.set_xlabel('Algorithmic Confusion Intensity ($\sigma$)', fontsize=12)
    ax.set_ylabel('Normalized Values', fontsize=12)
    ax.set_title('Equilibrium Outcomes Under Algorithmic Confusion', fontsize=13, pad=15)
    ax.legend(loc='best', frameon=True, shadow=True, fancybox=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 1.05])
    
    # Add annotations
    ax.annotate('Price increases\nwith confusion',
                xy=(2.0, prices_norm[int(len(sigma_range)*2/3)]), 
                xytext=(1.2, 0.85),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='blue', alpha=0.7),
                fontsize=9, color='blue')
    
    ax.annotate('Welfare declines\nconvexly',
                xy=(2.0, 1 - welfare_norm[int(len(sigma_range)*2/3)]), 
                xytext=(2.2, 0.4),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='red', alpha=0.7),
                fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Figure 1 saved as '{save_path}'")
    plt.show()


def generate_figure2(model, save_path='fig2_transparency.pdf'):
    """
    Generate Figure 2: Transparency regulation and welfare-cost frontier.
    
    Panel (a): Net social welfare vs transparency intensity
    Panel (b): Effective confusion vs transparency intensity
    """
    # Generate transparency intensity range
    tau_range = np.linspace(0, 1, 200)
    
    # Calculate outcomes
    net_welfare = np.array([model.net_welfare(tau) for tau in tau_range])
    confusion = np.array([model.confusion_with_transparency(tau) for tau in tau_range])
    welfare_gross = np.array([model.welfare(model.confusion_with_transparency(tau)) 
                              for tau in tau_range])
    enforcement_costs = np.array([model.enforcement_cost(tau) for tau in tau_range])
    
    # Find optimal transparency
    tau_star = model.optimal_transparency()
    net_welfare_star = model.net_welfare(tau_star)
    confusion_star = model.confusion_with_transparency(tau_star)
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel (a): Net welfare vs transparency
    ax1.plot(tau_range, net_welfare, 'b-', linewidth=2.5, label='Net Welfare $\Omega(\tau)$')
    ax1.plot(tau_range, welfare_gross, 'g--', linewidth=2, alpha=0.7, label='Gross Welfare $W(\sigma(\tau))$')
    ax1.plot(tau_range, enforcement_costs, 'r-.', linewidth=2, alpha=0.7, label='Enforcement Cost $C(\tau)$')
    
    # Mark optimum
    ax1.plot(tau_star, net_welfare_star, 'ko', markersize=10, 
             label=f'Optimum $\\tau^* = {tau_star:.3f}$', zorder=5)
    ax1.axvline(tau_star, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(net_welfare_star, color='gray', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Transparency Intensity ($\\tau$)', fontsize=12)
    ax1.set_ylabel('Welfare (monetary units)', fontsize=12)
    ax1.set_title('(a) Welfare-Cost Frontier', fontsize=13, pad=12)
    ax1.legend(loc='best', frameon=True, shadow=True, fancybox=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, 1])
    
    # Annotations
    ax1.annotate('Second-best optimum',
                xy=(tau_star, net_welfare_star),
                xytext=(tau_star + 0.2, net_welfare_star - 50),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black', alpha=0.7),
                fontsize=9)
    
    # Panel (b): Confusion vs transparency
    ax2.plot(tau_range, confusion, 'purple', linewidth=2.5, 
             label='Effective Confusion $\sigma(\tau)$')
    ax2.plot(tau_star, confusion_star, 'ko', markersize=10, 
             label=f'$\sigma(\\tau^*) = {confusion_star:.3f}$', zorder=5)
    ax2.axvline(tau_star, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(confusion_star, color='gray', linestyle=':', alpha=0.5)
    
    # Add exponential decay reference
    ax2.plot(tau_range, model.sigma_0 * np.ones_like(tau_range), 
             'k--', linewidth=1, alpha=0.5, label='$\sigma_0$ (no regulation)')
    
    ax2.set_xlabel('Transparency Intensity ($\\tau$)', fontsize=12)
    ax2.set_ylabel('Confusion Level ($\sigma$)', fontsize=12)
    ax2.set_title('(b) Confusion Reduction', fontsize=13, pad=12)
    ax2.legend(loc='best', frameon=True, shadow=True, fancybox=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, model.sigma_0 * 1.1])
    
    # Annotations
    ax2.annotate('Exponential\nconfusion decay\n$\sigma(\tau) = \sigma_0 e^{-\\beta\\tau}$',
                xy=(0.3, model.confusion_with_transparency(0.3)),
                xytext=(0.5, model.sigma_0 * 0.7),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='purple', alpha=0.7),
                fontsize=9, color='purple')
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Figure 2 saved as '{save_path}'")
    plt.show()


def print_summary_statistics(model):
    """Print key model statistics and optimal values."""
    tau_star = model.optimal_transparency()
    
    print("\n" + "="*70)
    print("MODEL SUMMARY STATISTICS")
    print("="*70)
    print(f"\nBaseline Parameters:")
    print(f"  Demand intercept (a):           {model.a}")
    print(f"  Demand slope (b):               {model.b}")
    print(f"  Marginal cost (c):              {model.c}")
    print(f"  Baseline elasticity (η₀):       {model.eta_0}")
    print(f"  Confusion sensitivity (α):      {model.alpha}")
    print(f"  Initial confusion (σ₀):         {model.sigma_0}")
    
    print(f"\nEquilibrium Outcomes at σ=0 (No Confusion):")
    print(f"  Optimal price:                  {model.optimal_price(0):.2f}")
    print(f"  Firm profit:                    {model.profit(0):.2f}")
    print(f"  Consumer surplus:               {model.consumer_surplus(0):.2f}")
    print(f"  Total welfare:                  {model.welfare(0):.2f}")
    
    print(f"\nEquilibrium Outcomes at σ={model.sigma_0} (High Confusion):")
    print(f"  Optimal price:                  {model.optimal_price(model.sigma_0):.2f}")
    print(f"  Firm profit:                    {model.profit(model.sigma_0):.2f}")
    print(f"  Consumer surplus:               {model.consumer_surplus(model.sigma_0):.2f}")
    print(f"  Total welfare:                  {model.welfare(model.sigma_0):.2f}")
    
    print(f"\nWelfare Loss from Confusion:")
    welfare_loss = model.welfare(0) - model.welfare(model.sigma_0)
    welfare_loss_pct = 100 * welfare_loss / model.welfare(0)
    print(f"  Absolute loss:                  {welfare_loss:.2f}")
    print(f"  Percentage loss:                {welfare_loss_pct:.2f}%")
    
    print(f"\nOptimal Transparency Policy:")
    print(f"  Optimal τ*:                     {tau_star:.4f}")
    print(f"  Effective confusion σ(τ*):      {model.confusion_with_transparency(tau_star):.4f}")
    print(f"  Net welfare Ω(τ*):              {model.net_welfare(tau_star):.2f}")
    print(f"  Enforcement cost C(τ*):         {model.enforcement_cost(tau_star):.2f}")
    
    print(f"\nComparative Analysis:")
    net_welfare_no_reg = model.net_welfare(0)
    net_welfare_optimal = model.net_welfare(tau_star)
    net_welfare_full = model.net_welfare(1)
    print(f"  Net welfare (no regulation):    {net_welfare_no_reg:.2f}")
    print(f"  Net welfare (optimal τ*):       {net_welfare_optimal:.2f}")
    print(f"  Net welfare (full disclosure):  {net_welfare_full:.2f}")
    print(f"  Gain from optimal policy:       {net_welfare_optimal - net_welfare_no_reg:.2f}")
    print("="*70 + "\n")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("ALGORITHMIC OPACITY AND CONSUMER WELFARE")
    print("Publication-Quality Figure Generation")
    print("="*70)
    
    # Initialize model with baseline parameters
    model = AIMarketModel(
        a=100,           # Demand intercept
        b=2,             # Demand slope
        c=20,            # Marginal cost
        eta_0=-3,        # Baseline elasticity
        alpha=0.1,       # Confusion sensitivity
        sigma_0=2,       # Initial confusion
        beta=2,          # Transparency effectiveness
        lambda_cost=50   # Enforcement cost parameter
    )
    
    # Print summary statistics
    print_summary_statistics(model)
    
    # Generate figures
    print("Generating figures...")
    print("-" * 70)
    
    generate_figure1(model, save_path='fig1_equilibrium.pdf')
    generate_figure2(model, save_path='fig2_transparency.pdf')
    
    print("\n" + "="*70)
    print("All figures generated successfully!")
    print("Files saved: fig1_equilibrium.pdf, fig2_transparency.pdf")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()