# Algorithmic Opacity and Consumer Welfare

Publication-quality figure generation for discrete choice analysis of AI-mediated markets.

## Overview

This repository contains code for generating figures that analyze the economic impacts of algorithmic opacity in AI-mediated markets. The model examines how consumer confusion induced by opaque algorithms affects market equilibrium, pricing strategies, firm profits, and social welfare.

## Research Paper

**Title:** Algorithmic Opacity and Consumer Welfare: A Discrete Choice Analysis of AI-Mediated Markets

**Abstract:** This research develops a discrete choice model to analyze how algorithmic confusion affects market outcomes and evaluates optimal transparency policies for AI-mediated platforms.

## Features

- **Discrete Choice Market Model:** Analyzes equilibrium outcomes under varying levels of algorithmic confusion
- **Welfare Analysis:** Computes consumer surplus, firm profits, and total social welfare
- **Policy Evaluation:** Determines optimal transparency regulation balancing welfare gains against enforcement costs
- **Publication-Quality Figures:** Generates PDF figures suitable for academic publication

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- SciPy

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd algorithmic-opacity

# Install dependencies
pip install numpy matplotlib scipy
```

Alternatively, using conda:

```bash
conda install numpy matplotlib scipy
```

## Usage

### Basic Usage

Run the main script to generate all figures with default parameters:

```bash
python figure_generation.py
```

This will:
1. Initialize the market model with baseline parameters
2. Print summary statistics to the console
3. Generate two PDF figures: `fig1_equilibrium.pdf` and `fig2_transparency.pdf`

### Custom Parameters

To customize model parameters, modify the `AIMarketModel` initialization in the `main()` function:

```python
model = AIMarketModel(
    a=100,           # Demand intercept
    b=2,             # Demand slope
    c=20,            # Marginal cost
    eta_0=-3,        # Baseline price elasticity
    alpha=0.1,       # Confusion sensitivity
    sigma_0=2,       # Initial confusion level
    beta=2,          # Transparency policy effectiveness
    lambda_cost=50   # Enforcement cost parameter
)
```

### Programmatic Usage

You can also import and use the model in your own scripts:

```python
from figure_generation import AIMarketModel, generate_figure1, generate_figure2

# Initialize model
model = AIMarketModel()

# Get specific outcomes
price = model.optimal_price(sigma=1.5)
profit = model.profit(sigma=1.5)
welfare = model.welfare(sigma=1.5)

# Find optimal policy
optimal_tau = model.optimal_transparency()

# Generate individual figures
generate_figure1(model, save_path='custom_fig1.pdf')
generate_figure2(model, save_path='custom_fig2.pdf')
```

## Model Description

### Core Framework

The model analyzes a monopolistic market where an AI platform mediates consumer-firm interactions. Key features:

- **Algorithmic Confusion (σ):** Consumers' difficulty in understanding AI-driven recommendations and pricing
- **Price Elasticity:** Effective elasticity decreases with confusion: η(σ) = η₀(1 - ασ²)
- **Firm Optimization:** Platform chooses profit-maximizing prices given consumer confusion
- **Transparency Policy (τ):** Regulatory intervention that reduces confusion exponentially: σ(τ) = σ₀e^(-βτ)

### Key Equations

**Optimal Price:**
```
p*(σ) = c / (1 - 1/η(σ))
```

**Consumer Surplus:**
```
CS(σ) = 0.5 × q²/b
```

**Total Welfare:**
```
W(σ) = CS(σ) + Π(σ)
```

**Net Welfare (with policy costs):**
```
Ω(τ) = W(σ(τ)) - C(τ)
```

where C(τ) = 0.5λτ² is the quadratic enforcement cost.

## Output Files

### Figure 1: Equilibrium Outcomes Under Algorithmic Confusion
**File:** `fig1_equilibrium.pdf`

Shows how key market outcomes evolve with confusion intensity (σ):
- Equilibrium price p*(σ)
- Firm profit Π*(σ)
- Welfare loss (normalized)

**Key Insights:**
- Prices increase with confusion as demand becomes less elastic
- Firm profits rise due to enhanced market power
- Social welfare declines convexly

### Figure 2: Transparency Regulation Analysis
**File:** `fig2_transparency.pdf`

Two-panel figure analyzing optimal transparency policy:

**Panel (a) - Welfare-Cost Frontier:**
- Net social welfare Ω(τ)
- Gross welfare W(σ(τ))
- Enforcement costs C(τ)
- Second-best optimal transparency τ*

**Panel (b) - Confusion Reduction:**
- Effective confusion σ(τ) as function of transparency
- Exponential decay pattern
- Optimal confusion level σ(τ*)

## Model Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Demand intercept | a | 100 | Market size parameter |
| Demand slope | b | 2 | Price sensitivity in demand |
| Marginal cost | c | 20 | Unit production cost |
| Baseline elasticity | η₀ | -3 | Price elasticity with no confusion |
| Confusion sensitivity | α | 0.1 | How confusion affects elasticity |
| Initial confusion | σ₀ | 2 | Confusion without regulation |
| Transparency effectiveness | β | 2 | Policy impact on confusion |
| Enforcement cost | λ | 50 | Cost of implementing transparency |

### Parameter Guidelines

- **η₀ < -1:** Ensures elastic demand (required for meaningful monopoly problem)
- **0 < α < 1:** Moderate confusion effect on elasticity
- **σ₀ > 0:** Positive baseline confusion
- **β > 0:** Transparency policy has positive effect
- **λ > 0:** Enforcement is costly

## Customization

### Changing Figure Styles

Modify the `plt.rcParams` dictionary at the top of the script to customize:
- Font family and sizes
- Line widths and styles
- DPI and resolution
- Grid appearance
- Legend formatting

### Adding Analysis

The `AIMarketModel` class can be extended with additional methods:

```python
def deadweight_loss(self, sigma):
    """Calculate deadweight loss from confusion."""
    return self.welfare(0) - self.welfare(sigma)

def price_markup(self, sigma):
    """Calculate price-cost markup."""
    p = self.optimal_price(sigma)
    return (p - self.c) / p
```

### Alternative Specifications

To implement alternative functional forms:

1. Modify `elasticity()` method for different confusion-elasticity relationships
2. Modify `demand()` method for non-linear demand functions
3. Modify `confusion_with_transparency()` for alternative policy effects

## Interpreting Results

### Console Output

The script prints comprehensive summary statistics including:
- Baseline model parameters
- Equilibrium outcomes with no confusion (σ=0)
- Equilibrium outcomes with high confusion (σ=σ₀)
- Welfare loss from confusion
- Optimal transparency policy τ*
- Comparative welfare analysis

### Policy Implications

The model demonstrates:
1. **Market Power Effect:** Algorithmic opacity increases firm profits by reducing price sensitivity
2. **Welfare Loss:** Consumer confusion leads to higher prices and reduced total welfare
3. **Optimal Regulation:** Complete transparency is generally not optimal due to enforcement costs
4. **Second-Best Policy:** Balances welfare gains from reduced confusion against regulatory costs

## Troubleshooting

**Issue:** Figures not displaying
- **Solution:** Ensure matplotlib backend is properly configured. Try adding `plt.switch_backend('TkAgg')` before imports.

**Issue:** Optimization fails to converge
- **Solution:** Check that η₀ < -1 and parameter values are economically meaningful.

**Issue:** PDF files not saving
- **Solution:** Ensure write permissions in output directory and matplotlib has PDF backend support.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{authorYYYY,
  title={Algorithmic Opacity and Consumer Welfare: A Discrete Choice Analysis of AI-Mediated Markets},
  author={[Author Names]},
  journal={[Journal Name]},
  year={YYYY},
  volume={XX},
  pages={XXX-XXX}
}
```

## License

[Specify license - e.g., MIT, Apache 2.0, GPL-3.0]

## Contact

For questions or issues, please contact [contact information] or open an issue on GitHub.

## Acknowledgments

[Add any acknowledgments, funding sources, or contributors]

---

**Last Updated:** October 2025
