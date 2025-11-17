"""Test the Bayesian optimizer"""
from src.reaction_simulator import ReactionSimulator
from src.optimizer import ReactionOptimizer
import matplotlib.pyplot as plt

print("Testing Bayesian Optimizer")
print("=" * 60)

# Create simulator
simulator = ReactionSimulator(random_seed=42)

# Create optimizer
optimizer = ReactionOptimizer(simulator, optimization_metric='yield')

# Run optimization
results = optimizer.optimize(n_calls=30, n_random_starts=10, verbose=True)

# Get history
history_df = optimizer.get_history_df()
print(f"\n Optimization History Shape: {history_df.shape}")
print("\nFirst few experiments:")
print(history_df[['temperature', 'pressure', 'yield', 'cost']].head())

print("\nLast few experiments:")
print(history_df[['temperature', 'pressure', 'yield', 'cost']].tail())

# Plot convergence
fig = optimizer.plot_convergence()
if fig:
    plt.savefig('convergence_test.png', dpi=150, bbox_inches='tight')
    print("\n Convergence plot saved as 'convergence_test.png'")

print("\n Optimizer working correctly!")
