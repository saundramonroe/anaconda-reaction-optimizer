"""
Bayesian optimization for reaction parameters using scikit-optimize.
"""

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence, plot_objective
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from .reaction_simulator import ReactionSimulator, ReactionParameters

class ReactionOptimizer:
    """
    Optimizes reaction conditions using Bayesian optimization.
    """

    def __init__(self, simulator: ReactionSimulator, optimization_metric: str = 'yield'):
        """
        Initialize the optimizer.

        Args:
            simulator: ReactionSimulator instance
            optimization_metric: Metric to optimize ('yield', 'roi', 'selectivity')
        """
        self.simulator = simulator
        self.optimization_metric = optimization_metric

        # Define search space for reaction parameters
        self.space = [
            Real(80, 180, name='temperature'),
            Real(1.0, 10.0, name='pressure'),
            Real(0.1, 5.0, name='catalyst_conc'),
            Real(0.5, 24.0, name='reaction_time')
        ]

        self.dimension_names = ['temperature', 'pressure', 'catalyst_conc', 'reaction_time']

        self.results_history = []
        self.optimization_result = None

    def objective_function(self, params: List[float]) -> float:
        """
        Objective function to minimize (negative of metric since we minimize).

        Args:
            params: List of [temperature, pressure, catalyst_conc, reaction_time]

        Returns:
            Negative metric value (for minimization)
        """
        temp, pressure, catalyst, time = params

        try:
            results = self.simulator.run_experiment(temp, pressure, catalyst, time)
        except ValueError as e:
            # Return high penalty for invalid parameters
            return 1e6

        # Store history
        self.results_history.append({
            'iteration': len(self.results_history),
            'temperature': temp,
            'pressure': pressure,
            'catalyst_conc': catalyst,
            'reaction_time': time,
            **results
        })

        # Return negative because we're minimizing
        return -results[self.optimization_metric]

    def optimize(self, n_calls: int = 50, n_random_starts: int = 10,
                 verbose: bool = True) -> Dict:
        """
        Run Bayesian optimization.

        Args:
            n_calls: Total number of evaluations
            n_random_starts: Number of random exploration points at start
            verbose: Print progress information

        Returns:
            Dictionary with optimal parameters and results
        """
        if verbose:
            print(f"Starting Bayesian Optimization")
            print(f"=" * 60)
            print(f"Optimization metric: {self.optimization_metric}")
            print(f"Total evaluations: {n_calls}")
            print(f"Random starts: {n_random_starts}")
            print(f"Bayesian iterations: {n_calls - n_random_starts}")
            print("")

        # Clear previous history
        self.results_history = []

        # Run optimization
        self.optimization_result = gp_minimize(
            self.objective_function,
            self.space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            random_state=42,
            verbose=False,
            n_jobs=1
        )

        # Extract optimal parameters
        optimal_params = {
            'temperature': self.optimization_result.x[0],
            'pressure': self.optimization_result.x[1],
            'catalyst_conc': self.optimization_result.x[2],
            'reaction_time': self.optimization_result.x[3]
        }

        # Get results at optimal point
        optimal_results = self.simulator.run_experiment(**optimal_params)

        # Calculate improvement
        improvement = self._calculate_improvement(n_random_starts)

        if verbose:
            print(f"\n{'='*60}")
            print(f"🎉 Optimization Complete!")
            print(f"{'='*60}")
            print(f"\nOptimal Parameters:")
            for key, value in optimal_params.items():
                print(f"  {key:20s}: {value:8.2f}")

            print(f"\nOptimal Results:")
            for key, value in optimal_results.items():
                print(f"  {key:20s}: {value:8.2f}")

            print(f"\nPerformance:")
            print(f"  Best {self.optimization_metric:12s}: {optimal_results[self.optimization_metric]:.2f}")
            print(f"  Improvement vs random : {improvement:.1f}%")
            print(f"  Total experiments     : {n_calls}")

        return {
            'optimal_params': optimal_params,
            'optimal_results': optimal_results,
            'n_evaluations': n_calls,
            'improvement': improvement,
            'best_value': optimal_results[self.optimization_metric]
        }

    def _calculate_improvement(self, n_random: int) -> float:
        """Calculate improvement vs random search baseline"""
        if len(self.results_history) < n_random:
            return 0.0

        df = pd.DataFrame(self.results_history)

        # Best of first n_random (random phase)
        random_best = df.iloc[:n_random][self.optimization_metric].max()

        # Best overall
        overall_best = df[self.optimization_metric].max()

        if random_best == 0:
            return 0.0

        return ((overall_best - random_best) / random_best) * 100

    def get_history_df(self) -> pd.DataFrame:
        """Get optimization history as DataFrame"""
        return pd.DataFrame(self.results_history)

    def plot_convergence(self, figsize=(10, 6)):
        """Plot optimization convergence"""
        if self.optimization_result is None:
            print("⚠️  Run optimization first!")
            return None

        fig, ax = plt.subplots(figsize=figsize)
        plot_convergence(self.optimization_result, ax=ax)
        ax.set_title(f"Convergence Plot - Optimizing {self.optimization_metric}")
        ax.set_xlabel("Number of Evaluations")
        ax.set_ylabel(f"Best {self.optimization_metric} Found")
        plt.tight_layout()
        return fig

    def plot_parameter_importance(self, figsize=(10, 8)):
        """Plot which parameters matter most"""
        if self.optimization_result is None:
            print("⚠️  Run optimization first!")
            return None

        try:
            fig, ax = plt.subplots(figsize=figsize)
            plot_objective(self.optimization_result, dimensions=self.dimension_names)
            plt.suptitle("Parameter Importance Analysis")
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Could not create parameter importance plot: {e}")
            return None
