"""Bayesian Optimizer for Chemical Reactions"""
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
import warnings
warnings.filterwarnings('ignore')

from .reaction_simulator import ReactionSimulator

class ReactionOptimizer:
    """Bayesian optimizer for reaction conditions"""

    def __init__(self, simulator: ReactionSimulator, optimization_metric: str = 'yield'):
        self.simulator = simulator
        self.optimization_metric = optimization_metric

        # Define search space
        self.space = [
            Real(80, 180, name='temperature'),
            Real(1.0, 10.0, name='pressure'),
            Real(0.1, 5.0, name='catalyst_conc'),
            Real(0.5, 24.0, name='reaction_time')
        ]

        self.results_history = []
        self.optimization_result = None

    def objective_function(self, params):
        """Objective function to minimize"""
        temp, pressure, catalyst, time = params

        try:
            results = self.simulator.run_experiment(temp, pressure, catalyst, time)
        except:
            return 1e6

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

    def optimize(self, n_calls: int = 50, n_random_starts: int = 10, verbose: bool = True):
        """Run Bayesian optimization"""
        if verbose:
            print(f"Starting optimization with {n_calls} evaluations...")

        self.results_history = []

        # Run optimization
        self.optimization_result = gp_minimize(
            self.objective_function,
            self.space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            random_state=42,
            verbose=False
        )

        # Extract optimal parameters
        optimal_params = {
            'temperature': self.optimization_result.x[0],
            'pressure': self.optimization_result.x[1],
            'catalyst_conc': self.optimization_result.x[2],
            'reaction_time': self.optimization_result.x[3]
        }

        optimal_results = self.simulator.run_experiment(**optimal_params)

        # Calculate improvement
        df = pd.DataFrame(self.results_history)
        random_best = df.iloc[:n_random_starts][self.optimization_metric].max()
        overall_best = df[self.optimization_metric].max()
        improvement = ((overall_best - random_best) / random_best * 100) if random_best > 0 else 0

        if verbose:
            print(f"Optimization complete! Best {self.optimization_metric}: {optimal_results[self.optimization_metric]:.2f}")

        return {
            'optimal_params': optimal_params,
            'optimal_results': optimal_results,
            'n_evaluations': n_calls,
            'improvement': improvement,
            'best_value': optimal_results[self.optimization_metric]
        }

    def get_history_df(self):
        """Get optimization history as DataFrame"""
        return pd.DataFrame(self.results_history)
