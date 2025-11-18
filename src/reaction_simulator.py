"""
Chemical Reaction Simulator
Simulates reaction yields based on multiple parameters.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ReactionParameters:
    """Container for reaction parameters"""
    temperature: float      # Celsius (80-180)
    pressure: float         # Bar (1-10)
    catalyst_conc: float    # mol% (0.1-5.0)
    reaction_time: float    # Hours (0.5-24)

class ReactionSimulator:
    """
    Simulates chemical reaction yields based on parameters.
    Models MDI or polyurethane reaction conditions.
    """

    def __init__(self, noise_level: float = 0.05, random_seed: int = None):
        """
        Initialize simulator.

        Args:
            noise_level: Experimental noise level (default 0.05)
            random_seed: Random seed for reproducibility
        """
        self.noise_level = noise_level
        if random_seed is not None:
            np.random.seed(random_seed)

        # Optimal conditions (representing ideal reaction conditions)
        self.optimal_temp = 140.0
        self.optimal_pressure = 5.5
        self.optimal_catalyst = 2.0
        self.optimal_time = 8.0

    def calculate_yield(self, params: ReactionParameters) -> Dict[str, float]:
        """
        Calculate reaction yield and quality metrics.

        Args:
            params: ReactionParameters object

        Returns:
            Dictionary with yield, selectivity, cost, and ROI
        """
        # Temperature effect (Arrhenius-like)
        temp_effect = np.exp(-((params.temperature - self.optimal_temp) ** 2) / 800)

        # Pressure effect
        pressure_effect = 1 - abs(params.pressure - self.optimal_pressure) / 10
        pressure_effect = max(0.3, pressure_effect)

        # Catalyst effect (logarithmic with diminishing returns)
        catalyst_effect = np.log1p(params.catalyst_conc) / np.log1p(self.optimal_catalyst)

        # Time effect
        if params.reaction_time < self.optimal_time:
            time_effect = params.reaction_time / self.optimal_time
        else:
            time_effect = 1 - 0.05 * (params.reaction_time - self.optimal_time)
            time_effect = max(0.5, time_effect)

        # Combined yield
        base_yield = 85 * temp_effect * pressure_effect * catalyst_effect * time_effect
        noise = np.random.normal(0, self.noise_level * base_yield)
        final_yield = np.clip(base_yield + noise, 0, 100)

        # Cost calculation
        energy_cost = params.temperature * params.reaction_time * 0.5
        catalyst_cost = params.catalyst_conc * 100
        pressure_cost = params.pressure * 20
        time_cost = params.reaction_time * 50
        total_cost = energy_cost + catalyst_cost + pressure_cost + time_cost

        # Selectivity
        selectivity = final_yield * (0.85 + 0.15 * temp_effect * catalyst_effect)

        # ROI
        revenue = final_yield * 1000
        roi = (revenue - total_cost) / total_cost if total_cost > 0 else 0

        return {
            'yield': float(final_yield),
            'selectivity': float(selectivity),
            'cost': float(total_cost),
            'roi': float(roi),
            'energy_cost': float(energy_cost),
            'catalyst_cost': float(catalyst_cost),
            'pressure_cost': float(pressure_cost),
            'time_cost': float(time_cost)
        }

    def run_experiment(self, temperature: float, pressure: float,
                      catalyst_conc: float, reaction_time: float) -> Dict[str, float]:
        """
        Run a single experiment.

        Args:
            temperature: Reaction temperature (°C)
            pressure: Reaction pressure (Bar)
            catalyst_conc: Catalyst concentration (mol%)
            reaction_time: Reaction time (hours)

        Returns:
            Dictionary with experiment results
        """
        params = ReactionParameters(temperature, pressure, catalyst_conc, reaction_time)
        return self.calculate_yield(params)

    def get_optimal_conditions(self) -> Dict[str, float]:
        """
        Return the simulator's optimal conditions.

        Returns:
            Dictionary with optimal parameter values
        """
        return {
            'temperature': self.optimal_temp,
            'pressure': self.optimal_pressure,
            'catalyst_conc': self.optimal_catalyst,
            'reaction_time': self.optimal_time
        }

    def batch_experiments(self, param_list: List[Tuple[float, float, float, float]]) -> List[Dict[str, float]]:
        """
        Run multiple experiments in batch.

        Args:
            param_list: List of tuples (temp, pressure, catalyst, time)

        Returns:
            List of result dictionaries
        """
        results = []
        for temp, pressure, catalyst, time in param_list:
            result = self.run_experiment(temp, pressure, catalyst, time)
            result.update({
                'temperature': temp,
                'pressure': pressure,
                'catalyst_conc': catalyst,
                'reaction_time': time
            })
            results.append(result)
        return results
