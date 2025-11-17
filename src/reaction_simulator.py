"""
Simulates chemical reaction yield based on multiple parameters.
In production, this would interface with real experimental data.
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class ReactionParameters:
    """Reaction condition parameters"""
    temperature: float      # Celsius (80-180)
    pressure: float         # Bar (1-10)
    catalyst_conc: float    # mol% (0.1-5.0)
    reaction_time: float    # Hours (0.5-24)

    def __post_init__(self):
        """Validate parameters"""
        if not 80 <= self.temperature <= 180:
            raise ValueError("Temperature must be between 80-180°C")
        if not 1.0 <= self.pressure <= 10.0:
            raise ValueError("Pressure must be between 1-10 Bar")
        if not 0.1 <= self.catalyst_conc <= 5.0:
            raise ValueError("Catalyst concentration must be between 0.1-5.0 mol%")
        if not 0.5 <= self.reaction_time <= 24.0:
            raise ValueError("Reaction time must be between 0.5-24 hours")

class ReactionSimulator:
    """
    Simulates MDI or polyurethane reaction yields.
    Based on realistic chemical engineering principles.
    """

    def __init__(self, noise_level: float = 0.05, random_seed: int = None):
        """
        Initialize the simulator.

        Args:
            noise_level: Standard deviation of experimental noise (default 0.05)
            random_seed: Seed for reproducibility (default None)
        """
        self.noise_level = noise_level
        if random_seed is not None:
            np.random.seed(random_seed)

        # Optimal conditions (representing ideal MDI reaction conditions)
        self.optimal_temp = 140
        self.optimal_pressure = 5.5
        self.optimal_catalyst = 2.0
        self.optimal_time = 8.0

    def calculate_yield(self, params: ReactionParameters) -> Dict[str, float]:
        """
        Calculate reaction yield and quality metrics.

        Returns:
            Dict with yield, selectivity, cost, and ROI metrics
        """
        # Temperature effect (Arrhenius-like behavior)
        temp_effect = np.exp(-((params.temperature - self.optimal_temp) ** 2) / 800)

        # Pressure effect (linear with optimal point)
        pressure_effect = 1 - abs(params.pressure - self.optimal_pressure) / 10
        pressure_effect = max(0.3, pressure_effect)  # Minimum 30% effect

        # Catalyst effect (logarithmic with diminishing returns)
        catalyst_effect = np.log1p(params.catalyst_conc) / np.log1p(self.optimal_catalyst)

        # Time effect (with diminishing returns after optimum)
        if params.reaction_time < self.optimal_time:
            time_effect = params.reaction_time / self.optimal_time
        else:
            # Penalty for too long (degradation)
            time_effect = 1 - 0.05 * (params.reaction_time - self.optimal_time)
            time_effect = max(0.5, time_effect)

        # Combined yield (0-100%)
        base_yield = 85 * temp_effect * pressure_effect * catalyst_effect * time_effect

        # Add realistic experimental noise
        noise = np.random.normal(0, self.noise_level * base_yield)
        final_yield = np.clip(base_yield + noise, 0, 100)

        # Calculate cost factors (arbitrary units)
        energy_cost = params.temperature * params.reaction_time * 0.5
        catalyst_cost = params.catalyst_conc * 100
        pressure_cost = params.pressure * 20
        time_cost = params.reaction_time * 50
        total_cost = energy_cost + catalyst_cost + pressure_cost + time_cost

        # Selectivity (fewer byproducts at optimal conditions)
        selectivity = final_yield * (0.85 + 0.15 * temp_effect * catalyst_effect)

        # ROI calculation (simplified)
        revenue = final_yield * 1000  # Assume 1000 units revenue per % yield
        roi = (revenue - total_cost) / total_cost if total_cost > 0 else 0

        return {
            'yield': float(final_yield),
            'selectivity': float(selectivity),
            'cost': float(total_cost),
            'roi': float(roi),
            'energy_cost': float(energy_cost),
            'catalyst_cost': float(catalyst_cost)
        }

    def run_experiment(self, temperature: float, pressure: float,
                      catalyst_conc: float, reaction_time: float) -> Dict[str, float]:
        """
        Convenience method for running experiments.

        Args:
            temperature: Reaction temperature in Celsius (80-180)
            pressure: Reaction pressure in Bar (1-10)
            catalyst_conc: Catalyst concentration in mol% (0.1-5.0)
            reaction_time: Reaction time in hours (0.5-24)

        Returns:
            Dictionary with experiment results
        """
        params = ReactionParameters(temperature, pressure, catalyst_conc, reaction_time)
        return self.calculate_yield(params)

    def batch_experiments(self, param_list: list) -> list:
        """
        Run multiple experiments from a list of parameters.

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
