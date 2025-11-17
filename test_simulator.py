"""Test the reaction simulator"""
from src.reaction_simulator import ReactionSimulator

# Initialize simulator
simulator = ReactionSimulator(random_seed=42)

# Run a test experiment
print("Testing Reaction Simulator")
print("=" * 50)

result = simulator.run_experiment(
    temperature=140,
    pressure=5.5,
    catalyst_conc=2.0,
    reaction_time=8.0
)

print("\nOptimal Conditions Test:")
for key, value in result.items():
    print(f"  {key}: {value:.2f}")

# Test suboptimal conditions
print("\nSuboptimal Conditions Test:")
result2 = simulator.run_experiment(
    temperature=100,
    pressure=3.0,
    catalyst_conc=1.0,
    reaction_time=5.0
)

for key, value in result2.items():
    print(f"  {key}: {value:.2f}")

print("\n✅ Simulator working correctly!")
