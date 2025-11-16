"""
Generate synthetic AI evaluation data immediately.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generator import AIEvaluationDataGenerator

# Generate data
print("Generating synthetic AI evaluation data...")
generator = AIEvaluationDataGenerator()
df = generator.generate_dataset(n_samples=1000)

# Save it
os.makedirs('data', exist_ok=True)
generator.save_dataset(df, filepath='data/ai_evaluations.csv')

print("\n✓ Data generated successfully!")
print(f"  Location: data/ai_evaluations.csv")
print(f"  Records: {len(df):,}")
print(f"\nSample of the data:")
print(df.head(10))
print(f"\nColumn types:")
print(df.dtypes)
