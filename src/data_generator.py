"""
Generate synthetic AI task evaluation data for failure analysis.
Simulates realistic patterns in AI agent performance across finance tasks.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)


class AIEvaluationDataGenerator:
    """Generates synthetic evaluation data with realistic failure patterns."""
    
    def __init__(self):
        self.task_types = [
            'Financial Analysis', 'Risk Assessment', 'Report Generation',
            'Data Extraction', 'Compliance Check', 'Market Research'
        ]
        
        self.file_types = ['PDF', 'Excel', 'CSV', 'JSON', 'Word']
        
        self.finance_domains = [
            'Investment Banking', 'Retail Banking', 'Insurance',
            'Asset Management', 'Trading', 'Risk Management'
        ]
        
        self.criteria = [
            'Accuracy', 'Completeness', 'Formatting',
            'Compliance', 'Timeliness', 'Data Quality'
        ]
        
        self.agents = ['Agent_A', 'Agent_B', 'Agent_C']
        
    def generate_dataset(self, n_samples=1000):
        """Generate evaluation dataset with built-in failure patterns."""
        
        data = []
        start_date = datetime(2024, 1, 1)
        
        for i in range(n_samples):
            task_type = np.random.choice(self.task_types)
            file_type = np.random.choice(self.file_types)
            domain = np.random.choice(self.finance_domains)
            criterion = np.random.choice(self.criteria)
            agent = np.random.choice(self.agents)
            
            # Inject realistic failure patterns
            base_success_rate = 0.75
            
            # Pattern 1: PDFs are harder
            if file_type == 'PDF':
                base_success_rate -= 0.15
                
            # Pattern 2: Complex financial analysis is harder
            if task_type == 'Financial Analysis':
                base_success_rate -= 0.10
                
            # Pattern 3: Compliance checks fail more with certain file types
            if criterion == 'Compliance' and file_type in ['JSON', 'CSV']:
                base_success_rate -= 0.12
                
            # Pattern 4: Some domains are more challenging
            if domain == 'Investment Banking':
                base_success_rate -= 0.08
                
            # Pattern 5: Agent performance varies
            agent_adjustment = {'Agent_A': 0.05, 'Agent_B': 0, 'Agent_C': -0.08}
            base_success_rate += agent_adjustment[agent]
            
            # Pattern 6: Time-based degradation
            if i > n_samples * 0.7:  # Later tasks show degradation
                base_success_rate -= 0.05
            
            # Determine success/failure
            success = np.random.random() < np.clip(base_success_rate, 0.1, 0.95)
            
            # Score (0-100)
            if success:
                score = np.random.normal(85, 8)
            else:
                score = np.random.normal(45, 15)
            score = np.clip(score, 0, 100)
            
            # Error types for failures
            error_types = [
                'Hallucination', 'Missing Data', 'Format Error',
                'Logic Error', 'Timeout', 'Incomplete Output'
            ]
            error_type = np.random.choice(error_types) if not success else None
            
            # Processing time (seconds)
            base_time = 30
            if file_type == 'PDF':
                base_time += 15
            if task_type == 'Financial Analysis':
                base_time += 20
            processing_time = np.random.normal(base_time, 10)
            
            data.append({
                'task_id': f'TASK_{i:04d}',
                'timestamp': start_date + timedelta(hours=i),
                'task_type': task_type,
                'file_type': file_type,
                'finance_domain': domain,
                'evaluation_criterion': criterion,
                'agent': agent,
                'success': success,
                'score': round(score, 2),
                'error_type': error_type,
                'processing_time_sec': round(processing_time, 2),
                'file_size_kb': np.random.randint(10, 5000),
                'complexity_score': np.random.randint(1, 11)
            })
        
        df = pd.DataFrame(data)
        return df
    
    def save_dataset(self, df, filepath='data/ai_evaluations.csv'):
        """Save generated dataset to CSV."""
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        print(f"Shape: {df.shape}")
        print(f"Success rate: {df['success'].mean():.2%}")


if __name__ == "__main__":
    generator = AIEvaluationDataGenerator()
    df = generator.generate_dataset(n_samples=1000)
    generator.save_dataset(df)
