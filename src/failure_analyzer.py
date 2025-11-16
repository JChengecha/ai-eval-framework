"""
Core failure analysis module for AI task evaluation.
Identifies patterns, root causes, and statistical anomalies.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FailureAnalyzer:
    """Analyzes failure patterns across multiple dimensions."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.failures = df[df['success'] == False].copy()
        self.successes = df[df['success'] == True].copy()
        
    def overall_metrics(self) -> Dict:
        """Calculate high-level performance metrics."""
        return {
            'total_tasks': len(self.df),
            'success_count': len(self.successes),
            'failure_count': len(self.failures),
            'success_rate': len(self.successes) / len(self.df),
            'avg_score': self.df['score'].mean(),
            'avg_processing_time': self.df['processing_time_sec'].mean()
        }
    
    def failure_by_dimension(self, dimension: str) -> pd.DataFrame:
        """Analyze failure rates across a single dimension."""
        grouped = self.df.groupby(dimension).agg({
            'success': ['count', 'sum', 'mean'],
            'score': 'mean',
            'processing_time_sec': 'mean'
        }).round(3)
        
        grouped.columns = ['total_tasks', 'successes', 'success_rate', 
                          'avg_score', 'avg_time']
        grouped['failure_rate'] = 1 - grouped['success_rate']
        grouped = grouped.sort_values('failure_rate', ascending=False)
        
        return grouped
    
    def multidimensional_analysis(self) -> pd.DataFrame:
        """Analyze failure patterns across multiple dimensions simultaneously."""
        dimensions = ['task_type', 'file_type', 'finance_domain', 
                     'evaluation_criterion', 'agent']
        
        results = []
        for dim in dimensions:
            dim_analysis = self.failure_by_dimension(dim)
            dim_analysis['dimension'] = dim
            dim_analysis['category'] = dim_analysis.index
            results.append(dim_analysis.reset_index(drop=True))
        
        combined = pd.concat(results, ignore_index=True)
        return combined[['dimension', 'category', 'total_tasks', 'failure_rate', 
                        'avg_score', 'avg_time']]
    
    def interaction_analysis(self, dim1: str, dim2: str) -> pd.DataFrame:
        """Analyze failure patterns across two dimensions (interaction effects)."""
        pivot = self.df.pivot_table(
            values='success',
            index=dim1,
            columns=dim2,
            aggfunc=['count', 'mean']
        )
        
        failure_rates = 1 - pivot['mean']
        counts = pivot['count']
        
        return failure_rates, counts
    
    def statistical_significance_test(self, dimension: str) -> Dict:
        """
        Test if failure rates differ significantly across dimension categories.
        Uses chi-square test for independence.
        """
        contingency_table = pd.crosstab(self.df[dimension], self.df['success'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        return {
            'dimension': dimension,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05
        }
    
    def identify_high_risk_segments(self, threshold: float = 0.4) -> pd.DataFrame:
        """Identify segments with failure rates above threshold."""
        high_risk = []
        
        dimensions = ['task_type', 'file_type', 'finance_domain', 
                     'evaluation_criterion', 'agent']
        
        for dim in dimensions:
            analysis = self.failure_by_dimension(dim)
            risky = analysis[analysis['failure_rate'] > threshold]
            
            for idx, row in risky.iterrows():
                high_risk.append({
                    'dimension': dim,
                    'category': idx,
                    'failure_rate': row['failure_rate'],
                    'sample_size': row['total_tasks'],
                    'avg_score': row['avg_score']
                })
        
        return pd.DataFrame(high_risk).sort_values('failure_rate', ascending=False)
    
    def error_type_distribution(self) -> pd.DataFrame:
        """Analyze distribution of error types among failures."""
        if self.failures.empty:
            return pd.DataFrame()
        
        error_dist = self.failures['error_type'].value_counts()
        error_pct = self.failures['error_type'].value_counts(normalize=True)
        
        result = pd.DataFrame({
            'count': error_dist,
            'percentage': error_pct * 100
        }).round(2)
        
        return result
    
    def correlation_analysis(self) -> pd.DataFrame:
        """Analyze correlations between numerical features and success."""
        numerical_cols = ['score', 'processing_time_sec', 'file_size_kb', 
                         'complexity_score']
        
        correlations = []
        for col in numerical_cols:
            corr = self.df[col].corr(self.df['success'].astype(int))
            
            # T-test between successes and failures
            success_vals = self.successes[col]
            failure_vals = self.failures[col]
            t_stat, p_value = stats.ttest_ind(success_vals, failure_vals)
            
            correlations.append({
                'feature': col,
                'correlation_with_success': corr,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_difference': p_value < 0.05
            })
        
        return pd.DataFrame(correlations)
    
    def temporal_analysis(self) -> pd.DataFrame:
        """Analyze how failure rates change over time."""
        self.df['date'] = pd.to_datetime(self.df['timestamp']).dt.date
        
        temporal = self.df.groupby('date').agg({
            'success': ['count', 'mean'],
            'score': 'mean',
            'processing_time_sec': 'mean'
        }).round(3)
        
        temporal.columns = ['total_tasks', 'success_rate', 'avg_score', 'avg_time']
        temporal['failure_rate'] = 1 - temporal['success_rate']
        
        return temporal
    
    def root_cause_hypothesis(self) -> List[Dict]:
        """
        Generate hypotheses about root causes based on statistical patterns.
        """
        hypotheses = []
        
        # Test each dimension for significance
        dimensions = ['task_type', 'file_type', 'finance_domain', 
                     'evaluation_criterion', 'agent']
        
        for dim in dimensions:
            sig_test = self.statistical_significance_test(dim)
            if sig_test['significant']:
                analysis = self.failure_by_dimension(dim)
                worst_category = analysis['failure_rate'].idxmax()
                worst_rate = analysis['failure_rate'].max()
                
                hypotheses.append({
                    'dimension': dim,
                    'hypothesis': f"Failures are significantly associated with {dim}",
                    'evidence': f"Chi-square test p-value: {sig_test['p_value']:.4f}",
                    'worst_performer': worst_category,
                    'failure_rate': worst_rate,
                    'recommendation': self._get_recommendation(dim, worst_category)
                })
        
        return hypotheses
    
    def _get_recommendation(self, dimension: str, category: str) -> str:
        """Generate recommendations based on failure patterns."""
        recommendations = {
            'task_type': f"Review task design and rubrics for {category} tasks. Consider breaking down complex tasks.",
            'file_type': f"Improve file parsing and extraction for {category} files. May need specialized preprocessing.",
            'finance_domain': f"Enhance domain knowledge and training data for {category}. Consider domain-specific fine-tuning.",
            'evaluation_criterion': f"Clarify evaluation rubrics for {category}. May need more specific guidelines.",
            'agent': f"Investigate {category} configuration and training. Consider model updates or parameter tuning."
        }
        return recommendations.get(dimension, "Further investigation needed.")


if __name__ == "__main__":
    # Test the analyzer
    df = pd.read_csv('data/ai_evaluations.csv')
    analyzer = FailureAnalyzer(df)
    
    print("Overall Metrics:")
    print(analyzer.overall_metrics())
    
    print("\nFailure by File Type:")
    print(analyzer.failure_by_dimension('file_type'))
