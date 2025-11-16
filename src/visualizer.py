"""
Visualization module for failure analysis results.
Creates publication-ready charts and dashboards.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class FailureVisualizer:
    """Creates visualizations for failure analysis insights."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.failures = df[df['success'] == False]
        self.successes = df[df['success'] == True]
        
    def plot_failure_rates_by_dimension(self, dimension: str, 
                                       output_path: str = None) -> None:
        """Bar chart of failure rates across a dimension."""
        failure_rates = 1 - self.df.groupby(dimension)['success'].mean()
        counts = self.df.groupby(dimension).size()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Failure rates
        failure_rates.sort_values(ascending=True).plot(
            kind='barh', ax=ax1, color='#e74c3c'
        )
        ax1.set_xlabel('Failure Rate')
        ax1.set_title(f'Failure Rate by {dimension}')
        ax1.axvline(x=failure_rates.mean(), color='black', 
                   linestyle='--', label='Mean')
        ax1.legend()
        
        # Sample sizes
        counts.sort_values(ascending=True).plot(
            kind='barh', ax=ax2, color='#3498db'
        )
        ax2.set_xlabel('Number of Tasks')
        ax2.set_title(f'Sample Size by {dimension}')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_interaction_heatmap(self, dim1: str, dim2: str,
                                 output_path: str = None) -> None:
        """Heatmap showing failure rates across two dimensions."""
        pivot = self.df.pivot_table(
            values='success',
            index=dim1,
            columns=dim2,
            aggfunc='mean'
        )
        
        failure_pivot = 1 - pivot
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(failure_pivot, annot=True, fmt='.2%', 
                   cmap='RdYlGn_r', center=0.3,
                   cbar_kws={'label': 'Failure Rate'})
        plt.title(f'Failure Rate Heatmap: {dim1} vs {dim2}')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_type_distribution(self, output_path: str = None) -> None:
        """Pie chart of error types among failures."""
        if self.failures.empty:
            print("No failures to visualize")
            return
        
        error_counts = self.failures['error_type'].value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        colors = sns.color_palette('Set2', len(error_counts))
        ax1.pie(error_counts, labels=error_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax1.set_title('Distribution of Error Types')
        
        # Bar chart
        error_counts.plot(kind='bar', ax=ax2, color=colors)
        ax2.set_xlabel('Error Type')
        ax2.set_ylabel('Count')
        ax2.set_title('Error Type Frequency')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_score_distributions(self, output_path: str = None) -> None:
        """Compare score distributions between successes and failures."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Score distribution
        axes[0, 0].hist(self.successes['score'], bins=30, alpha=0.6, 
                       label='Success', color='green', edgecolor='black')
        axes[0, 0].hist(self.failures['score'], bins=30, alpha=0.6,
                       label='Failure', color='red', edgecolor='black')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Score Distribution: Success vs Failure')
        axes[0, 0].legend()
        
        # Processing time
        axes[0, 1].hist(self.successes['processing_time_sec'], bins=30, 
                       alpha=0.6, label='Success', color='green', edgecolor='black')
        axes[0, 1].hist(self.failures['processing_time_sec'], bins=30,
                       alpha=0.6, label='Failure', color='red', edgecolor='black')
        axes[0, 1].set_xlabel('Processing Time (sec)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Processing Time: Success vs Failure')
        axes[0, 1].legend()
        
        # Complexity score
        axes[1, 0].hist(self.successes['complexity_score'], bins=10,
                       alpha=0.6, label='Success', color='green', edgecolor='black')
        axes[1, 0].hist(self.failures['complexity_score'], bins=10,
                       alpha=0.6, label='Failure', color='red', edgecolor='black')
        axes[1, 0].set_xlabel('Complexity Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Task Complexity: Success vs Failure')
        axes[1, 0].legend()
        
        # File size
        axes[1, 1].hist(self.successes['file_size_kb'], bins=30,
                       alpha=0.6, label='Success', color='green', edgecolor='black')
        axes[1, 1].hist(self.failures['file_size_kb'], bins=30,
                       alpha=0.6, label='Failure', color='red', edgecolor='black')
        axes[1, 1].set_xlabel('File Size (KB)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('File Size: Success vs Failure')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_trends(self, output_path: str = None) -> None:
        """Plot failure rates over time."""
        self.df['date'] = pd.to_datetime(self.df['timestamp']).dt.date
        
        temporal = self.df.groupby('date').agg({
            'success': 'mean',
            'score': 'mean'
        })
        
        temporal['failure_rate'] = 1 - temporal['success']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Failure rate over time
        ax1.plot(temporal.index, temporal['failure_rate'], 
                marker='o', color='#e74c3c', linewidth=2)
        ax1.fill_between(temporal.index, temporal['failure_rate'], 
                        alpha=0.3, color='#e74c3c')
        ax1.set_ylabel('Failure Rate')
        ax1.set_title('Failure Rate Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=temporal['failure_rate'].mean(), 
                   color='black', linestyle='--', label='Mean')
        ax1.legend()
        
        # Average score over time
        ax2.plot(temporal.index, temporal['score'], 
                marker='s', color='#3498db', linewidth=2)
        ax2.fill_between(temporal.index, temporal['score'], 
                        alpha=0.3, color='#3498db')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Average Score')
        ax2.set_title('Average Task Score Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_multidimensional_summary(self, output_path: str = None) -> None:
        """Create a comprehensive dashboard view."""
        dimensions = ['task_type', 'file_type', 'finance_domain', 
                     'evaluation_criterion', 'agent']
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, dim in enumerate(dimensions):
            failure_rates = 1 - self.df.groupby(dim)['success'].mean()
            failure_rates.sort_values(ascending=True).plot(
                kind='barh', ax=axes[idx], color='#e74c3c'
            )
            axes[idx].set_xlabel('Failure Rate')
            axes[idx].set_title(f'{dim}')
            axes[idx].axvline(x=failure_rates.mean(), 
                            color='black', linestyle='--', alpha=0.5)
        
        # Overall metrics in the last subplot
        axes[-1].axis('off')
        overall_text = f"""
        Overall Performance Summary
        ─────────────────────────
        Total Tasks: {len(self.df):,}
        Success Rate: {self.df['success'].mean():.1%}
        Failure Rate: {(1-self.df['success'].mean()):.1%}
        Avg Score: {self.df['score'].mean():.1f}
        Avg Time: {self.df['processing_time_sec'].mean():.1f}s
        """
        axes[-1].text(0.1, 0.5, overall_text, fontsize=12,
                     family='monospace', verticalalignment='center')
        
        plt.suptitle('AI Task Evaluation - Failure Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    df = pd.read_csv('data/ai_evaluations.csv')
    viz = FailureVisualizer(df)
    
    viz.plot_multidimensional_summary(output_path='outputs/dashboard.png')
