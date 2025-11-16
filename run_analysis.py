"""
Quick demonstration script - generates data, runs analysis, creates visualizations.
Run this to see the framework in action.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import AIEvaluationDataGenerator
from failure_analyzer import FailureAnalyzer
from visualizer import FailureVisualizer
import pandas as pd

def main():
    print("="*70)
    print(" AI TASK EVALUATION - FAILURE ANALYSIS FRAMEWORK")
    print("="*70)
    
    # 1. Generate Data
    print("\n[1/4] Generating synthetic evaluation data...")
    generator = AIEvaluationDataGenerator()
    df = generator.generate_dataset(n_samples=1000)
    generator.save_dataset(df, filepath='data/ai_evaluations.csv')
    
    # 2. Run Analysis
    print("\n[2/4] Running failure analysis...")
    analyzer = FailureAnalyzer(df)
    
    # Overall metrics
    metrics = analyzer.overall_metrics()
    print("\n" + "="*70)
    print("OVERALL PERFORMANCE METRICS")
    print("="*70)
    print(f"Total Tasks.........: {metrics['total_tasks']:,}")
    print(f"Success Rate........: {metrics['success_rate']:.1%}")
    print(f"Failure Rate........: {1-metrics['success_rate']:.1%}")
    print(f"Average Score.......: {metrics['avg_score']:.1f}/100")
    
    # High-risk segments
    print("\n" + "="*70)
    print("HIGH-RISK SEGMENTS (Top 5)")
    print("="*70)
    high_risk = analyzer.identify_high_risk_segments(threshold=0.3)
    print(high_risk.head().to_string(index=False))
    
    # Statistical tests
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*70)
    dimensions = ['task_type', 'file_type', 'finance_domain', 'agent']
    for dim in dimensions:
        test = analyzer.statistical_significance_test(dim)
        sig = "✓ Significant" if test['significant'] else "✗ Not significant"
        print(f"{dim:.<30} p={test['p_value']:.4f} {sig}")
    
    # Root causes
    print("\n" + "="*70)
    print("ROOT CAUSE HYPOTHESES & RECOMMENDATIONS")
    print("="*70)
    hypotheses = analyzer.root_cause_hypothesis()
    for i, hyp in enumerate(hypotheses[:3], 1):
        print(f"\n[{i}] {hyp['dimension'].upper()}")
        print(f"    Worst: {hyp['worst_performer']} ({hyp['failure_rate']:.1%} failure rate)")
        print(f"    → {hyp['recommendation']}")
    
    # 3. Generate Visualizations
    print("\n[3/4] Generating visualizations...")
    viz = FailureVisualizer(df)
    
    os.makedirs('outputs', exist_ok=True)
    
    print("    - Creating dashboard...")
    viz.plot_multidimensional_summary(output_path='outputs/dashboard.png')
    
    print("    - Creating interaction heatmap...")
    viz.plot_interaction_heatmap('task_type', 'file_type', 
                                 output_path='outputs/interaction_heatmap.png')
    
    print("    - Creating temporal trends...")
    viz.plot_temporal_trends(output_path='outputs/temporal_trends.png')
    
    print("    - Creating error distribution...")
    viz.plot_error_type_distribution(output_path='outputs/error_distribution.png')
    
    # 4. Summary
    print("\n[4/4] Analysis complete!")
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print(f"✓ Analyzed {len(df):,} AI task evaluations")
    print(f"✓ Identified {len(high_risk)} high-risk segments")
    print(f"✓ Found {len(hypotheses)} statistically significant patterns")
    print(f"✓ Generated 4 visualizations in outputs/")
    print("\nNext steps:")
    print("  1. Review outputs/ folder for visualizations")
    print("  2. Open notebooks/failure_analysis.ipynb for detailed walkthrough")
    print("  3. Customize analysis for your specific use case")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
