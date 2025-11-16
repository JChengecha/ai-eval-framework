# AI Task Evaluation - Failure Analysis Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive statistical framework for analyzing AI agent performance and identifying failure patterns across multiple dimensions. Built for evaluating LLM-based systems in production environments, particularly in finance sector applications.

## 🎯 Overview

This framework enables data-driven analysis of AI task evaluation results, identifying:
- **Failure patterns** across task types, file formats, domains, and evaluation criteria
- **Root causes** through multi-dimensional statistical analysis
- **High-risk segments** requiring immediate attention
- **Temporal trends** in model performance degradation
- **Actionable recommendations** for system improvements

## 🚀 Key Features

### Statistical Analysis
- **Multi-dimensional failure pattern detection** - Analyze performance across 5+ dimensions simultaneously
- **Chi-square significance testing** - Validate that observed patterns aren't due to chance
- **Correlation analysis** - Identify relationships between features and success rates
- **Root cause hypothesis generation** - Data-driven explanations for failure clusters

### Visualization
- **Interactive dashboards** - Comprehensive visual analytics for stakeholder communication
- **Failure heatmaps** - Interaction effects between dimensions
- **Temporal trend analysis** - Track performance over time
- **Error distribution plots** - Understand failure modes

### Production-Ready
- **Modular architecture** - Easy to integrate into existing pipelines
- **Scalable design** - Handles thousands of evaluations efficiently
- **Export capabilities** - Generate publication-ready reports
- **Extensible framework** - Add custom dimensions and metrics

## 📊 Use Cases

Perfect for:
- **AI/ML Quality Assurance Teams** - Monitor production model performance
- **Data Labeling Operations** - Improve evaluation rubrics and task design
- **Model Evaluation** - Benchmark multiple models systematically
- **Finance AI Applications** - Domain-specific failure analysis for banking, trading, risk management

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-eval-framework.git
cd ai-eval-framework

# Install dependencies
pip install -r requirements.txt
```

## 📖 Quick Start

### 1. Generate Sample Data
```python
from src.data_generator import AIEvaluationDataGenerator

generator = AIEvaluationDataGenerator()
df = generator.generate_dataset(n_samples=1000)
generator.save_dataset(df, filepath='data/ai_evaluations.csv')
```

### 2. Run Failure Analysis
```python
from src.failure_analyzer import FailureAnalyzer

analyzer = FailureAnalyzer(df)

# Overall metrics
metrics = analyzer.overall_metrics()
print(f"Success Rate: {metrics['success_rate']:.1%}")

# Dimension analysis
file_analysis = analyzer.failure_by_dimension('file_type')
print(file_analysis)

# Statistical significance
sig_test = analyzer.statistical_significance_test('task_type')
print(f"Chi-square p-value: {sig_test['p_value']:.4f}")

# High-risk segments
high_risk = analyzer.identify_high_risk_segments(threshold=0.35)
print(high_risk)

# Root cause hypotheses
hypotheses = analyzer.root_cause_hypothesis()
for hyp in hypotheses:
    print(f"{hyp['dimension']}: {hyp['recommendation']}")
```

### 3. Create Visualizations
```python
from src.visualizer import FailureVisualizer

viz = FailureVisualizer(df)

# Comprehensive dashboard
viz.plot_multidimensional_summary(output_path='outputs/dashboard.png')

# Specific analyses
viz.plot_failure_rates_by_dimension('file_type')
viz.plot_interaction_heatmap('task_type', 'file_type')
viz.plot_temporal_trends()
```

### 4. Run Complete Analysis (Jupyter Notebook)
```bash
jupyter notebook notebooks/failure_analysis.ipynb
```

## 📁 Project Structure

```
ai-eval-framework/
├── src/
│   ├── data_generator.py      # Synthetic data generation with realistic patterns
│   ├── failure_analyzer.py    # Core statistical analysis engine
│   └── visualizer.py           # Visualization and reporting
├── notebooks/
│   └── failure_analysis.ipynb # Complete walkthrough with examples
├── data/
│   └── ai_evaluations.csv     # Generated evaluation dataset
├── outputs/
│   └── *.png                   # Generated charts and dashboards
├── requirements.txt
└── README.md
```

## 🔬 Analysis Capabilities

### Dimensions Analyzed
- **Task Types** - Financial Analysis, Risk Assessment, Report Generation, Data Extraction, etc.
- **File Types** - PDF, Excel, CSV, JSON, Word documents
- **Finance Domains** - Investment Banking, Retail Banking, Insurance, Trading, Risk Management
- **Evaluation Criteria** - Accuracy, Completeness, Formatting, Compliance, Timeliness
- **AI Agents** - Multi-agent comparison

### Statistical Methods
- **Chi-square tests** for independence
- **T-tests** for numerical feature differences
- **Correlation analysis** between features and success
- **Confidence intervals** for failure rates
- **Temporal trend analysis** with moving averages

### Output Formats
- CSV exports of analysis results
- PNG/PDF visualizations
- Jupyter notebook reports
- Structured data for dashboarding tools (Tableau, Looker, etc.)

## 💡 Example Insights

The framework can identify patterns like:

```
HIGH-RISK SEGMENTS (Failure Rate > 35%)
────────────────────────────────────────────
Dimension          Category           Failure Rate    Sample Size
file_type         PDF                      42.3%           234
task_type         Financial Analysis       38.7%           198
finance_domain    Investment Banking       36.9%           167
```

```
ROOT CAUSE HYPOTHESES
─────────────────────────────────────────────────────
[1] File Type Analysis
    Evidence: Chi-square p-value: 0.0001 (highly significant)
    Worst Performer: PDF files (42.3% failure rate)
    Recommendation: Improve PDF parsing and extraction pipeline.
                   Consider specialized preprocessing for complex tables.
```

## 🎓 Skills Demonstrated

This project showcases:
- **Statistical Analysis** - Hypothesis testing, correlation, significance testing
- **Python Proficiency** - Pandas, NumPy, SciPy, Matplotlib, Seaborn
- **Data Visualization** - Multi-dimensional charts, heatmaps, dashboards
- **AI/ML Evaluation** - LLM quality metrics, failure analysis, benchmarking
- **Domain Knowledge** - Finance sector applications, task evaluation frameworks
- **Communication** - Clear documentation, stakeholder-ready visualizations

## 📈 Performance

- Processes 1000+ evaluations in < 2 seconds
- Generates comprehensive dashboard in < 5 seconds
- Memory efficient for large datasets (tested up to 100K records)

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional statistical tests (ANOVA, regression analysis)
- Interactive dashboards (Plotly, Streamlit)
- Real-time monitoring capabilities
- Integration with ML experiment tracking (MLflow, Weights & Biases)

## 📝 License

MIT License - see LICENSE file for details

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

Built as a demonstration of statistical analysis and AI evaluation methodologies for production ML systems.

---

**Note**: This framework uses synthetic data for demonstration. For production use, integrate your actual AI evaluation data following the schema in `data_generator.py`.
