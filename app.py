"""
Interactive Chemical Reaction Optimizer Dashboard
==================================================
This interactive web application demonstrates how Bayesian optimization can
revolutionize chemical R&D by intelligently finding optimal reaction conditions
(temperature, pressure, catalyst, time) with 90% fewer experiments than traditional approaches.
Instead of running thousands of trial-and-error experiments over months,
the algorithm uses machine learning to smart-search the parameter space in just 30-100 experiments,
reducing a 3-12 month process to 1-4 weeks and saving $475K-$950K in lab costs per optimization project.

The demo directly showcases Anaconda's Platform value for chemical organizations by proving
that complex AI-enabled workflows can be deployed consistently across global R&D sites
using reproducible conda environments. Every site gets the exact same verified packages
(scikit-optimize, NumPy, Plotly, RDKit) with no "works on my machine" issues,
built-in supply chain security, and Intel MKL-accelerated performance.
The dashboard provides both automated optimization and manual experimentation modes
with rich 3D visualizations, making it perfect for discovery calls, technical validation,
and ROI discussions.
"""

import panel as pn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.reaction_simulator import ReactionSimulator
from src.optimizer import ReactionOptimizer

# Initialize Panel with required extensions
pn.extension('plotly', 'tabulator', sizing_mode="stretch_width")


class ReactionOptimizerApp:
    """
    Interactive dashboard for chemical reaction optimization.

    Demonstrates Bayesian optimization for finding optimal reaction conditions
    with 90% fewer experiments than traditional methods.
    """

    def __init__(self):
        """Initialize the application"""
        self.simulator = ReactionSimulator(random_seed=42)
        self.optimizer = None
        self.optimization_complete = False

        # Create all widgets
        self._create_widgets()

        # Create the main layout
        self.layout = self._create_layout()

    def _create_widgets(self):
        """Create all interactive widgets for the dashboard"""

        # ============================================================
        # OPTIMIZATION SETTINGS WIDGETS
        # ============================================================

        self.n_calls_slider = pn.widgets.IntSlider(
            name='Number of Evaluations',
            start=20,
            end=100,
            step=10,
            value=30,
            width=280
        )

        self.metric_select = pn.widgets.Select(
            name='Optimization Metric',
            options=['yield', 'roi', 'selectivity'],
            value='yield',
            width=280
        )

        self.optimize_button = pn.widgets.Button(
            name=' Run Optimization',
            button_type='primary',
            width=280,
            height=50
        )
        self.optimize_button.on_click(self._run_optimization)

        # ============================================================
        # MANUAL EXPERIMENT WIDGETS
        # ============================================================

        self.temp_slider = pn.widgets.FloatSlider(
            name='Temperature (°C)',
            start=80,
            end=180,
            step=5,
            value=130,
            width=280
        )

        self.pressure_slider = pn.widgets.FloatSlider(
            name='Pressure (Bar)',
            start=1.0,
            end=10.0,
            step=0.5,
            value=5.0,
            width=280
        )

        self.catalyst_slider = pn.widgets.FloatSlider(
            name='Catalyst (mol%)',
            start=0.1,
            end=5.0,
            step=0.1,
            value=2.0,
            width=280
        )

        self.time_slider = pn.widgets.FloatSlider(
            name='Reaction Time (hours)',
            start=0.5,
            end=24.0,
            step=0.5,
            value=8.0,
            width=280
        )

        self.run_experiment_button = pn.widgets.Button(
            name=' Run Experiment',
            button_type='success',
            width=280,
            height=50
        )
        self.run_experiment_button.on_click(self._run_single_experiment)

        # ============================================================
        # DISPLAY WIDGETS
        # ============================================================

        self.results_pane = pn.pane.Markdown(
            """
##  Welcome to the Reaction Optimizer!

### Quick Start Guide:

**Option 1: Manual Experiments**
- Use the sliders to set reaction conditions
- Click "Run Experiment" to see results
- Great for understanding parameter effects

**Option 2: Bayesian Optimization**
- Set number of evaluations (30 is good for demo)
- Click "Run Optimization"
- Watch the algorithm find optimal conditions automatically

---

 **For Chemical Organizations**: This demonstrates how Bayesian optimization
reduces experimental burden by 90%, saving weeks of lab time and significant costs.
""",
            styles={
                'background-color': '#f8f9fa',
                'padding': '20px',
                'border-radius': '8px',
                'border-left': '5px solid #667eea'
            }
        )

        self.plot_pane = pn.pane.Plotly(height=900)

        self.history_table = pn.widgets.Tabulator(
            pagination='remote',
            page_size=10,
            height=350,
            theme='modern'
        )

    def _run_optimization(self, event):
        """
        Run Bayesian optimization when button is clicked.

        This performs intelligent parameter search using Gaussian Processes
        and Expected Improvement acquisition function.
        """
        # Update UI to show progress
        self.results_pane.object = """
## Optimization Running...

Please wait 30-60 seconds while the algorithm searches for optimal conditions.

**What's Happening:**
- Building Gaussian Process surrogate model
- Evaluating Expected Improvement acquisition function
- Running experiments at most promising locations
- Learning from each result to improve predictions

The algorithm balances **exploration** (testing unknown regions) with
**exploitation** (testing near known good conditions).
"""

        try:
            # Create new optimizer
            self.optimizer = ReactionOptimizer(
                self.simulator,
                optimization_metric=self.metric_select.value
            )

            # Run optimization
            results = self.optimizer.optimize(
                n_calls=self.n_calls_slider.value,
                n_random_starts=min(10, self.n_calls_slider.value // 5),
                verbose=False
            )

            self.optimization_complete = True

            # Update all displays
            self._update_results(results)
            self._update_plots()

        except Exception as e:
            self.results_pane.object = f"""
##  Error During Optimization
```
{str(e)}
```

"""

    def _run_single_experiment(self, event):
        """Run a single experiment with manually set parameters"""
        # Get current slider values
        temp = self.temp_slider.value
        pressure = self.pressure_slider.value
        catalyst = self.catalyst_slider.value
        time = self.time_slider.value

        # Run experiment
        results = self.simulator.run_experiment(
            temperature=temp,
            pressure=pressure,
            catalyst_conc=catalyst,
            reaction_time=time
        )

        # Format results for display
        result_md = f"""
##  Experiment Results

### Input Parameters
| Parameter | Value |
|-----------|-------|
|  Temperature | **{temp:.1f}°C** |
|  Pressure | **{pressure:.1f} Bar** |
|  Catalyst | **{catalyst:.2f} mol%** |
|  Time | **{time:.1f} hours** |

### Performance Metrics
| Metric | Value | Description |
|--------|-------|-------------|
|  **Yield** | **{results['yield']:.2f}%** | Product yield (higher is better) |
|  Selectivity | {results['selectivity']:.2f}% | Reaction selectivity (fewer byproducts) |
|  Cost | {results['cost']:.0f} units | Total process cost |
|  ROI | {results['roi']:.2f} | Return on investment |

---

###  Try This:
- **Increase temperature** to 140-150°C to see optimal yield
- **Adjust pressure** to around 5-6 Bar for best results
- **Compare** different conditions to understand the response surface

**Tip**: Use the Bayesian Optimization tab to find optimal conditions automatically!
"""

        self.results_pane.object = result_md

    def _update_results(self, results):
        """Update the results pane with optimization summary"""
        opt_params = results['optimal_params']
        opt_results = results['optimal_results']
        improvement = results['improvement']

        result_md = f"""
##  Optimization Complete!

###  Optimal Parameters Identified
| Parameter | Optimal Value |
|-----------|---------------|
|  Temperature | **{opt_params['temperature']:.1f}°C** |
|  Pressure | **{opt_params['pressure']:.1f} Bar** |
|  Catalyst Concentration | **{opt_params['catalyst_conc']:.2f} mol%** |
|  Reaction Time | **{opt_params['reaction_time']:.1f} hours** |

###  Performance at Optimum
| Metric | Value |
|--------|-------|
|  **Yield** | **{opt_results['yield']:.2f}%** |
|  Selectivity | {opt_results['selectivity']:.2f}% |
|  Total Cost | {opt_results['cost']:.0f} units |
|  ROI | {opt_results['roi']:.2f} |

###  Optimization Performance
- **Improvement vs Random Search**: {improvement:.1f}%
- **Total Experiments Run**: {results['n_evaluations']}
- **Best {self.metric_select.value.title()} Found**: {results['best_value']:.2f}

---

###  Business Value for Chemical Organizations

This optimization demonstrates:

 **90% Reduction in Experiments** - Only {results['n_evaluations']} experiments instead of 1000s

 **Time Savings** - Days instead of weeks/months for optimization

 **Cost Savings** - Minimized reagent usage and lab time

 **Knowledge Retention** - Reproducible workflow across global sites

 **Strategic Alignment** - Direct support for AI-enabled R&D initiatives

---

 **Scroll down** to see detailed visualizations of the optimization process!
"""

        self.results_pane.object = result_md

    def _update_plots(self):
        """Create comprehensive visualization plots with enhanced readability"""
        if not self.optimization_complete:
            return

        df = self.optimizer.get_history_df()

        # Create multi-panel figure with better spacing
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '<b>Parameter Space Exploration</b>',
                '<b>Convergence: Best Yield Found</b>',
                '<b>Key Parameters Over Time</b>',
                '<b>Cost vs Yield Trade-off</b>'
            ),
            specs=[
                [{'type': 'scatter3d'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )

        # ============================================================
        # 1. Enhanced 3D scatter plot
        # ============================================================
        fig.add_trace(
            go.Scatter3d(
                x=df['temperature'],
                y=df['pressure'],
                z=df['yield'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['yield'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="<b>Yield (%)</b>", font=dict(size=14)),
                        x=0.46,
                        len=0.4,
                        tickfont=dict(size=12)
                    ),
                    line=dict(width=1, color='white'),
                    opacity=0.9
                ),
                name='Experiments',
                hovertemplate='<b>Experiment %{text}</b><br>' +
                             'Temperature: %{x:.1f}°C<br>' +
                             'Pressure: %{y:.1f} Bar<br>' +
                             '<b>Yield: %{z:.1f}%</b><br>' +
                             '<extra></extra>',
                text=[f"{i}" for i in df.index]
            ),
            row=1, col=1
        )

        # Update 3D scene
        fig.update_scenes(
            xaxis=dict(
                title=dict(text='<b>Temperature (°C)</b>', font=dict(size=14)),
                tickfont=dict(size=12),
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title=dict(text='<b>Pressure (Bar)</b>', font=dict(size=14)),
                tickfont=dict(size=12),
                gridcolor='lightgray'
            ),
            zaxis=dict(
                title=dict(text='<b>Yield (%)</b>', font=dict(size=14)),
                tickfont=dict(size=12),
                gridcolor='lightgray'
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            row=1, col=1
        )

        # ============================================================
        # 2. Enhanced convergence plot
        # ============================================================
        df['best_so_far'] = df['yield'].cummax()

        # Shaded area under best line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['best_so_far'],
                fill='tozeroy',
                fillcolor='rgba(46, 204, 113, 0.1)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=2
        )

        # Best yield line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['best_so_far'],
                mode='lines+markers',
                name='Best Yield',
                line=dict(color='#27ae60', width=4),
                marker=dict(size=8, symbol='star'),
                hovertemplate='<b>Iteration %{x}</b><br>' +
                             'Best Yield: <b>%{y:.2f}%</b><br>' +
                             '<extra></extra>'
            ),
            row=1, col=2
        )

        # Individual experiments
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['yield'],
                mode='markers',
                name='Experiments',
                marker=dict(
                    color='#3498db',
                    size=6,
                    opacity=0.6,
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>Experiment %{x}</b><br>' +
                             'Yield: %{y:.2f}%<br>' +
                             '<extra></extra>'
            ),
            row=1, col=2
        )

        # ============================================================
        # 3. Simplified parameter evolution (just temperature & pressure)
        # ============================================================
        # Temperature (primary parameter)
        temp_norm = (df['temperature'] - df['temperature'].min()) / (df['temperature'].max() - df['temperature'].min())
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=temp_norm,
                mode='lines+markers',
                name='Temperature',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=6),
                hovertemplate='<b>Temperature</b><br>' +
                             'Iteration: %{x}<br>' +
                             'Actual: ' + df['temperature'].round(1).astype(str) + '°C<br>' +
                             '<extra></extra>'
            ),
            row=2, col=1
        )

        # Pressure (secondary parameter)
        press_norm = (df['pressure'] - df['pressure'].min()) / (df['pressure'].max() - df['pressure'].min())
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=press_norm,
                mode='lines+markers',
                name='Pressure',
                line=dict(color='#3498db', width=3),
                marker=dict(size=6),
                hovertemplate='<b>Pressure</b><br>' +
                             'Iteration: %{x}<br>' +
                             'Actual: ' + df['pressure'].round(1).astype(str) + ' Bar<br>' +
                             '<extra></extra>'
            ),
            row=2, col=1
        )

        # ============================================================
        # 4. Enhanced Cost vs Yield scatter
        # ============================================================
        fig.add_trace(
            go.Scatter(
                x=df['cost'],
                y=df['yield'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=df.index,
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="<b>Iteration</b>", font=dict(size=14)),
                        x=1.15,
                        len=0.4,
                        tickfont=dict(size=12)
                    ),
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                name='Experiments',
                hovertemplate='<b>Experiment %{text}</b><br>' +
                             'Cost: %{x:.0f} units<br>' +
                             '<b>Yield: %{y:.1f}%</b><br>' +
                             '<extra></extra>',
                text=[f"{i}" for i in df.index],
                showlegend=False
            ),
            row=2, col=2
        )

        # Highlight best point
        best_idx = df['yield'].idxmax()
        fig.add_trace(
            go.Scatter(
                x=[df.loc[best_idx, 'cost']],
                y=[df.loc[best_idx, 'yield']],
                mode='markers',
                marker=dict(
                    size=20,
                    color='#f39c12',
                    symbol='star',
                    line=dict(width=3, color='white')
                ),
                name='Best Result',
                hovertemplate='<b> BEST RESULT</b><br>' +
                             'Cost: %{x:.0f}<br>' +
                             'Yield: %{y:.2f}%<br>' +
                             '<extra></extra>',
                showlegend=True
            ),
            row=2, col=2
        )

        # ============================================================
        # Update overall layout
        # ============================================================
        fig.update_layout(
            height=900,
            showlegend=True,
            title=dict(
                text="<b>Optimization Analysis Dashboard</b>",
                font=dict(size=24, color='#2c3e50'),
                x=0.5,
                xanchor='center'
            ),
            hovermode='closest',
            legend=dict(
                font=dict(size=12),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#bdc3c7',
                borderwidth=2
            ),
            plot_bgcolor='#ecf0f1',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12)
        )

        # Update all subplot titles
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=16, color='#2c3e50')

        # Update x and y axes with better styling
        fig.update_xaxes(
            title_text="<b>Evaluation Number</b>",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            gridcolor='white',
            showline=True,
            linewidth=2,
            linecolor='#bdc3c7',
            row=1, col=2
        )
        fig.update_yaxes(
            title_text="<b>Yield (%)</b>",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            gridcolor='white',
            showline=True,
            linewidth=2,
            linecolor='#bdc3c7',
            row=1, col=2
        )

        fig.update_xaxes(
            title_text="<b>Evaluation Number</b>",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            gridcolor='white',
            showline=True,
            linewidth=2,
            linecolor='#bdc3c7',
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="<b>Normalized Value (0-1)</b>",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            gridcolor='white',
            showline=True,
            linewidth=2,
            linecolor='#bdc3c7',
            row=2, col=1
        )

        fig.update_xaxes(
            title_text="<b>Total Cost (units)</b>",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            gridcolor='white',
            showline=True,
            linewidth=2,
            linecolor='#bdc3c7',
            row=2, col=2
        )
        fig.update_yaxes(
            title_text="<b>Yield (%)</b>",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            gridcolor='white',
            showline=True,
            linewidth=2,
            linecolor='#bdc3c7',
            row=2, col=2
        )

        self.plot_pane.object = fig

        # ============================================================
        # Update history table with better formatting
        # ============================================================
        display_df = df[['iteration', 'temperature', 'pressure', 'catalyst_conc',
                         'reaction_time', 'yield', 'cost', 'roi']].round(2)
        display_df.columns = ['#', 'Temp (°C)', 'Press (Bar)', 'Catalyst (mol%)',
                             'Time (hr)', 'Yield (%)', 'Cost', 'ROI']

       # Add star marker for best result
        display_df['Best'] = ['⭐' if i == df['yield'].idxmax() else ''
                              for i in df.index]

        self.history_table.value = display_df

    def _create_layout(self):
        """Create the complete dashboard layout"""

        # ============================================================
        # HEADER
        # ============================================================
        header = pn.pane.Markdown(
            """
#  Chemical Reaction Optimizer
## Bayesian Optimization Demo for Chemical Organizations
### Powered by Anaconda Platform

Demonstrating intelligent parameter search for chemical reactions using
**Gaussian Process** surrogate models and **Expected Improvement** acquisition.
""",
            styles={
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'color': 'white',
                'padding': '30px',
                'border-radius': '10px',
                'text-align': 'center',
                'margin-bottom': '20px'
            }
        )

        # ============================================================
        # VALUE PROPOSITION BANNER
        # ============================================================
        value_banner = pn.pane.Markdown(
            """
### Why Anaconda Adds Value to Chemical Organizations

| Feature | Benefit |
|---------|---------|
|  **Package Management** | One-command environment reproduction across the globe |
|  **Supply Chain Security** | Verified packages (RDKit, scikit-optimize, scipy) from trusted sources |
|  **Global Consistency** | Identical results across all global sites - no "works on my machine" issues |
|  **Optimized Performance** | Accelerated Python workflows for faster computations |
|  **Governance & Compliance** | Control package access, audit dependencies, ensure reproducibility |
""",
            styles={
                'background-color': '#f0f8ff',
                'padding': '20px',
                'border-radius': '8px',
                'border-left': '5px solid #667eea',
                'margin-bottom': '20px'
            }
        )

        # ============================================================
        # TAB 1: BAYESIAN OPTIMIZATION
        # ============================================================
        opt_tab = pn.Column(
            pn.pane.Markdown(
                "## Configuration",
                styles={'color': '#667eea', 'font-size': '1.3em'}
            ),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("### Optimization Settings"),
                    self.n_calls_slider,
                    self.metric_select,
                    pn.pane.Markdown("---"),
                    self.optimize_button,
                    pn.pane.Markdown(
                        """
**Tips:**
- Start with 30 evaluations for quick demo
- Use 50+ for better optimization
- Larger values = more accurate but slower
""",
                        styles={'font-size': '0.9em', 'color': '#7f8c8d'}
                    ),
                    styles={
                        'background-color': '#f8f9fa',
                        'padding': '20px',
                        'border-radius': '8px',
                        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
                    },
                    width=340
                ),
                pn.Column(
                    self.results_pane,
                    width=700
                )
            ),
            pn.pane.Markdown(
                "##  Optimization Visualizations",
                styles={'color': '#667eea', 'font-size': '1.3em', 'margin-top': '30px'}
            ),
            self.plot_pane,
            pn.pane.Markdown(
                "##  Experiment History",
                styles={'color': '#667eea', 'font-size': '1.3em', 'margin-top': '20px'}
            ),
            pn.pane.Markdown(
                "Complete log of all experiments. The ⭐ indicates the best result found."
            ),
            self.history_table
        )

        # ============================================================
        # TAB 2: MANUAL EXPERIMENTS
        # ============================================================
        manual_tab = pn.Column(
            pn.pane.Markdown(
                """
##  Manual Experimentation

Test specific reaction conditions to understand how parameters affect outcomes.

**Perfect for:**
-  Testing hypotheses about parameter effects
-  Validating optimization results
-  Educational demonstrations
-  Exploring the response surface
""",
                styles={
                    'background-color': '#e8f5e9',
                    'padding': '15px',
                    'border-radius': '5px',
                    'border-left': '5px solid #4caf50'
                }
            ),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("### Reaction Parameters"),
                    self.temp_slider,
                    self.pressure_slider,
                    self.catalyst_slider,
                    self.time_slider,
                    pn.pane.Markdown("---"),
                    self.run_experiment_button,
                    pn.pane.Markdown(
                        """
**Optimal Ranges:**
- Temperature: 135-145°C
- Pressure: 5-6 Bar
- Catalyst: 1.8-2.2 mol%
- Time: 7-9 hours
""",
                        styles={'font-size': '0.9em', 'color': '#7f8c8d', 'margin-top': '10px'}
                    ),
                    styles={
                        'background-color': '#f8f9fa',
                        'padding': '20px',
                        'border-radius': '8px',
                        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
                    },
                    width=340
                ),
                pn.Column(
                    self.results_pane,
                    width=700
                )
            )
        )

        # ============================================================
        # TAB 3: ABOUT & DOCUMENTATION
        # ============================================================
        about_tab = pn.pane.Markdown(
            """
# About This Demo

##  Purpose

This interactive application demonstrates **Anaconda's Platform value** for
Chemical Organization's AI-enabled R&D initiatives, specifically addressing the stated use case:

> *"Bayesian algorithms for reaction optimization that rapidly determine optimal
> conditions from thousands of experimental combinations"*

---

##  Technical Implementation

### Algorithm: Bayesian Optimization

- **Surrogate Model**: Gaussian Process with RBF kernel
- **Acquisition Function**: Expected Improvement (EI)
- **Optimization**: Sequential model-based optimization
- **Balance**: Automatic exploration-exploitation trade-off

### Chemical Model

The simulator uses realistic chemical engineering principles:
- **Arrhenius Temperature Effect**: `exp(-(T-T_opt)²/σ²)`
- **Pressure Equilibrium**: Linear relationship with optimal point
- **Catalyst Kinetics**: Michaelis-Menten with diminishing returns
- **Time Dynamics**: Conversion vs degradation trade-off
- **Experimental Noise**: Gaussian, 5% standard deviation

---

## Business Value Quantified

| Metric | Traditional | With Bayesian | Savings |
|--------|-------------|---------------|---------|
| **Experiments** | 1,000-10,000 | 50-100 | **90-99%** |
| **Time** | 3-12 months | 1-4 weeks | **85-95%** |
| **Lab Costs** | $500K-$1M | $25K-$50K | **$475K-$950K** |
| **Reagent Usage** | High | Minimal | **90%+** |

### Strategic Impact

 **Faster Time-to-Market**: New formulations in weeks, not months

 **Competitive Advantage**: First-mover in emerging markets

 **Knowledge Retention**: Reproducible workflows preserve institutional IP

 **Global Consistency**: Same results in across the globe

 **Scalable Approach**: Apply to MDI, TDI, polyurethane, catalyst design

---

##  Technology Stack
```python
# Core Scientific Computing
Python 3.11              # Modern, performant
NumPy 1.26              # Optimized linear algebra (Intel MKL)
SciPy 1.11              # Scientific algorithms
Pandas 2.1              # Data manipulation

# Machine Learning
scikit-learn 1.3        # ML framework
scikit-optimize 0.9     # Bayesian optimization

# Chemistry
RDKit 2023.09           # Cheminformatics (available for future use)

# Visualization & UI
Plotly 5.18             # Interactive 3D plots
Panel 1.3               # Web dashboards
Matplotlib 3.8          # Static plots
```

**All managed through Anaconda's Core package management system.**

---

##  Alignment with Chemical Organizations's Goals

| Chemical Organizatons Needs | This Demo Addresses |
|------------------|---------------------|
| "AI-enabled R&D" |  ML-powered optimization |
| "Molecular discovery and reaction optimization" |  Intelligent parameter search |
| "Digital transformation" |  Interactive dashboards |
| "Global collaboration" |  Reproducible environments |
| "Intelligent digitalization" |  Automated workflows |


---

##  Contact

**For questions about:**
- **This Demo**: Saundra Monroe
- **Anaconda Products**: https://www.anaconda.com/contact
- **Technical Support**: support@anaconda.com

---

**Built with the power of Anaconda Distribution**

© 2025 Anaconda, Inc. All Rights Reserved.
""",
            styles={'padding': '30px', 'line-height': '1.6'}
        )

        # ============================================================
        # CREATE TABS
        # ============================================================
        tabs = pn.Tabs(
            (' Bayesian Optimization', opt_tab),
            (' Manual Experiments', manual_tab),
            (' About', about_tab),
            dynamic=True,
            tabs_location='above'
        )

        # ============================================================
        # ASSEMBLE COMPLETE LAYOUT
        # ============================================================
        return pn.Column(
            header,
            value_banner,
            tabs,
            sizing_mode='stretch_width'
        )

    def servable(self):
        """Return the servable layout for Panel"""
        return self.layout


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # When run as a script
    app = ReactionOptimizerApp()
    app.servable().servable()

elif __name__.startswith("bokeh"):
    # When served with panel serve
    app = ReactionOptimizerApp()
    app.servable().servable()
