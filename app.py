"""
Interactive web interface for reaction optimization.
Built with Panel for easy deployment.
"""

import panel as pn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.reaction_simulator import ReactionSimulator
from src.optimizer import ReactionOptimizer

# Initialize Panel
pn.extension('plotly', sizing_mode="stretch_width", notifications=True)

class ReactionOptimizerApp:
    """Interactive dashboard for reaction optimization"""

    def __init__(self):
        self.simulator = ReactionSimulator(random_seed=42)
        self.optimizer = None
        self.optimization_complete = False

        # Create widgets
        self._create_widgets()

        # Create layout
        self.layout = self._create_layout()

    def _create_widgets(self):
        """Create interactive widgets"""

        # Optimization settings
        self.n_calls_slider = pn.widgets.IntSlider(
            name='Number of Evaluations',
            start=20, end=100, step=10, value=50,
            width=280
        )

        self.metric_select = pn.widgets.Select(
            name='Optimization Metric',
            options=['yield', 'roi', 'selectivity'],
            value='yield',
            width=280
        )

        self.optimize_button = pn.widgets.Button(
            name='🚀 Run Optimization',
            button_type='primary',
            width=280,
            height=50
        )
        self.optimize_button.on_click(self._run_optimization)

        # Manual experiment controls
        self.temp_slider = pn.widgets.FloatSlider(
            name='Temperature (°C)',
            start=80, end=180, step=5, value=130,
            width=280
        )

        self.pressure_slider = pn.widgets.FloatSlider(
            name='Pressure (Bar)',
            start=1.0, end=10.0, step=0.5, value=5.0,
            width=280
        )

        self.catalyst_slider = pn.widgets.FloatSlider(
            name='Catalyst (mol%)',
            start=0.1, end=5.0, step=0.1, value=2.0,
            width=280
        )

        self.time_slider = pn.widgets.FloatSlider(
            name='Reaction Time (hrs)',
            start=0.5, end=24.0, step=0.5, value=8.0,
            width=280
        )

        self.run_experiment_button = pn.widgets.Button(
            name='🧪 Run Single Experiment',
            button_type='success',
            width=280,
            height=50
        )
        self.run_experiment_button.on_click(self._run_single_experiment)

        # Results display
        self.results_pane = pn.pane.Markdown(
            "## 👋 Welcome!\n\nChoose 'Bayesian Optimization' tab to run automated optimization,\nor 'Manual Experiments' to test individual conditions.",
            styles={'background-color': '#f8f9fa', 'padding': '20px', 'border-radius': '5px'}
        )
        self.plot_pane = pn.pane.Plotly(height=700)
        self.history_table = pn.widgets.Tabulator(
            pagination='remote',
            page_size=10,
            height=400
        )

    def _run_optimization(self, event):
        """Run Bayesian optimization"""
        self.results_pane.object = "## 🔄 Optimization running...\n\nThis may take 30-60 seconds depending on the number of evaluations."
        pn.io.notifications.info("Starting optimization...", duration=3000)

        # Create optimizer
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

        # Update results
        self._update_results(results)
        self._update_plots()

        pn.io.notifications.success("Optimization complete!", duration=3000)

    def _run_single_experiment(self, event):
        """Run a single experiment with manual parameters"""
        results = self.simulator.run_experiment(
            temperature=self.temp_slider.value,
            pressure=self.pressure_slider.value,
            catalyst_conc=self.catalyst_slider.value,
            reaction_time=self.time_slider.value
        )

        result_md = f"""
        ## 🧪 Single Experiment Results

        ### Parameters:
        | Parameter | Value |
        |-----------|-------|
        | Temperature | {self.temp_slider.value:.1f} °C |
        | Pressure | {self.pressure_slider.value:.1f} Bar |
        | Catalyst | {self.catalyst_slider.value:.2f} mol% |
        | Time | {self.time_slider.value:.1f} hrs |

        ### Results:
        | Metric | Value |
        |--------|-------|
        | ✅ **Yield** | **{results['yield']:.2f}%** |
        | 🎯 Selectivity | {results['selectivity']:.2f}% |
        | 💰 Cost | {results['cost']:.0f} units |
        | 📈 ROI | {results['roi']:.2f} |

        ---

        💡 **Tip**: Try adjusting parameters to see how they affect the yield!
        """

        self.results_pane.object = result_md
        pn.io.notifications.success("Experiment complete!", duration=2000)

    def _update_results(self, results):
        """Update results display"""
        opt_params = results['optimal_params']
        opt_results = results['optimal_results']
        improvement = results['improvement']

        result_md = f"""
        ## 🎉 Optimization Complete!

        ### Optimal Parameters Found:
        | Parameter | Value |
        |-----------|-------|
        | 🌡️ Temperature | **{opt_params['temperature']:.1f} °C** |
        | ⚡ Pressure | **{opt_params['pressure']:.1f} Bar** |
        | ⚗️ Catalyst | **{opt_params['catalyst_conc']:.2f} mol%** |
        | ⏱️ Time | **{opt_params['reaction_time']:.1f} hrs** |

        ### Performance Metrics:
        | Metric | Value |
        |--------|-------|
        | ✅ **Yield** | **{opt_results['yield']:.2f}%** |
        | 🎯 Selectivity | {opt_results['selectivity']:.2f}% |
        | 💰 Cost | {opt_results['cost']:.0f} units |
        | 📈 ROI | {opt_results['roi']:.2f} |

        ### Improvement:
        **{improvement:.1f}%** improvement vs random search

        ---

        💡 **Value Proposition**: This optimization reduced experimental cycles from
        potentially **thousands** to just **{results['n_evaluations']} evaluations**, saving
        **weeks of lab time** and **significant resources**.

        🔍 Check out the visualizations below to see how the algorithm explored the parameter space!
        """

        self.results_pane.object = result_md

    def _update_plots(self):
        """Update visualization plots"""
        if not self.optimization_complete:
            return

        df = self.optimizer.get_history_df()

        # Create comprehensive visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '3D: Yield vs Temperature & Pressure',
                'Optimization Progress Over Time',
                'Parameter Evolution',
                'Cost vs Yield Trade-off'
            ),
            specs=[
                [{'type': 'scatter3d'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # 1. 3D scatter of temp, pressure, yield
        fig.add_trace(
            go.Scatter3d(
                x=df['temperature'],
                y=df['pressure'],
                z=df['yield'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=df['yield'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Yield (%)",
                        x=0.46
                    ),
                    line=dict(width=0.5, color='white')
                ),
                text=[f"Exp {i}<br>Yield: {y:.1f}%" for i, y in enumerate(df['yield'])],
                hovertemplate="<b>Experiment %{text}</b><br>" +
                              "Temp: %{x:.1f}°C<br>" +
                              "Pressure: %{y:.1f} Bar<br>" +
                              "Yield: %{z:.1f}%<br>" +
                              "<extra></extra>",
                name='Experiments'
            ),
            row=1, col=1
        )

        # 2. Optimization progress
        df['best_yield_so_far'] = df['yield'].cummax()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['best_yield_so_far'],
                mode='lines+markers',
                name='Best Yield',
                line=dict(color='#2ecc71', width=3),
                marker=dict(size=6),
                fill='tonexty',
                fillcolor='rgba(46, 204, 113, 0.1)'
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['yield'],
                mode='markers',
                name='Individual Experiments',
                marker=dict(color='#95a5a6', size=4, opacity=0.5)
            ),
            row=1, col=2
        )

        # 3. Parameter evolution (normalized)
        param_colors = {
            'temperature': '#e74c3c',
            'pressure': '#3498db',
            'catalyst_conc': '#9b59b6',
            'reaction_time': '#f39c12'
        }

        for param, color in param_colors.items():
            # Normalize to 0-1 scale for comparison
            normalized = (df[param] - df[param].min()) / (df[param].max() - df[param].min())
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=normalized,
                    mode='lines',
                    name=param.replace('_', ' ').title(),
                    line=dict(color=color, width=2),
                    opacity=0.7
                ),
                row=2, col=1
            )

        # 4. Cost vs Yield (Pareto frontier concept)
        fig.add_trace(
            go.Scatter(
                x=df['cost'],
                y=df['yield'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df.index,
                    colorscale='Plasma',
                    showscale=False,
                    line=dict(width=1, color='white')
                ),
                text=[f"Exp {i}" for i in df.index],
                hovertemplate="<b>%{text}</b><br>" +
                              "Cost: %{x:.0f}<br>" +
                              "Yield: %{y:.1f}%<br>" +
                              "<extra></extra>",
                name='Experiments'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Reaction Optimization Analysis Dashboard",
            title_font_size=20,
            hovermode='closest'
        )

        # Update axes labels
        fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Pressure (Bar)", row=1, col=1)

        fig.update_xaxes(title_text="Evaluation Number", row=1, col=2)
        fig.update_yaxes(title_text="Yield (%)", row=1, col=2)

        fig.update_xaxes(title_text="Evaluation Number", row=2, col=1)
        fig.update_yaxes(title_text="Normalized Parameter Value", row=2, col=1)

        fig.update_xaxes(title_text="Cost (units)", row=2, col=2)
        fig.update_yaxes(title_text="Yield (%)", row=2, col=2)

        self.plot_pane.object = fig

        # Update history table
        display_df = df[[
            'iteration', 'temperature', 'pressure', 'catalyst_conc',
            'reaction_time', 'yield', 'cost', 'roi'
        ]].round(2)
        self.history_table.value = display_df

    def _create_layout(self):
        """Create the dashboard layout"""

        # Header
        header = pn.pane.Markdown("""
        # 🧪 Chemical Reaction Optimizer
        ## Powered by Anaconda Enterprise Python Stack

        This demo showcases **Bayesian optimization** for chemical reaction conditions,
        directly addressing **Wanhua Chemical's** use case of *"rapidly determining optimal
        conditions from thousands of experimental combinations."*
        """, styles={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                     'color': 'white', 'padding': '30px', 'border-radius': '10px',
                     'margin-bottom': '20px'})

        # Value propositions
        value_props = pn.pane.Markdown("""
        ### 🚀 Why Anaconda Adds Value:

        | Feature | Benefit |
        |---------|---------|
        | 📦 **Package Management** | One-click environment reproduction across Yantai, Ningbo, Hungary |
        | 🔒 **Supply Chain Security** | Verified scientific packages (RDKit, scikit-optimize, scipy) |
        | 🤝 **Collaboration** | Share notebooks & environments globally with confidence |
        | ⚡ **Performance** | Optimized builds of NumPy, SciPy for faster computation |
        | 🎯 **Governance** | Control which packages your R&D teams can access |
        """, styles={'background-color': '#f0f8ff', 'padding': '20px',
                     'border-radius': '8px', 'border-left': '5px solid #667eea'})

        # Create tabs
        opt_tab = pn.Column(
            pn.pane.Markdown("## ⚙️ Configuration", styles={'color': '#667eea'}),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("### Optimization Settings"),
                    self.n_calls_slider,
                    self.metric_select,
                    self.optimize_button,
                    styles={'background-color': '#f8f9fa', 'padding': '20px',
                            'border-radius': '8px'},
                    width=320
                ),
                pn.Column(
                    self.results_pane,
                    width=600
                )
            ),
            pn.pane.Markdown("## 📊 Visualization", styles={'color': '#667eea', 'margin-top': '20px'}),
            self.plot_pane,
            pn.pane.Markdown("## 📋 Experiment History", styles={'color': '#667eea'}),
            self.history_table
        )

        manual_tab = pn.Column(
            pn.pane.Markdown("""
            ## 🔬 Manual Experimentation

            Use the sliders below to set reaction conditions and run individual experiments.
            This is useful for:
            - Testing specific hypotheses
            - Validating optimization results
            - Educational demonstrations
            """),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("### Experiment Parameters"),
                    self.temp_slider,
                    self.pressure_slider,
                    self.catalyst_slider,
                    self.time_slider,
                    self.run_experiment_button,
                    styles={'background-color': '#f8f9fa', 'padding': '20px',
                            'border-radius': '8px'},
                    width=320
                ),
                pn.Column(
                    self.results_pane,
                    width=600
                )
            )
        )

        about_tab = pn.pane.Markdown("""
        ## About This Demo

        This interactive application demonstrates how **Anaconda** enables enterprise-scale
        scientific computing for chemical R&D.

        ### 🎯 Technical Implementation

        - **Bayesian Optimization**: Using Gaussian Processes to efficiently search parameter space
        - **Chemical Simulation**: Realistic reaction modeling with temperature, pressure, catalyst effects
        - **Interactive Visualization**: Real-time 3D plotting and progress tracking
        - **Reproducible Environments**: Single YAML file ensures consistency across global teams

        ### 💰 Business Value for Wanhua Chemical

        | Metric | Value |
        |--------|-------|
        | **Experimental Reduction** | 90%+ fewer physical experiments needed |
        | **Time Savings** | Weeks → Days for reaction optimization |
        | **Cost Savings** | Minimize expensive lab time and materials |
        | **Global Consistency** | Same environment in Yantai, Ningbo, Hungary, Europe |
        | **Knowledge Retention** | Reproducible workflows preserve institutional IP |

        ### 🛠️ Technologies Used
```python
        # Core scientific stack
        - Python 3.11
        - NumPy 1.26 (optimized linear algebra)
        - SciPy 1.11 (scientific computing)
        - Pandas 2.1 (data manipulation)

        # Machine learning & optimization
        - scikit-learn 1.3
        - scikit-optimize 0.9 (Bayesian optimization)

        # Chemistry-specific
        - RDKit 2023.09 (cheminformatics)

        # Visualization & UI
        - Plotly 5.18 (interactive plots)
        - Panel 1.3 (dashboards)
        - Matplotlib 3.8
```

        ### 📚 Mapping to Wanhua's Stated Needs

        > "Wanhua is actively exploring and promoting the application of AI and big data
        > algorithms in its processes, especially in **molecular discovery and reaction optimization**"

        This demo directly addresses:

        1. ✅ **Reaction Optimization**: Bayesian methods for parameter tuning
        2. ✅ **Reduced Experimental Cycles**: From 1000s to 50 evaluations
        3. ✅ **AI-Enabled R&D**: Machine learning for chemical processes
        4. ✅ **Global Collaboration**: Reproducible environments across sites
        5. ✅ **Supply Chain Security**: Verified package sources

        ### 🚀 Next Steps

        1. **Pilot Project**: Apply to real MDI reaction data from Wanhua's labs
        2. **Integration**: Connect to existing Huawei Cloud ModelArts platform
        3. **Scaling**: Deploy Anaconda Enterprise for team-wide access
        4. **Training**: Enable R&D scientists with best practices
        5. **Governance**: Establish package approval workflows

        ---

        **Built with ❤️ using Anaconda Distribution**

        For more information: [https://www.anaconda.com](https://www.anaconda.com)
        """, styles={'padding': '20px'})

        tabs = pn.Tabs(
            ('🎯 Bayesian Optimization', opt_tab),
            ('🔬 Manual Experiments', manual_tab),
            ('📖 About', about_tab),
            dynamic=True
        )

        return pn.Column(
            header,
            value_props,
            tabs,
            sizing_mode='stretch_width'
        )

    def servable(self):
        """Return servable layout"""
        return self.layout

# Create and serve app
if __name__ == "__main__":
    app = ReactionOptimizerApp()
    app.servable().servable()
elif __name__.startswith("bokeh"):
    # When served with panel serve
    app = ReactionOptimizerApp()
    app.servable().servable()

