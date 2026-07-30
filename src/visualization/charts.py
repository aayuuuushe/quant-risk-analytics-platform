import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import norm

LAYOUT = dict(
    paper_bgcolor='#1E2130',
    plot_bgcolor='#1E2130',
    font=dict(color='#8B9DC3', family='monospace', size=11),
    title_font=dict(color='#FFFFFF', size=13),
    xaxis=dict(
        gridcolor='#2A2D3E',
        zerolinecolor='#2A2D3E',
        linecolor='#2A2D3E',
        tickfont=dict(color='#8B9DC3')
    ),
    yaxis=dict(
        gridcolor='#2A2D3E',
        zerolinecolor='#2A2D3E',
        linecolor='#2A2D3E',
        tickfont=dict(color='#8B9DC3')
    ),
    legend=dict(
        bgcolor='#1A1D27',
        bordercolor='#2A2D3E',
        borderwidth=1,
        font=dict(color='#8B9DC3', size=10)
    ),
    hoverlabel=dict(
        bgcolor='#1A1D27',
        bordercolor='#00A3FF',
        font=dict(color='#FFFFFF')
    ),
    margin=dict(l=50, r=20, t=50, b=40)
)

COLORS = ['#00A3FF', '#00D4AA', '#FFB347', '#C084FC', '#FF6B6B']


def _fig(title=''):
    layout = LAYOUT.copy()
    if title:
        layout['title'] = title
    fig = go.Figure()
    fig.update_layout(**layout)
    return fig


def cumulative_performance_chart(log_returns: pd.DataFrame):
    cumulative = np.exp(log_returns.cumsum())
    fig = _fig('Growth of $1 Invested')
    for i, ticker in enumerate(log_returns.columns):
        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative[ticker],
            name=ticker,
            line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            hovertemplate=f'<b>{ticker}</b><br>%{{x|%Y-%m-%d}}'
                          f'<br>$%{{y:.2f}}<extra></extra>'
        ))
    fig.add_hline(y=1, line_dash='dash',
                  line_color='#2A2D3E', line_width=1)
    return fig


def return_distribution_chart(portfolio_returns: pd.Series,
                               var_95: float,
                               var_99: float):
    fig = _fig('Return Distribution with VaR Thresholds')
    fig.add_trace(go.Histogram(
        x=portfolio_returns,
        nbinsx=80,
        histnorm='probability density',
        name='Daily Returns',
        marker_color='#00A3FF',
        opacity=0.6
    ))
    tail = portfolio_returns[portfolio_returns <= -var_99]
    if len(tail) > 0:
        fig.add_trace(go.Histogram(
            x=tail, nbinsx=20,
            histnorm='probability density',
            name='99% Tail',
            marker_color='#FF4444',
            opacity=0.5
        ))
    fig.add_vline(x=-var_95, line_color='#FFB347',
                  line_dash='dash', line_width=2,
                  annotation_text='95% VaR',
                  annotation_font_color='#FFB347')
    fig.add_vline(x=-var_99, line_color='#FF4444',
                  line_dash='dash', line_width=2,
                  annotation_text='99% VaR',
                  annotation_font_color='#FF4444')
    return fig


def parametric_overlay_chart(portfolio_returns: pd.Series,
                              mu_p: float, sigma_p: float,
                              p_var_95: float, p_var_99: float,
                              h_var_95: float, h_var_99: float):
    fig = _fig('Actual Returns vs Fitted Normal')
    fig.add_trace(go.Histogram(
        x=portfolio_returns, nbinsx=80,
        histnorm='probability density',
        name='Actual Returns',
        marker_color='#00A3FF', opacity=0.5
    ))
    x = np.linspace(
        portfolio_returns.mean() - 4 * portfolio_returns.std(),
        portfolio_returns.mean() + 4 * portfolio_returns.std(),
        300
    )
    fig.add_trace(go.Scatter(
        x=x, y=norm.pdf(x, mu_p, sigma_p),
        name='Fitted Normal',
        line=dict(color='#00D4AA', width=2)
    ))

    fig.add_vline(x=-p_var_95, line_color='#FFB347', line_dash='dash', line_width=2,
              annotation_text='95% Param', annotation_position='top left')
    fig.add_vline(x=-p_var_99, line_color='#FF4444', line_dash='dash', line_width=2,
              annotation_text='99% Param', annotation_position='top right')
    fig.add_vline(x=-h_var_95, line_color='#FFB347', line_dash='dot', line_width=2,
              annotation_text='95% Hist', annotation_position='bottom left')
    fig.add_vline(x=-h_var_99, line_color='#FF4444', line_dash='dot', line_width=2,
              annotation_text='99% Hist', annotation_position='bottom right')
    return fig


def var_comparison_chart(methods: list,
                         var_95: list,
                         var_99: list):

    fig = _fig('VaR Comparison — All Methods (%)')
    fig.add_trace(
        go.Bar(
            name='95% VaR',
            x=methods,
            y=var_95,
            marker_color='#6E8FAF'     ))
    fig.add_trace(
        go.Bar(
            name='99% VaR',
            x=methods,
            y=var_99,
            marker_color='#4F6D8C'   ))
    fig.update_layout(
        barmode='group',
        bargap=0.28,
        bargroupgap=0.10)
    return fig


def ewma_volatility_chart(dates, ewma_vol: np.ndarray):
    fig = _fig('EWMA Portfolio Volatility (λ=0.94)')
    fig.add_trace(go.Scatter(
        x=dates, y=ewma_vol * 100,
        name='EWMA Volatility',
        line=dict(color='#00A3FF', width=1.2),
        hovertemplate='%{x|%Y-%m-%d}<br>Vol: %{y:.3f}%<extra></extra>'
    ))
    crises = [
        ('2020-02-01', '2020-06-01', 'COVID', '#FF4444'),
        ('2018-09-01', '2019-01-01', '2018 Selloff', '#FFB347'),
        ('2022-01-01', '2022-10-31', '2022 Rate Shock', '#C084FC'),
    ]
    for start, end, label, color in crises:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color, opacity=0.08, line_width=0,
            annotation_text=label,
            annotation_position='top left',
            annotation_font_color=color,
            annotation_font_size=9
        )
    return fig


def ewma_var_chart(dates,
                   rolling_var_95: np.ndarray,
                   rolling_var_99: np.ndarray):
    fig = _fig('EWMA VaR Through Time')
    fig.add_trace(go.Scatter(
        x=dates, y=rolling_var_95 * 100,
        name='95% EWMA VaR',
        line=dict(color='#FFB347', width=1.2)
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=rolling_var_99 * 100,
        name='99% EWMA VaR',
        line=dict(color='#FF4444', width=1.2)
    ))
    for start, end, color in [
        ('2020-02-01', '2020-06-01', '#FF4444'),
        ('2018-09-01', '2019-01-01', '#FFB347')
    ]:
        fig.add_vrect(x0=start, x1=end,
                      fillcolor=color, opacity=0.08, line_width=0)
    return fig


def backtest_chart(dates, returns: pd.Series,
                   var_series: pd.Series,
                   breach_dates,
                   breach_returns: pd.Series,
                   confidence: float):
    fig = _fig(f'{confidence*100:.0f}% Rolling VaR — Breach Analysis')
    fig.add_trace(go.Scatter(
        x=dates, y=returns * 100,
        name='Portfolio Return',
        line=dict(color='#4A90D9', width=0.6),
        opacity=0.8
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=-var_series * 100,
        name=f'{confidence*100:.0f}% VaR',
        line=dict(color='#FFB347', width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=breach_dates,
        y=breach_returns * 100,
        mode='markers',
        name=f'Breaches ({len(breach_dates)})',
        marker=dict(color='#FF4444', size=5)
    ))
    return fig


def stress_cumulative_chart(scenario_results: dict):
    fig = _fig('Cumulative Return During Stress Periods')
    colors = ['#FF4444', '#00A3FF', '#FFB347', '#00D4AA']
    for i, (name, data) in enumerate(scenario_results.items()):
        cum = (np.exp(data['returns'].cumsum()) - 1) * 100
        fig.add_trace(go.Scatter(
            x=list(range(len(cum))),
            y=cum.values,
            name=name,
            line=dict(color=colors[i % len(colors)], width=1.8)
        ))
    fig.add_hline(y=0, line_color='#2A2D3E',
                  line_dash='dash', line_width=1)
    return fig


def shock_bar_chart(shock_losses: dict, portfolio_value: float):
    names = list(shock_losses.keys())
    losses = [v / 1000 for v in shock_losses.values()]
    colors = ['#FFB347', '#FF7B47', '#FF5544', '#FF2222']
    fig = _fig(f'Shock Analysis — ${portfolio_value:,.0f} Portfolio')
    fig.add_trace(go.Bar(
        x=losses, y=names,
        orientation='h',
        marker_color=colors
    ))
    return fig


def correlation_heatmap(log_returns: pd.DataFrame):
    corr = log_returns.corr()
    fig = _fig('Asset Correlation Matrix')
    fig.add_trace(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[
            [0, '#FF4444'], [0.5, '#1E2130'], [1, '#00A3FF']
        ],
        zmid=0,
        text=corr.round(3).values,
        texttemplate='%{text}',
        textfont=dict(size=11, color='#FFFFFF')
    ))
    return fig


def diversification_bar_chart(tickers: list,
                               individual_vars: list,
                               weighted_avg: float,
                               portfolio_var: float):
    labels = tickers + ['Weighted Avg', 'Portfolio']
    values = individual_vars + [weighted_avg, portfolio_var]
    colors = (['#00A3FF'] * len(tickers) +
               ['#FFB347', '#00D4AA'])
    fig = _fig('95% VaR — Individual vs Portfolio')
    fig.add_trace(go.Bar(
        x=labels,
        y=[v * 100 for v in values],
        marker_color=colors
    ))
    return fig