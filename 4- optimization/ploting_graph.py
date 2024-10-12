import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
from models import *

def plot_candlestick(df : pd.DataFrame, symbol : str , id: int):
    # Create a subplot figure with 2 rows
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.8, 0.2])

    # Add the candlestick chart to the first row
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlesticks'
    ), row=1, col=1)

    # Add the volume bars to the second row
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['volume'],
        marker=dict(
            color=['green' if df['close'][i] > df['open'][i] else 'red' for i in range(len(df))]
        ),
        name='Volume'
    ), row=2, col=1)

    if('MA50' in df.columns):
        # Add moving average (MA50) to the first row
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['MA50'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='MA50'
        ), row=1, col=1)

    # Define shapes and markers for strategy signals and gaps
    shapes = []
    annotations = []
    dashed_lines = []

    arrow_length = 40
    position_counter = 1
    open_positions = []
    for i in range(1, len(df)):
        if df['strategy_signal'][i] == StrategySignal.ENTER_LONG:
            price = df['open'][i] 
            if df['position open order'][i] == StrategySignal.ENTER_LONG:
                open_positions.append((df['date'][i], df['open_position_price'][i]))
                price = df['open_position_price'][i]
                color = 'green'
            else:
                color = 'yellow'
            arrow_length = abs(arrow_length)
            gap_signal = 'Gap&Go' if df["type of gap strategy"][i] == GapSignal.Gap_N_Go else 'Close Gap' 
            annotations.append(dict(
                x=df['date'][i],
                y=price,
                xref='x', yref='y1',
                text=f'{gap_signal} - LONG - {position_counter} - {price:.2f}',
                showarrow=True,
                arrowhead=1,
                arrowwidth=3,
                ax=0, ay=arrow_length,  # Make the arrow longer
                arrowcolor=color
            ))
            arrow_length = (-1) * arrow_length
        elif df['strategy_signal'][i] == StrategySignal.ENTER_SHORT:
            price = df['open'][i]
            if df['position open order'][i] == StrategySignal.ENTER_SHORT:
                open_positions.append((df['date'][i], df['open_position_price'][i]))
                price = df['open_position_price'][i]
                color = 'red'
            else:
                color = 'yellow'
            arrow_length = -abs(arrow_length)
            gap_signal = 'Gap&Go' if df["type of gap strategy"][i] == GapSignal.Gap_N_Go else 'Close Gap' 
            annotations.append(dict(
                x=df['date'][i],
                y=price,
                xref='x', yref='y1',
                text=f'{gap_signal} - SHORT - {position_counter} - {price:.2f}',
                showarrow=True,
                arrowhead=1,
                arrowwidth=3,
                ax=0, ay=arrow_length,  # Make the arrow longer
                arrowcolor=color
            ))
            arrow_length = (-1) * arrow_length
        if df['position close order'][i] == StrategySignal.CLOSE_LONG or df['position close order'][i] == StrategySignal.CLOSE_SHORT:
            arrow_direction = 0
            if df['position close order'][i] == StrategySignal.CLOSE_LONG:
                color = 'green'
                arrow_direction = -1
            else:
                color = 'red'
                arrow_direction = 1
            annotations.append(dict(
                x=df['date'][i],
                y=df['close_position_price'][i],
                xref='x', yref='y1',
                text=f'{df["exit_signal"][i]} {position_counter} - {df["close_position_price"][i]:.2f}',
                showarrow=True,
                arrowhead=1,
                arrowwidth=3,
                ax=0, ay=arrow_length,  # Make the arrow longer
                arrowcolor=color
            ))
            dashed_lines.append(dict(
                    type='line',
                    x0=open_positions[position_counter-1][0],
                    y0=open_positions[position_counter-1][1],
                    x1=df['date'][i],
                    y1=df['close_position_price'][i],
                    line=dict(color='black', dash='dash', width=2.5)
                ))
            position_counter += 1
        elif df['position close order'][i] == StrategySignal.TAKEPROFIT:
            annotations.append(dict(
                x=df['date'][i],
                y=df['close_position_price'][i],
                xref='x', yref='y1',
                text=f'% Take Profit % {position_counter} - {df["close_position_price"][i]:.2f}',
                showarrow=True,
                arrowhead=1,
                arrowwidth=3,
                ax=0, ay=arrow_length,  # Make the arrow longer
                arrowcolor='green'
            ))
            dashed_lines.append(dict(
                    type='line',
                    x0=open_positions[position_counter-1][0],
                    y0=open_positions[position_counter-1][1],
                    x1=df['date'][i],
                    y1=df['close_position_price'][i],
                    line=dict(color='black', dash='dash', width=2.5)
                ))
            position_counter += 1
        elif df['position close order'][i] == StrategySignal.STOPLOSS:
            annotations.append(dict(
                x=df['date'][i],
                y=df['close_position_price'][i],
                xref='x', yref='y1',
                text=f'% Stop Loss % {position_counter} - {df["close_position_price"][i]:.2f}',
                showarrow=True,
                arrowhead=1,
                arrowwidth=3,
                ax=0, ay=arrow_length,  # Make the arrow longer
                arrowcolor='red'
            ))
            dashed_lines.append(dict(
                    type='line',
                    x0=open_positions[position_counter-1][0],
                    y0=open_positions[position_counter-1][1],
                    x1=df['date'][i],
                    y1=df['close_position_price'][i],
                    line=dict(color='black', dash='dash', width=2.5)
                ))
            position_counter += 1



        if df['gap_down'][i]:
            shapes.append(dict(
                type='rect',
                xref='x', yref='y1',
                x0=df['date'][i-1], y0=df['low'][i-1],
                x1=df['date'][i], y1=df['open'][i],
                fillcolor='yellow', opacity=0.5, line_width=0
            ))
        if df['gap_up'][i]:
            shapes.append(dict(
                type='rect',
                xref='x', yref='y1',
                x0=df['date'][i-1], y0=df['high'][i-1],
                x1=df['date'][i], y1=df['open'][i],
                fillcolor='yellow', opacity=0.5, line_width=0
            ))

    # Update layout for secondary y-axis (for volume)
    fig.update_layout(
        title_text=f'{symbol} Candlestick Chart - Strategy ID {id}',
        yaxis_title='Price',
        xaxis_title='Date',
        shapes=shapes + dashed_lines,
        annotations=annotations,
        xaxis=dict(
            rangeslider=dict(visible=False),
            rangebreaks=[
                dict(bounds=["sat", "mon"])  # Hide weekends from Saturday to Monday
            ]
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": ["annotations", []],
                        "label": "Hide Annotations",
                        "method": "relayout"
                    },
                    {
                        "args": ["annotations", annotations],
                        "label": "Show Annotations",
                        "method": "relayout"
                    },
                    {
                        "args": ["shapes", []],
                        "label": "Hide Shapes",
                        "method": "relayout"
                    },
                    {
                        "args": ["shapes", shapes + dashed_lines],
                        "label": "Show Shapes",
                        "method": "relayout"
                    }
                ],
                "direction": "down",
                "showactive": True,
            }
        ]
    )

    # Update y-axis titles and other layout settings for subplots
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    # fig.show()
    fig.write_html(f'GRAPHS/{symbol}_backtesting_{id}.html')
