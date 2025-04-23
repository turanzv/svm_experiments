from datetime import datetime

import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import re

def convert_to_unixtime(timestamp):
    base_time, microseconds = timestamp.split('.')
    microseconds = microseconds[:6]
    dt = datetime.strptime(base_time, '%Y-%m-%dT%H:%M:%S')
    return int(dt.timestamp() * 1_000_000) + int(microseconds)

def aggregate_tps(row, df):
    mask = (df['start'] <= row['t']) & (df['end'] >= row['t'])
    return df.loc[mask, 'tps'].sum()

def parse_execution_logs(log_file):
    # read the appropriate data into a dataframe
    data = []
    with open(log_file, 'r') as file:
        for line in file:
            if "datapoint: replay-slot-stats" in line:

                timestamp_match = re.search(r"\[(.*?)Z", line)
                transactions_match = re.search(r"total_transactions=(\d+)i", line)
                execute_us_match = re.search(r"execute_us=(\d+)i", line)
                
                if timestamp_match and transactions_match and execute_us_match:
                    timestamp = timestamp_match.group(1)
                    total_transactions_executed = int(transactions_match.group(1))
                    execute_us = int(execute_us_match.group(1))

                    data.append([timestamp, total_transactions_executed, execute_us])

    df = pd.DataFrame(data, columns=['timestamp', 'total_transactions_executed', 'execute_us'])
    
    # convert log timestamps into unix timestamps, apply to tps as well
    df['end'] = df['timestamp'].apply(convert_to_unixtime)
    df['start'] = df['end'] - df['execute_us']
    df['tps'] = df['total_transactions_executed'] * 1_000_000 / df['execute_us']
    
    start_time = df.start.min() // 1_000_000 * 1_000_000
    end_time = (df.end.max() + 1_000_000) // 1_000_000 * 1_000_000
    time_step = 100_000 # 100,000 microsends = .1 second

    time_series = pd.DataFrame({
        't': np.arange(start_time, end_time + time_step, time_step)
    })

    time_series['sum_tps'] = time_series.apply(aggregate_tps, axis=1, df=df)
    # window=50 means creating a moving average with the past 5 seconds' data
    time_series['moving_average'] = time_series['sum_tps'].rolling(window=50, min_periods=1).mean()
    
    # Display every 10 seconds
    filtered_time_series = time_series[time_series['t'] % 10000000 == 0].copy()
    # from micro sec to sec
    filtered_time_series['t'] = np.floor(filtered_time_series['t'] / 1000000)

    # Calculate time elapsed in seconds from the start of the log
    start_time = filtered_time_series['t'].min()  # First timestamp
    filtered_time_series['elapsed_time'] = filtered_time_series['t'] - start_time
    filtered_time_series['elapsed_time_hours'] = filtered_time_series['elapsed_time'] / 3600

    # Convert elapsed time to HH:MM:SS format
    filtered_time_series['elapsed_time_formatted'] = filtered_time_series['elapsed_time'].apply(
        lambda x: f"{int(x // 3600):02}:{int((x % 3600) // 60):02}:{int(x % 60):02}"
    )

    return filtered_time_series

def make_tps_plot(experiment, df):
    # Determine the range for the x-axis
    x_min = df['elapsed_time_hours'].min()
    x_max = df['elapsed_time_hours'].max()

    fig = go.Figure()

    # fig.add_trace(go.Scatter(x=df['elapsed_time_hours'], y=df['sum_tps'], mode='markers', name='TPS', marker=dict(size=2)))
    fig.add_trace(go.Scatter(x=df['elapsed_time_hours'], y=df['moving_average'], mode='lines', name='TPS Moving Average(1sec)', line=dict(width=0.5)))

    # Add horizontal lines for max and average TPS
    # fig.add_hline(y=average_tps, line_dash="dot", line_color="green", annotation=None)
    # fig.add_hline(y=max_tps, line_dash="dash", line_color="red", annotation_text=f"Max TPS: {max_tps:.2f}", annotation_position="top right")

    fig.update_layout(
        showlegend=False,
        title=None,
        font=dict(
            family="serif",
            size=20
        ),
        xaxis=dict(
            title=None,
            tickmode='linear',
            dtick=1,
            tickformat=".1f",
            range=[x_min, x_max],
            anchor="y",
            position=0
        ),
        yaxis=dict(
            title=None,
            range=[0, 15000],
            anchor="x",
            position=0,
            tickformat="~s",
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        template="simple_white",
        width=800,
        height=400
    )

    fig.write_image(f"{experiment}_Exec_TPS.pdf")

df = parse_execution_logs('logs/1_5TB/2025-01-07-04-20-00-mainnet-beta-1_5TB.log')
make_tps_plot("1_5TB_1", df)