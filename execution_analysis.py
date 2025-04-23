from datetime import datetime

import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import re

def parse_tps_df(log_file):
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
    df['end'] = df['timestamp'].apply(convert_to_unixtime)
    df['start'] = df['end'] - df['execute_us']
    df['tps'] = df['total_transactions_executed'] * 1_000_000 / df['execute_us']

    start_time = df.start.min() // 1000000 * 1000000
    end_time = (df.start.max() + 1000000) // 1000000 * 1000000
    time_step = 100_000 # 100,000 nanosends = .1 second

    time_series = pd.DataFrame({
        't': np.arange(start_time, end_time + time_step, time_step)
    })

    time_series['sum_tps'] = time_series.apply(aggregate_tps, axis=1, df=df)
    # window=50 means creating a moving average with the past 5 second's data
    time_series['moving_average'] = time_series['sum_tps'].rolling(window=50, min_periods=1).mean()

    # Display every 10 seconds
    filtered_time_series = time_series[time_series['t'] % 10000000 == 0].copy()
    # from micro sec to sec
    filtered_time_series.loc[:, 't'] = np.floor(filtered_time_series['t'] / 1000000)

    # Calculate time elapsed in seconds from the start of the log
    start_time = filtered_time_series['t'].min()  # First timestamp
    filtered_time_series.loc[:, 'elapsed_time'] = filtered_time_series['t'] - start_time
    filtered_time_series.loc[:, 'elapsed_time_hours'] = filtered_time_series['elapsed_time'] / 3600

    # Convert elapsed time to HH:MM:SS format
    filtered_time_series.loc[:, 'elapsed_time_formatted'] = filtered_time_series['elapsed_time'].apply(
        lambda x: f"{int(x // 3600):02}:{int((x % 3600) // 60):02}:{int(x % 60):02}"
    )
    return filtered_time_series

def parse_program_cache_df(log_file):
    program_cache_data = []
    with open(log_file, 'r') as file:
        for line in file:
            if "datapoint: replay-slot-stats" in line:
                timestamp_match = re.search(r"\[(.*?)Z", line)
                slot_match = re.search(r"slot=(\d+)i", line)
                program_cache_match = re.search(r"program_cache_us=(\d+)i", line)
                if timestamp_match and slot_match and program_cache_match:
                    timestamp = timestamp_match.group(1)
                    slot = int(slot_match.group(1))
                    program_cache_us = int(program_cache_match.group(1))
                    program_cache_data.append([timestamp, slot, program_cache_us])
    df_pc = pd.DataFrame(program_cache_data, columns=['timestamp', 'slot', 'program_cache_us'])
    df_pc['end'] = df_pc['timestamp'].apply(convert_to_unixtime)
    df_pc['start'] = df_pc['end'] - df_pc['program_cache_us']
    df_pc['duration_us'] = df_pc['program_cache_us']
    return df_pc

def parse_program_cache_prune_df(log_file):
    program_cache_prune_data = []
    with open(log_file, 'r') as file:
        for line in file:
            if "datapoint: bank-forks_set_root" in line:
                timestamp_match = re.search(r"\[(.*?)Z", line)
                slot_match = re.search(r"slot=(\d+)i", line)
                prune_match = re.search(r"program_cache_prune_ms=(\d+)i", line)
                if timestamp_match and slot_match and prune_match:
                    timestamp = timestamp_match.group(1)
                    slot = int(slot_match.group(1))
                    program_cache_prune_ms = int(prune_match.group(1))
                    program_cache_prune_data.append([timestamp, slot, program_cache_prune_ms])
    df_pcp = pd.DataFrame(program_cache_prune_data, columns=['timestamp', 'slot', 'program_cache_prune_ms'])
    df_pcp['end'] = df_pcp['timestamp'].apply(convert_to_unixtime)
    df_pcp['start'] = df_pcp['end'] - df_pcp['program_cache_prune_ms'] * 1000
    df_pcp['duration_us'] = df_pcp['program_cache_prune_ms'] * 1000
    return df_pcp

def parse_loaded_programs_cache_df(log_file):
    loaded_data = []
    with open(log_file, 'r') as file:
        for line in file:
            if "datapoint: loaded-programs-cache-stats" in line:
                timestamp_match = re.search(r"\[(.*?)Z", line)
                slot_match = re.search(r"slot=(\d+)i", line)
                if timestamp_match and slot_match:
                    fields = dict(re.findall(r"(\w+)=(-?\d+)i", line))
                    fields = {k: int(v) for k, v in fields.items()}
                    fields['timestamp'] = timestamp_match.group(1)
                    fields['slot'] = int(slot_match.group(1))
                    loaded_data.append(fields)
    df_lpc = pd.DataFrame(loaded_data)
    df_lpc['unixtime_us'] = df_lpc['timestamp'].apply(convert_to_unixtime)
    return df_lpc

def convert_to_unixtime(timestamp):
    base_time, microseconds = timestamp.split('.')
    microseconds = microseconds[:6]
    dt = datetime.strptime(base_time, '%Y-%m-%dT%H:%M:%S')
    return int(dt.timestamp() * 1_000_000) + int(microseconds)

def aggregate_tps(row, df):
    mask = (df['start'] <= row['t']) & (df['end'] >= row['t'])
    return df.loc[mask, 'tps'].sum()

def parse_execution_logs(log_file):
    df = parse_tps_df(log_file)
    df_program_cache = parse_program_cache_df(log_file)
    df_program_cache_prune = parse_program_cache_prune_df(log_file)
    df_loaded_programs_cache = parse_loaded_programs_cache_df(log_file)
    return (df, df_program_cache, df_program_cache_prune, df_loaded_programs_cache)

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

tps_df = parse_tps_df('logs/1_5TB/2025-01-07-04-20-00-mainnet-beta-1_5TB.log')
pc_df = parse_program_cache_df('logs/1_5TB/2025-01-07-04-20-00-mainnet-beta-1_5TB.log')
pcp_df = parse_program_cache_prune_df('logs/1_5TB/2025-01-07-04-20-00-mainnet-beta-1_5TB.log')
lp_df = parse_loaded_programs_cache_df('logs/1_5TB/2025-01-07-04-20-00-mainnet-beta-1_5TB.log')
make_tps_plot("1_5TB_1", tps_df)