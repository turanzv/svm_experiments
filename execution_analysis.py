from datetime import datetime

import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import re

pio.kaleido.scope.mathjax = None

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

    return df

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

    add_elapsed_time_column(df_pc, time_col='start')
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

    add_elapsed_time_column(df_pcp, time_col='start')
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

    add_elapsed_time_column(df_lpc, time_col='unixtime_us')
    return df_lpc

def convert_to_unixtime(timestamp):
    base_time, microseconds = timestamp.split('.')
    microseconds = microseconds[:6]
    dt = datetime.strptime(base_time, '%Y-%m-%dT%H:%M:%S')
    return int(dt.timestamp() * 1_000_000) + int(microseconds)

def aggregate_tps(row, df):
    mask = (df['start'] <= row['t']) & (df['end'] >= row['t'])
    return df.loc[mask, 'tps'].sum()

def add_elapsed_time_column(df, time_col='start'):
    base_time = df[time_col].min()
    df['elapsed_time_hours'] = (df[time_col] - base_time) / 3_600_000_000  # µs to hours

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

def make_pc_trend_plot(pc_sizes: list[int],
                        df_program_caches: list[pd.DataFrame],
                        df_lp_stats: list[pd.DataFrame]):
    # experiments should look like [2048, 1024, 512]
    # df_lp_stats should look like [2048_df_lp_stats, 1024_df_lp_stats, 512_df_lp_stats]
    data = {
        "PC Count":                     pc_sizes,
        "Misses Mean":                  [df_lp_stat['misses'].mean()                    for df_lp_stat in df_lp_stats],
        "Evictions Mean":               [df_lp_stat['evictions'].mean()                 for df_lp_stat in df_lp_stats],
        "Program Cache Time Mean (µs)": [df_program_cache['program_cache_us'].mean()    for df_program_cache in df_program_caches],
    }

    # Create and sort DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values("PC Count")

    # Build figure with secondary y
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Primary y-axis traces (Misses & Evictions)
    fig.add_trace(
        go.Scatter(
            x=df["PC Count"],
            y=df["Misses Mean"],
            mode="lines+markers",
            marker=dict(symbol="circle", size=8),
            name="Misses Mean"
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=df["PC Count"],
            y=df["Evictions Mean"],
            mode="lines+markers",
            marker=dict(symbol="square", size=8),
            name="Evictions Mean"
        ),
        secondary_y=False
    )

    # Secondary y-axis trace (Program Cache Time)
    fig.add_trace(
        go.Scatter(
            x=df["PC Count"],
            y=df["Program Cache Time Mean (µs)"],
            mode="lines+markers",
            marker=dict(symbol="triangle-up", size=8),
            name="Program Cache Time (µs)"
        ),
        secondary_y=True
    )

    # Update axes ranges and ticks
    fig.update_yaxes(
        title_text="Average Evictions",
        range=[0, 1.0],
        tickvals=[0.0, 0.25, 0.5, 0.75, 1.0],
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Average Program Cache Time (μs)",
        range=[0, 45000],
        tickvals=[0, 15000, 30000, 45000],
        ticktext=["0", "15k", "30k", "45k"],
        secondary_y=True
    )

    # X-axis ticks and range
    fig.update_xaxes(
        title_text="Program Cache Size",
        tickmode="array",
        tickvals=[512, 1024, 1536, 2048],
        ticktext=["512", "1024", "", "2048"],
        range=[500, 2100]
    )

    # Global layout styling
    fig.update_layout(
        font=dict(family="serif", size=20),
        showlegend=True,
        template="simple_white",
        margin=dict(l=20, r=20, t=20, b=20),
        width=800,
        height=400,
        legend=dict(
            x=.9,    # Far right
            y=1,    # Top
            xanchor="right",  # Anchor at the right edge
            yanchor="top",    # Anchor at the top edge
            bordercolor="black",
            borderwidth=1,
            bgcolor="white"   # (optional) white legend background for clarity
        )
    )

    # Export to PDF
    fig.write_image("figures/CachePerformance.pdf")

def add_pc_histogram(df, fig, color, experiment, column, size=1):
    fig.add_trace(go.Histogram(
        x=df[column],
        marker=dict(color=color, line=dict(width=1, color='black')),
        opacity=0.75,
        name=experiment,
        xbins=dict(
            start=int(df[column].min()),
            end = int(df[column].max()),
            size = size,
        )
    ))

def generate_pc_grouped_histogram():
    fig = go.Figure()
    for experiment in program_cache.keys():
        df = parse_loaded_programs_cache_df(program_cache[experiment]["log"])
        add_pc_histogram(df, fig, program_cache[experiment]["color"], experiment, "misses")
    fig.update_layout(
        barmode='group',
        showlegend=True,
        template="simple_white",
        width=900,
        height=480,
        font=dict(family="serif", size=20),
        margin=dict(l=20, r=20, t=25, b=25),
        xaxis_title=f"Misses per measurement",
        yaxis=dict(
            title="Occurrences / count",
            tickformat="d",
        ),
        xaxis=dict(
            range=[0,20]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5,
            bordercolor="Black",
            borderwidth=1
        ),
    )
    fig.write_image("figures/grouped_misses_hist.pdf")

    # Evictions histogram
    fig = go.Figure()
    for experiment in program_cache.keys():
        df = parse_loaded_programs_cache_df(program_cache[experiment]["log"])
        add_pc_histogram(df, fig, program_cache[experiment]["color"], experiment, "evictions")
    fig.update_layout(
        barmode='group',
        showlegend=False,
        template="simple_white",
        width=900,
        height=420,
        font=dict(family="serif", size=20),
        margin=dict(l=20, r=20, t=25, b=25),
        xaxis_title="Evictions per Measurement",
        yaxis=dict(
            title="Occurrences / count",
            tickformat="d"
        )
        
    )
    fig.write_image("figures/grouped_evictions_hist.pdf")

     # Prune time histogram
    fig = go.Figure()
    for experiment in program_cache.keys():
        df = parse_program_cache_prune_df(program_cache[experiment]["log"])
        add_pc_histogram(df, fig, program_cache[experiment]["color"], experiment, column="program_cache_prune_ms")
    fig.update_layout(
        barmode='group',
        showlegend=False,
        template="simple_white",
        width=900,
        height=420,
        font=dict(family="serif", size=20),
        margin=dict(l=20, r=20, t=25, b=25),
        xaxis=dict(
            title="Prune Time per Measurement (ms)",
            tickformat="d",
            range=[1, 20]
        ),
        yaxis=dict(
            title="Occurrences / count",
            tickformat="d"
        )
    )
    fig.write_image("figures/grouped_prune_hist.pdf")

    fig = go.Figure()
    for experiment in program_cache.keys():
        df = parse_program_cache_df(program_cache[experiment]["log"])
        add_pc_histogram(df, fig, program_cache[experiment]["color"], experiment, "program_cache_us", size=100_000)
    fig.update_layout(
        barmode='group',
        showlegend=False,
        template="simple_white",
        width=900,
        height=420,
        font=dict(family="serif", size=20),
        margin=dict(l=20, r=20, t=25, b=25),
        xaxis=dict(
            title="Program Cache Time per Measurement (μs)",
            tickformat="~s",
            range=[0, 2_000_000],
        ),
        yaxis=dict(
            title="Occurrences / count",
            tickformat="~s",
            type="log",
        )
    )
    fig.write_image("figures/grouped_program_cache_time_hist.pdf")

def generate_tps_grouped_histogram():
    fig = go.Figure()
    for experiment in execution.keys():
        df_tps = parse_tps_df(execution[experiment]["log"])
        add_pc_histogram(df_tps, fig, execution[experiment]["color"], experiment, "tps", size=1000)
    fig.update_layout(
        barmode='group',
        showlegend=True,
        template="simple_white",
        width=900,
        height=420,
        font=dict(family="serif", size=20),
        margin=dict(l=20, r=20, t=25, b=25),
        xaxis=dict(
            title="TPS per Batch",
            tickformat="d",
            range=[0, 20000],
        ),
        yaxis=dict(
            title="Occurrences / count",
            tickformat="d",
        ),
    )
    fig.write_image("figures/grouped_tps_hist.pdf")

program_cache = {
    "PC_2048": {
        "log": "logs/2048PC/2048PC-2025-03-12-19-21-02-mainnet-beta.log",
        "color": "red",
    },
    "PC_1024": {
        "log": "logs/1024PC/1024PC-2025-03-12-07-03-06-mainnet-beta.log",
        "color": "orange",
    },
    "PC_512": {
        "log": "logs/512PC/512PC-2025-03-11-17-33-19-mainnet-beta.log",
        'color': "blue",
    }
}

execution = {
    # "1.5TB_1": {
    #     "log": "logs/1_5TB/2025-01-07-04-20-00-mainnet-beta-1_5TB.log",
    #     'color': "blue",
    # },
    "1.5TB_0": {
        "log": "logs/1_5TB/2025-01-01-04-05-37-mainnet-beta.log",
        'color': "red",
    },
    # "1TB_1": {
    #     "log": "logs/1TB/2025-01-07-18-58-39-mainnet-beta-1TB.log",
    #     'color': "blue",
    # },
    "1TB_0": {
        "log": "logs/1TB/2025-01-04-08-21-52-mainnet-beta-1TB.log",
        'color': "orange",
    },
    # "512GB_1": {
    #     "log": "logs/512G/2025-01-08-20-00-20-mainnet-beta-512G.log",
    #     'color': "blue",
    # },
    "512GB_0": {
        "log": "logs/512G/2025-01-04-21-54-59-mainnet-beta-512G.log",
        'color': "yellow",
    },
    # "256GB_1": {
    #     "log": "logs/256G/2025-01-14-22-46-43-mainnet-beta-256GB.log",
    #     'color': "blue",
    # },
    "256GB_0": {
        "log": "logs/256G/2025-01-05-10-04-13-mainnet-beta-256G.log",
        'color': "green",
    },
}

generate_tps_grouped_histogram()
generate_pc_grouped_histogram()