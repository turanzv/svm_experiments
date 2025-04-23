import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

# disable mathjax, no LaTeX in figures
pio.kaleido.scope.mathjax = None

def parse_memory_logs(log_file_path):

    columns = ["timestamp", "uid", "pid", "minflt/s", "majflt/s", "vsz", "rss", "%mem", "command"]
    data = []
    
    # Parse the log file
    with open(log_file_path, 'r') as file:
        header_found = False
        for line in file:
            # Ignore lines before the header
            if not header_found:
                if line.strip().startswith("Timestamp,UID,PID,minflt/s,majflt/s,VSZ,RSS,%MEM,Command"):
                    header_found = True
                continue

            # Parse the log data after the header
            parts = line.split(",")
            if len(parts) < 9:  # Skip lines that don't have enough columns
                continue
            timestamp = parts[0]
            uid = int(parts[1])
            pid = int(parts[2])
            minflt_s = float(parts[3]) if parts[3] else None
            majflt_s = float(parts[4]) if parts[4] else None
            vsz = float(parts[5]) if parts[5] else None
            rss = float(parts[6]) if parts[6] else None
            p_mem = float(parts[7]) if parts[7] else None
            command = parts[8].strip()
            data.append([timestamp, uid, pid, minflt_s, majflt_s, vsz, rss, p_mem, command])

    df = pd.DataFrame(data, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Compute time since start in hours (as float)
    start_time = df.index[0]
    df['hours_since_start'] = (df.index - start_time).total_seconds() / 3600
    df.set_index('hours_since_start', inplace=True)

    df['RSS (GB)'] = df['rss'] / (1024 * 1024)
    df['VSZ (GB)'] = df['vsz'] / (1024 * 1024)

    df = truncate_warmup(df)

    return df

def make_full_memory_plot(experiment, df):

    # Create subplots with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces for memory usage
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSS (GB)'], name="RSS (GB)", line=dict(color="blue")),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['VSZ (GB)'], name="VSZ (GB)", line=dict(color="cyan")),
        secondary_y=False
    )

    # Add traces for page faults
    fig.add_trace(
        go.Scatter(x=df.index, y=df['minflt/s'], name="Minor Page Faults (minflt/s)", line=dict(color="orange", dash="dot")),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['majflt/s'], name="Major Page Faults (majflt/s)", line=dict(color="red", dash="dot")),
        secondary_y=True
    )

    # Set y-axes titles and scaling
    fig.update_yaxes(title_text=None, secondary_y=False)
    fig.update_yaxes(title_text=None, secondary_y=True)
    
    fig.update_layout(
        font=dict(
            family="serif",
            size=20
        ),
        margin=dict(l=20, r=20, t=30, b=30),
        legend_title="Metrics",
        showlegend=False,
        template="simple_white",
        width=800,
        height=400,
    )

    fig.write_image(""+experiment+"_Mem.pdf")

def make_rss_vsz_plot(experiment, df):
    fig = go.Figure()

    # Add traces for memory usage
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSS (GB)'], name="RSS (GB)", line=dict(color="blue"))
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['VSZ (GB)'], name="VSZ (GB)", line=dict(color="cyan"))
    )
    
    fig.update_layout(
        font=dict(
            family="serif",
            size=20
        ),
        margin=dict(l=20, r=20, t=30, b=30),
        legend_title="Metrics",
        xaxis_title="Hours since start",
        yaxis_title="Memory (GB)",
        showlegend=True,
        template="simple_white",
        width=800,
        height=400,
    )

    fig.write_image(""+experiment+"_RSS_VSZ.pdf")

def make_page_fault_plot(experiment, df):

    fig = go.Figure()

    # Add traces for page faults
    fig.add_trace(
        go.Scatter(x=df.index, y=df['minflt/s'], name="minflt/s", line=dict(color="orange", dash="dot"))
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['majflt/s'], name="majflt/s", line=dict(color="red", dash="dot"))
    )
    
    fig.update_layout(
        font=dict(
            family="serif",
            size=20
        ),
        margin=dict(l=20, r=20, t=30, b=30),
        legend_title="Metrics",
        showlegend=True,
        xaxis_title="Hours since start",
        yaxis_title="Page Faults per Second",
        template="simple_white",
        width=800,
        height=400,
    )

    fig.write_image(""+experiment+"_page_faults.pdf")

def make_page_fault_histogram(experiment, df, maj=False):
    column = ""
    title_text = ""
    file_text = ""
    if maj:
        column = "majflt/s"
        title_text = "Major"
        file_text = "major"
    else:
        column = "minflt/s"
        title_text = "Minor"
        file_text = "minor"

    fig = go.Figure(
        data=[
            go.Histogram(
                x=df[column],
                xbins=dict(start=0, end=1000000, size=25000),
                marker=dict(line=dict(width=1, color="black")),
                opacity=0.7
            )
        ]
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=30),
        xaxis_title=f'{title_text} Page Faults per Second',
        yaxis=dict(title='Occurrences', type='log'),
        template='simple_white',
        width=800,
        height=400,
    )

    fig.write_image(f"{experiment}_{file_text}_faults.pdf")


def make_page_fault_overlay_plot(experiments: list[str], dfs: list[pd.DataFrame], output_file: str, maj: bool = False):
    """
    Create an overlaid page faults plot for multiple experiments.
    :param experiments: List of experiment names
    :param dfs: List of DataFrames corresponding to each experiment
    :param output_file: Base filename (without extension) for saving the PDF
    :param maj: If True, plot major page faults; otherwise, plot minor page faults.
    """
    fig = go.Figure()

    for exp, df in zip(experiments, dfs):
        if maj:
            series = df['majflt/s']
            label = f"{exp} Major PF"
            dash_style = 'dot'
        else:
            series = df['minflt/s']
            label = f"{exp} Minor PF"
            dash_style = 'solid'
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=series,
                mode='lines',
                name=label,
                line=dict(dash=dash_style)
            )
        )

    fig.update_layout(
        font=dict(
            family="serif",
            size=20
        ),
        margin=dict(l=20, r=20, t=30, b=30),
        legend_title="Experiment Metrics",
        showlegend=True,
        xaxis_title="Hours since start",
        yaxis_title="Page Faults per Second",
        template="simple_white",
        width=800,
        height=400,
    )

    # Save the overlaid plot
    fig.write_image(f"figures/{output_file}.pdf")

def make_page_fault_overlay_histogram(experiments: list[str],
                                      dfs: list[pd.DataFrame],
                                      output_file: str,
                                      maj: bool = False):
    """
    Create an overlaid histogram of page faults for multiple experiments.
    :param experiments: List of experiment names
    :param dfs: List of DataFrames corresponding to each experiment
    :param output_file: Base filename (without extension) for saving the PDF
    :param maj: If True, plot major page faults; otherwise, plot minor page faults.
    """
    # Select column and title based on maj flag
    column = "majflt/s" if maj else "minflt/s"
    title_text = "Major" if maj else "Minor"

    # 1) compute shared binning across all dataframes:
    all_vals = pd.concat([df[column] for df in dfs])
    min_val, max_val = all_vals.min(), all_vals.max()
    bin_size = (max_val - min_val) / 50  # or whatever # of bins you like

    fig = go.Figure()

    for exp, df in zip(experiments, dfs):
        fig.add_trace(
            go.Histogram(
                x=df[column],
                name=f"{exp} {title_text} PF",
                opacity=0.6,
                xbins=dict(start=min_val, end=max_val, size=bin_size),
                marker=dict(line=dict(width=1, color="black"))
            )
        )

    fig.update_layout(
        barmode="overlay",
        margin=dict(l=20, r=20, t=30, b=30),
        legend_title="Experiments",
        xaxis_title=f"{title_text} Page Faults per Second",
        yaxis_title="Occurrences",
        # yaxis_type="log",
        template="simple_white",
        width=800,
        height=400,
    )

    fig.write_image(f"figures/{output_file}.pdf")

def truncate_warmup(df):
    # Truncate entire series to start five minutes after the major fault peak
    peak_time = df['majflt/s'].idxmax()  # hours_since_start index of the major fault peak
    # Add a five-minute offset (5 minutes = 5/60 hours)
    window_start = peak_time + (5 / 60)
    # Slice DataFrame to include only entries after that time
    return df[df.index >= window_start]

def generate_figures(experiment, log_file):
    df = parse_memory_logs(log_file)
    make_full_memory_plot(experiment, df)
    make_rss_vsz_plot(experiment, df)
    make_page_fault_plot(experiment, df)
    make_page_fault_histogram(experiment, df)
    make_page_fault_histogram(experiment, df, maj=True)

TB_15_1 = 'logs/1_5TB/2025-01-07-04-19-59--memory.log'
TB_1_1 = 'logs/1TB/2025-01-07-18-55-31--memory.log'
GB_512_1 = 'logs/512G/2025-01-08-19-57-14--memory.log'
GB_256_1 = 'logs/256G/2025-01-14-22-46-05--memory.log'

PC_2048 = 'logs/2048PC/2048PC-2025-03-12-19-21-02--memory.log'
PC_1024 = 'logs/1024PC/1024PC-2025-03-12-07-03-06--memory.log'
PC_512 = 'logs/512PC/512PC-2025-03-11-17-33-19--memory.log'

pc_2048_df = parse_memory_logs(PC_2048)
pc_1024_df = parse_memory_logs(PC_1024)
pc_512_df = parse_memory_logs(PC_512)

make_page_fault_overlay_plot(
    ["512PC", "1024PC", "2048PC"],
    [pc_512_df, pc_1024_df, pc_2048_df],
    "Min_Flt"
)

make_page_fault_overlay_plot(
    ["512PC", "1024PC", "2048PC"],
    [pc_512_df, pc_1024_df, pc_2048_df],
    "Maj_Flt",
    True
)

make_page_fault_overlay_histogram(
    ["512PC", "1024PC", "2048PC"],
    [pc_512_df, pc_1024_df, pc_2048_df],
    "Min_Flt_Hist",
    False
)

make_page_fault_overlay_histogram(
    ["512PC", "1024PC", "2048PC"],
    [pc_512_df, pc_1024_df, pc_2048_df],
    "Maj_Flt_Hist",
    True
)

generate_figures("figures/memory/1_5TB",TB_15_1)
generate_figures("figures/memory/1_TB_1",TB_1_1)
generate_figures("figures/memory/512_GB_1",GB_512_1)
generate_figures("figures/memory/256_GB_1",GB_256_1)

generate_figures("figures/memory/2048_PC",PC_2048)
generate_figures("figures/memory/1024_PC",PC_1024)
generate_figures("figures/memory/512_PC",PC_512)
