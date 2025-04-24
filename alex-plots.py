import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal


def parse_memory_logs(log_file_path):
    def truncate_warmup(df):
        peak_time = df['majflt/s'].idxmax()
        window_start = peak_time + (5 / 60)
        return df[df.index >= window_start]

    columns = [
        "timestamp",
        "uid",
        "pid",
        "minflt/s",
        "majflt/s",
        "vsz",
        "rss",
        "%mem",
        "command"
    ]
    data = []

    # Parse the log file
    with open(log_file_path, 'r') as file:
        header_found = False
        for line in file:
            if not header_found:
                if line.strip().startswith("Timestamp,UID,PID,minflt/s,majflt/s,VSZ,RSS,%MEM,Command"):
                    header_found = True
                continue

            parts = line.split(",")
            if len(parts) < 9:
                continue
            time = parts[0]
            uid = int(parts[1])
            pid = int(parts[2])
            minflt_s = float(parts[3]) if parts[3] else None
            majflt_s = float(parts[4]) if parts[4] else None
            vsz = float(parts[5]) if parts[5] else None
            rss = float(parts[6]) if parts[6] else None
            p_mem = float(parts[7]) if parts[7] else None
            cmd = parts[8].strip()
            data.append([
                time,
                uid,
                pid,
                minflt_s,
                majflt_s,
                vsz,
                rss,
                p_mem,
                cmd
            ])

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


options = {
    "template": "simple_white",
    "width": 900,
    "height": 420,
    "showlegend": True,
    "margin": {
        "l": 20,
        "r": 20,
        "t": 25,
        "b": 25
    },
    "font": {
        "family": "serif",
        "size": 20,
    }
}


def make_rss_vsz_plot(df, fig, color, name):
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['RSS (GB)'],
            name=f"{name} RSS (GB)",
            line={"color": color, "dash": "solid"}
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['VSZ (GB)'],
            name=f"{name} VSZ (GB)",
            line={"color": color, "dash": "dot"})
    )
    fig.update_layout(
        legend_title="Memory Metrics",
        xaxis_title="Time / hours",
        yaxis_title="Memory / GB",
        **options
    )


# TODO: Subsample and clarify
def make_flt_plot(df, fig, color, name, idx):
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[idx].rolling(window=100).max().rolling(window=100).mean(),
            name=f"{name} {idx}",
            line={"color": color}
        )
    )
    fig.update_layout(
        legend_title="Page faults",
        **options
    )


def make_page_fault_histogram(df, fig, color, experiment, column):
    title_text = "Major" if column == "majflt/s" else "Minor"

    fig.add_trace(
        go.Histogram(
            x=df[column],
            xbins={
                'start': 0,
                'end': 1000000,
                'size': 25000,
            },
            marker=dict(line={"width": 1, "color": "black"}, color=color),
            opacity=0.7,
            name=experiment,
        )
    )

    fig.update_layout(
        xaxis_title=f"{title_text} Page faults per second",
        yaxis=dict(title="Occurrences / count", type='log'),
        **options,
    )


def make_full_memory_plot(experiment, df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces for memory usage
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSS (GB)'], name="RSS (GB)", line=dict(color="magenta")),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['VSZ (GB)'], name="VSZ (GB)", line=dict(color="cyan")),
        secondary_y=False
    )

    # Add traces for page faults
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['minflt/s'].rolling(window=100).max().rolling(window=100).mean(),
            name="Minor Page Faults (minflt/s)",
            line=dict(color="green")
        ),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['majflt/s'].rolling(window=100).max().rolling(window=100).mean(),
            name="Major Page Faults (majflt/s)",
            line=dict(color="black")
        ),
        secondary_y=True
    )

    # Set y-axes titles and scaling
    # fig.update_yaxes(title_text="Memory / GB", secondary_y=False)
    # fig.update_yaxes(title_text="Page Faults", secondary_y=True)

    fig.update_layout(
        legend_title="Metrics",
        **options
    )

    fig.write_image(""+experiment+"_Mem.pdf")


program_cache = {
    "PC_2048": {
        "log": 'logs/2048PC/2048PC-2025-03-12-19-21-02--memory.log',
        "color": "red"
    },
    "PC_1024": {
        "log": 'logs/1024PC/1024PC-2025-03-12-07-03-06--memory.log',
        "color": "orange"
    },
    "PC_512": {
        "log": 'logs/512PC/512PC-2025-03-11-17-33-19--memory.log',
        'color': "blue"
    }
}

memory_logs = {
    "TB_15_1": 'logs/1_5TB/2025-01-07-04-19-59--memory.log',
    "TB_1_1": 'logs/1TB/2025-01-07-18-55-31--memory.log',
    "GB_512_1": 'logs/512G/2025-01-08-19-57-14--memory.log',
    "GB_256_1": 'logs/256G/2025-01-14-22-46-05--memory.log',
}


def generate_rss_vsz():
    fig = go.Figure()
    for experiment in program_cache.keys():
        df = parse_memory_logs(program_cache[experiment]["log"])
        make_rss_vsz_plot(df, fig, program_cache[experiment]["color"], experiment)

    fig.write_image("figures/RSS_VSZ.pdf")


def generate_page_fault_histograms():
    fig = go.Figure()
    for experiment in program_cache.keys():
        df = parse_memory_logs(program_cache[experiment]["log"])
        make_page_fault_histogram(df, fig, program_cache[experiment]["color"], experiment, column="majflt/s")

    fig.write_image("figures/majflthist.pdf")

    fig = go.Figure()
    for experiment in program_cache.keys():
        df = parse_memory_logs(program_cache[experiment]["log"])
        make_page_fault_histogram(df, fig, program_cache[experiment]["color"], experiment, column="minflt/s")

    fig.write_image("figures/minflthist.pdf")


def generate_faults():
    fig = go.Figure()
    for experiment in program_cache.keys():
        df = parse_memory_logs(program_cache[experiment]["log"])
        make_flt_plot(df, fig, program_cache[experiment]["color"], experiment, "majflt/s")

    fig.write_image("figures/majfaults.pdf")

    fig = go.Figure()
    for experiment in program_cache.keys():
        df = parse_memory_logs(program_cache[experiment]["log"])
        make_flt_plot(df, fig, program_cache[experiment]["color"], experiment, "minflt/s")

    fig.write_image("figures/minfaults.pdf")


if __name__ == "__main__":
    generate_rss_vsz()
    generate_faults()
    df = parse_memory_logs(program_cache["PC_512"]["log"])
    make_full_memory_plot("figures/comparison.pdf", df)
    generate_page_fault_histograms()
