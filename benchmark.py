import os
import re
import pandas as pd
from datetime import datetime

def parse_logs(log_file):
    # Regular expression to match log entries
    log_pattern = re.compile(
        r'(?P<timestamp>\d+-\d+-\d+ \d+:\d+:\d+) - running reward: (?P<reward>[-+]?\d*\.\d+|\d+) at episode (?P<episode>\d+), frame count (?P<frame>\d+), loss: (?P<loss>[-+]?\d*\.\d+|\d+)'
    )

    # Lists to store parsed log data
    timestamps = []
    rewards = []
    episodes = []
    frames = []
    losses = []

    # Parse the log file with UTF-16 encoding to handle BOM
    with open(log_file, 'r', encoding='utf-16') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            match = log_pattern.match(line)
            if match:
                timestamps.append(match.group('timestamp'))
                rewards.append(float(match.group('reward')))
                episodes.append(int(match.group('episode')))
                frames.append(int(match.group('frame')))
                losses.append(float(match.group('loss')))
            # else:
            #     print(f"No match found for line: {line}")

    # Check if we have collected any data
    if not timestamps:
        print(f"No data parsed from log file: {log_file}")
        return pd.DataFrame()

    # Create a DataFrame from the parsed data
    log_data = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps),
        'reward': rewards,
        'episode': episodes,
        'frame': frames,
        'loss': losses
    })

    return log_data

def benchmark_log_data(log_data):
    if log_data.empty:
        return {
            'average_reward': None,
            'min_reward': None,
            'max_reward': None,
            'average_loss': None,
            'min_loss': None,
            'max_loss': None,
            'total_episodes': None,
            'total_frames': None,
            'total_training_time': None
        }

    # Calculate metrics
    metrics = {
        'average_reward': log_data['reward'].mean(),
        'min_reward': log_data['reward'].min(),
        'max_reward': log_data['reward'].max(),
        'average_loss': log_data['loss'].mean(),
        'min_loss': log_data['loss'].min(),
        'max_loss': log_data['loss'].max(),
        'total_episodes': log_data['episode'].max(),
        'total_frames': log_data['frame'].max(),
        'total_training_time': (log_data['timestamp'].max() - log_data['timestamp'].min()).total_seconds()
    }

    return metrics

def main(log_root_dir):
    # Dictionary to store benchmarks for each log file
    benchmarks = {}

    # Process each log directory in the root directory
    for dir_name in os.listdir(log_root_dir):
        log_dir = os.path.join(log_root_dir, dir_name)
        log_file = os.path.join(log_dir, 'gym.log')
        
        if os.path.isfile(log_file):
            print(f"Parsing log file: {log_file}")
            log_data = parse_logs(log_file)
            benchmarks[dir_name] = benchmark_log_data(log_data)
        else:
            print(f"No log file found in directory: {log_dir}")

    # Create a DataFrame from the benchmarks
    benchmark_df = pd.DataFrame.from_dict(benchmarks, orient='index')

    return benchmark_df

if __name__ == "__main__":
    log_root_dir = "./gym-output"  # Root directory containing the log directories
    benchmark_df = main(log_root_dir)

    # Display the benchmark results
    print(benchmark_df)

    # Save the benchmark results to a CSV file
    benchmark_df.to_csv('benchmark_results.csv')
