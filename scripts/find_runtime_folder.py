import sys
import os
from datetime import datetime

def parse_timestamp(line):
    try:
        # Extract the timestamp between the first '[' and the first space
        timestamp = line.split(' ')[0].strip("[]")
        # Truncate fractional seconds to 6 digits
        if "." in timestamp:
            base_time, fractional = timestamp.split(".")
            fractional = fractional[:6]  # Keep only the first 6 digits
            timestamp = f"{base_time}.{fractional}Z"
        # Parse the timestamp
        return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    except (ValueError, IndexError):
        return None

def get_first_and_last_timestamp(log_file_path):
    try:
        with open(log_file_path, 'r') as log_file:
            # Read the first line
            first_line = log_file.readline()
            first_timestamp = parse_timestamp(first_line)
            
            if not first_timestamp:
                print(f"  No valid timestamp in first line of {log_file_path}.")
                return None, None

            # Read the last line efficiently
            log_file.seek(0, os.SEEK_END)  # Move to end of file
            buffer_size = 1024
            while True:
                position = log_file.tell()
                seek_position = max(position - buffer_size, 0)
                log_file.seek(seek_position)
                lines = log_file.readlines()

                # Take the last non-empty line
                for line in reversed(lines):
                    last_timestamp = parse_timestamp(line)
                    if last_timestamp:
                        return first_timestamp, last_timestamp

                # If at start of file, no valid last line was found
                if seek_position == 0:
                    print(f"  No valid timestamp in last line of {log_file_path}.")
                    return None, None

    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred while processing {log_file_path}: {e}")
    return None, None

def calculate_log_duration(log_file_path):
    first_timestamp, last_timestamp = get_first_and_last_timestamp(log_file_path)
    if first_timestamp and last_timestamp:
        duration = last_timestamp - first_timestamp
        print(f"File: {log_file_path} | Duration: {duration}")
    else:
        print(f"File: {log_file_path} | Duration: Could not determine")

def process_folder(folder_path):
    # Recursively walk through all files in the directory and subdirectories
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".log"):
                file_path = os.path.join(root, file)
                calculate_log_duration(file_path)

# Check if the filename or folder was provided
if len(sys.argv) != 2:
    print("Usage: python calculate_duration.py <log_file_or_folder>")
    sys.exit(1)

# Get the path from command-line arguments
path = sys.argv[1]

# Determine if the path is a file or a folder
if os.path.isfile(path):
    calculate_log_duration(path)
elif os.path.isdir(path):
    print(f"Processing folder: {path}")
    process_folder(path)
else:
    print(f"Error: '{path}' is neither a file nor a directory.")