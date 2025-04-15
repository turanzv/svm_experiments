import sys
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

def calculate_log_duration(log_file_path):
    try:
        with open(log_file_path, 'r') as log_file:
            # Read the first line
            first_line = log_file.readline()
            first_timestamp = parse_timestamp(first_line)

            if not first_timestamp:
                print("No valid timestamp found in the first line.")
                return

            # Read the last line efficiently by seeking to the end of the file
            log_file.seek(0, 2)  # Move to the end of the file
            buffer_size = 1024
            while True:
                position = log_file.tell()
                seek_position = max(position - buffer_size, 0)
                log_file.seek(seek_position)
                lines = log_file.readlines()

                if len(lines) >= 2:
                    last_line = lines[-1]
                    last_timestamp = parse_timestamp(last_line)
                    if last_timestamp:
                        break
                if seek_position == 0:
                    print("No valid timestamp found in the last line.")
                    return

            # Calculate the duration
            duration = last_timestamp - first_timestamp
            print(f"Duration: {duration}")
    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Check if the filename was provided
if len(sys.argv) != 2:
    print("Usage: python calculate_duration.py <log_file_path>")
    sys.exit(1)

# Get the filename from command-line arguments
log_file_path = sys.argv[1]
calculate_log_duration(log_file_path)