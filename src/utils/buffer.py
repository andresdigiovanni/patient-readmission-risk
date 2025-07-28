from pathlib import Path

import pandas as pd


class DataBuffer:
    def __init__(self, file_name, buffer_size=100):
        self.buffer_file = Path("models", file_name)
        self.buffer_size = buffer_size
        self.buffer = None

        if self.buffer_file.exists():
            self.buffer = pd.read_csv(self.buffer_file)

    def append(self, data):
        if self.buffer is None:
            self.buffer = data
        else:
            self.buffer = pd.concat([self.buffer, data], ignore_index=True)
            self.buffer = self.buffer[-self.buffer_size :]

        self.buffer.to_csv(self.buffer_file, index=False)

        if len(self.buffer) >= self.buffer_size:
            return self.buffer  # Ready to check drift

        return None  # Not enough data yet
