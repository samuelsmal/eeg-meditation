# std

# 3p
import pandas as pd

# prj
from src import plots, helpers

if __name__ == "__main__":
    df = helpers.load_bandpower_all_epochs_all_recordings_df("meditation")
    print(df.head())
