import pandas as pd
import glob
import os

# Path to your directory
path = "/Users/darhanmashirapov/Desktop/RA-ship Denis/1T- Whole/1T cleaned_dup_Aigerim/"

# Find all .dta files
files = glob.glob(os.path.join(path, "*.dta"))

dfs = []
for file in files:
    df = pd.read_stata(file)

    # Extract year from filename (between "1t_" and "_cleaned")
    fname = os.path.basename(file)
    year = int(fname.split("_")[1])  # "2012", "2013", ...
    df["year"] = year

    dfs.append(df)

# Merge all datasets, aligning on all columns
merged = pd.concat(dfs, ignore_index=True)

# Save merged dataset
merged.to_stata(os.path.join(path, "merged_dataset.dta"), write_index=False)
