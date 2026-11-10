import pandas as pd

df = pd.read_csv("deliveries_updated_ipl_upto_2025.csv")

print("----- First 5 rows -----")
print(df.head())

print("\n----- Shape of dataset -----")
print(df.shape)  # (rows, columns)

# Display column names
print("\n----- Column names -----")
print(df.columns)

# Check for missing values
print("\n----- Missing values per column -----")
print(df.isnull().sum())

# Display basic info about dataset
print("\n----- Dataset info -----")
print(df.info())

# Show summary statistics (numerical columns)
print("\n----- Statistical summary -----")
print(df.describe())

# Check how many unique matches or seasons
if 'match_id' in df.columns:
    print("\nTotal matches:", df['match_id'].nunique())

if 'season' in df.columns:
    print("Seasons:", df['season'].unique())
