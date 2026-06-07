# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os # Import os module for path manipulation

# Load Dataset
# The dataset was downloaded to 'path' which is '/root/.cache/kagglehub/datasets/gokulrajkmv/unemployment-in-india/versions/5'
# The actual file name is likely 'Unemployment_Rate_upto_11_2020.csv'
file_name = "Unemployment_Rate_upto_11_2020.csv" # Assuming this is the correct filename inside the downloaded directory
full_file_path = os.path.join(path, file_name)
df = pd.read_csv(full_file_path)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Display First 5 Rows
print("First 5 Rows of Dataset:")
print(df.head())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove Missing Values
df.dropna(inplace=True)

# Display Column Names
print("\nColumn Names:")
print(df.columns.tolist())

# Find actual date column
date_col = None
for col in df.columns:
    if "date" in col.lower():
        date_col = col
        break

if date_col is None:
    raise ValueError("Date column not found in dataset.")

# Convert Date Column
df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True) # Changed to dayfirst=True for correct parsing

# Remove invalid dates
df.dropna(subset=[date_col], inplace=True)

# Extract Month and Year
df['Month'] = df[date_col].dt.month
df['Year'] = df[date_col].dt.year

# Find unemployment rate column
rate_col = None
for col in df.columns:
    if "unemployment rate" in col.lower():
        rate_col = col
        break

if rate_col is None:
    raise ValueError("Unemployment Rate column not found.")

# Find region column
region_col = None
for col in df.columns:
    if "region" in col.lower():
        region_col = col
        break

if region_col is None:
    raise ValueError("Region column not found.")

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Average Unemployment Rate
avg_rate = df[rate_col].mean()
print("\nAverage Unemployment Rate:", round(avg_rate, 2))

# Unemployment Rate by Region
region_data = df.groupby(region_col)[rate_col].mean()

print("\nAverage Unemployment Rate by Region:")
print(region_data)

# Plot Unemployment Rate by Region
plt.figure(figsize=(12, 6))
region_data.sort_values().plot(kind='bar')
plt.title("Average Unemployment Rate by Region")
plt.xlabel("Region")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Monthly Trend
monthly_data = df.groupby('Month')[rate_col].mean()

plt.figure(figsize=(10, 5))
monthly_data.plot(marker='o')
plt.title("Monthly Unemployment Trend")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.show()

# Yearly Trend
yearly_data = df.groupby('Year')[rate_col].mean()

plt.figure(figsize=(8, 5))
yearly_data.plot(marker='o')
plt.title("Yearly Unemployment Trend")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.show()

# Covid-19 Impact
covid_data = df[df['Year'] >= 2020]

plt.figure(figsize=(10, 5))
plt.plot(covid_data[date_col], covid_data[rate_col])
plt.title("Covid-19 Impact on Unemployment")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.show()

# Highest Unemployment Region
highest_region = region_data.idxmax()
highest_value = region_data.max()

print("\nRegion with Highest Unemployment:")
print(f"{highest_region} - {highest_value:.2f}%")

# Save Cleaned Dataset
df.to_csv("Cleaned_Unemployment_Data.csv", index=False)

print("\nAnalysis Completed Successfully!")