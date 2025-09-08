import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Extended sample dataset
data = {
    'Region': ['North', 'North', 'North', 'South', 'South', 'South', 'East', 'East', 'East', 'West', 'West', 'West'],
    'Frequency': ['Monthly'] * 12,
    'Date': pd.date_range(start='2020-01-01', periods=12, freq='M'),
    'Estimated_Unemployment': [5.0, 6.2, 14.5, 4.8, 6.5, 13.0, 5.5, 7.0, 15.0, 4.0, 5.8, 12.5],  # in %
    'Estimated_Employment': [95000, 93000, 85000, 96000, 92000, 84000, 94000, 90000, 83000, 97000, 94000, 85000],
    'Estimated_Labour_Force': [100000, 99300, 99500, 100800, 98500, 96700, 99500, 96800, 97600, 101000, 99600, 97000],
    'Area': ['Urban', 'Urban', 'Urban', 'Rural', 'Rural', 'Rural', 'Urban', 'Urban', 'Urban', 'Rural', 'Rural', 'Rural']
}

df = pd.DataFrame(data)

# Calculate unemployment rate from estimated data (for verification)
df['Unemployment_Rate_Calc'] = (df['Estimated_Unemployment'])  # Given directly in %

# Preview data
print(df.head())

# Plot unemployment rate over time by Region
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='Date', y='Estimated_Unemployment', hue='Region', marker='o')
plt.title('Estimated Unemployment Rate Over Time by Region')
plt.ylabel('Unemployment Rate (%)')
plt.xlabel('Date')
plt.grid(True)
plt.show()

# Plot Estimated Employment and Labour Force over time (aggregate)
plt.figure(figsize=(12,6))
df_grouped = df.groupby('Date').sum()[['Estimated_Employment', 'Estimated_Labour_Force']]
df_grouped.plot(marker='o')
plt.title('Estimated Employment and Labour Force Over Time (Aggregated)')
plt.ylabel('Number of People')
plt.xlabel('Date')
plt.grid(True)
plt.show()

# Boxplot of unemployment rate by Area (Urban vs Rural)
plt.figure(figsize=(8,5))
sns.boxplot(x='Area', y='Estimated_Unemployment', data=df)
plt.title('Distribution of Unemployment Rate by Area')
plt.ylabel('Unemployment Rate (%)')
plt.show()

# Correlation heatmap for numeric fields
plt.figure(figsize=(6,5))
sns.heatmap(df[['Estimated_Unemployment', 'Estimated_Employment', 'Estimated_Labour_Force']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Employment Data')
plt.show()

