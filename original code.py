import pandas as pd
import matplotlib.pyplot as plt

file_path = 'state data.xlsx' 
data = pd.read_excel(file_path)

sorted_data = data.sort_values(by='FoodInsecurityLowOrVLPercent', ascending=False)

top_10_high = sorted_data.head(10)
top_10_low = sorted_data.tail(10)

fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

axes[0].barh(top_10_high['state'], top_10_high['FoodInsecurityLowOrVLPercent'], color='red', alpha=0.7)
axes[0].set_title('Top 10 States with Highest Food Insecurity Percentages (Low or Very Low Security)')
axes[0].set_xlabel('Percentage (%)')
axes[0].set_ylabel('State')

axes[1].barh(top_10_low['state'], top_10_low['FoodInsecurityLowOrVLPercent'], color='green', alpha=0.7)
axes[1].set_title('Top 10 States with Lowest Food Insecurity Percentages (Low or Very Low Security)')
axes[1].set_xlabel('Percentage (%)')
axes[1].set_ylabel('State')

plt.tight_layout()
plt.show()
