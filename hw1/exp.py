import pandas as pd
import os

csv_folder_path = "/home/pp24/pp24s104/NTHU-ParallelProgramming/hw1/nsys_reports"

all_data = pd.DataFrame()

for i in range(48): 
    file_path = os.path.join(csv_folder_path, f"rank_{i}.csv")
    rank_data = pd.read_csv(file_path, skiprows=2)
    rank_data['Total Time (s)'] = rank_data['Total Time (ns)'] / 1e9
    rank_data['Rank'] = i  
    all_data = pd.concat([all_data, rank_data], ignore_index=True)

average_times = all_data.groupby('Name')['Total Time (s)'].mean().reset_index()

print("各函數在各 rank 中的平均執行時間 (s):")
print(average_times)
