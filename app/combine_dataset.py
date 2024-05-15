import os
import pandas as pd

folder_path = '/home/code/Desktop/MSC/Natural-Language-Processing/POS_luyha_project/Dataset/bukusu_pos'

dataframes = []

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        try:
            # Read CSV file
            df = pd.read_csv(file_path, header=None)
            num_cols = len(df.columns)
            
            if num_cols >= 2:
                df = df.iloc[:, :2]
                df.columns = ['WORD', 'SPEECH TAG']
                dataframes.append(df)
            else:
                print(f"Skipping file {file_path}: Insufficient columns")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    combined_csv_path = 'Dataset/Processed/combined_bukusu_data.csv'
    combined_df.to_csv(combined_csv_path, index=False)

    print(f"Combined data saved to: {combined_csv_path}")
else:
    print("No valid data to combine")
