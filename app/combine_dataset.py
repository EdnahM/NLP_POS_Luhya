import os
import csv

def process_luhya(folder_path):
    combined_data = []
    header_files = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter='\t') # my data had tab delimiter
                print(reader)
                # breakpoint()
                header = next(reader) 
                if header == ['WORD', 'SPEECH TAG']:
                    header_files.append(filename)
                else:
                    for row in reader:
                        if len(row) >= 1:
                            # word_pos = row[0].split('\t')  # check if your data has taab delimeter.
                            word_pos = row[0],row[1]
                            # breakpoint()
                            if len(word_pos) >= 2:
                                # breakpoint()
                                combined_data.append([word_pos[0], word_pos[1]])
                                # combined_data.append([word_pos])

    if combined_data:
        output_file = 'combined_dataset.csv'
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['word', 'pos'])  
            writer.writerows(combined_data)

<<<<<<< Updated upstream
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        try:
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
=======
        print(f"Combined dataset saved as {output_file}")
        print("Header files:", ', '.join(header_files))
    else:
        print("No data found matching the expected format.")
>>>>>>> Stashed changes


folder_path = '/home/code/Desktop/MSC/Natural-Language-Processing/POS_luyha_project/Dataset/bukusu_pos'
process_luhya(folder_path)