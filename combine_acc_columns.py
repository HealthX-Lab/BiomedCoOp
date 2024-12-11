
import pandas as pd

# File paths
file_paths = [
    'BTMRI.csv',
    'BUSI.csv',
    'COVID_19.csv',
    'CTKidney.csv',
    'DermaMNIST.csv',
    'Kvasir.csv',
    'Kather_texture.csv',
    'LungColon.csv',
    'RETINA.csv',
    'KneeXray.csv',
    'OCTMNIST.csv'
]

# Extracting filenames for column names (without .csv extension)
filenames = [path.split('/')[-1].replace('.csv', '') for path in file_paths]

# Reading the CSV files and combining only the 'acc' column from each
for k in ["clipadapter","tip","tipF","lp","lp++"]:
    dfs = []
    for path, filename in zip(file_paths, filenames):
        df = pd.read_csv(f"output_{k}_biomedclip/"+path)
        if 'std' in df.columns:
            dfs.append(df[['std']].rename(columns={'std': f'{filename}_std'}))

    # Concatenating the 'acc' columns
    combined_std_df = pd.concat(dfs, axis=1)

    # Saving the combined dataframe to a new CSV file
    output_path = f'combined_std_datasets_{k}.csv'
    combined_std_df.to_csv(output_path, index=False)

    print(f'Combined acc columns saved to {output_path}')
