import numpy as np
import pandas as pd
import os

def _new_dataset(root_path, data_path, target='Total Energy(kWh)',
                new_predictions=None):
    # Load the previous dataset
    df_raw = pd.read_csv(os.path.join(root_path, data_path))
    date = 'date'
    if 'index' in df_raw.columns:
        date = 'index'
    # Ensure the 'date' column is in datetime format
    df_raw[date] = pd.to_datetime(df_raw[date])

    # Generate a new date range based on the existing dataset's last date
    last_date = df_raw[date].iloc[-1]
    l_new_predictions = 0
    if new_predictions is not None:
        l_new_predictions = len(new_predictions[0])
    new_dates = pd.date_range(start=last_date + pd.Timedelta(minutes=15), 
                               periods=l_new_predictions, 
                               freq='15min')


    # Create a new DataFrame for new predictions
    if new_predictions is not None:
        df_new = pd.DataFrame({
            date: new_dates,
        })
        
        new_cols = list(df_raw.columns)[1:]
        for i, col in enumerate(new_cols):
            pred = new_predictions[:, :, i][0]
            df_new[col] = np.where(pred < 0.00, 0, pred)

        # Combine the old and new datasets
        df_combined = pd.concat([df_raw, df_new], ignore_index=True)
    else:
        df_combined = df_raw
    # print(df_raw.shape, df_new.shape)
    new_data_path = data_path
    if '_v2' not in data_path:
        new_data_path = data_path.split('.')[0]+'_v2.csv'
    df_combined.to_csv(os.path.join(root_path, new_data_path), index=False)
    print('data saved ...')
    return df_combined