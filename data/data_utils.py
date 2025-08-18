# data/data_utils.py
import pandas as pd
import numpy as np
import os
import gdown
import pickle
import json
from app.config import DATA_PATH, GOOGLE_DRIVE_LINKS, STORES_JSON, ITEMS_JSON, LSTM_FEATURES, XGBOOST_FEATURES,SCALER_X_PATH, SCALER_Y_PATH


def get_allowed_ids():
    """Loads allowed store and item IDs from JSON files."""
    with open(STORES_JSON, "r") as f:
        stores_data = json.load(f)
    with open(ITEMS_JSON, "r") as f:
        items_data = json.load(f)
    return stores_data, items_data

def get_lstm_scalers():
    """Loads allowed store and item IDs from JSON files."""
    with open(SCALER_X_PATH, 'rb') as f:
        scaler_X = pickle.load(f)

    with open(SCALER_Y_PATH, 'rb') as f:
        scaler_y = pickle.load(f)

    return scaler_X, scaler_y

    #store_numbers = stores_data
    #item_numbers = items_data

    #return store_numbers, item_numbers

def download_file(file_path, url):
    """Downloads a file from Google Drive if it doesn't exist locally."""
    if not os.path.exists(file_path):
        gdown.download(url, file_path, quiet=False)
    else:
        print(f"{file_path} already exists.")


def load_data(data_path=DATA_PATH):
    """Downloads necessary data from Google Drive and loads CSV files into DataFrames."""

    # Define the paths for all the required data files
    files = {
        "stores": f"{data_path}stores.csv",  # Path for stores data
        "items": f"{data_path}items.csv",  # Path for items data
        "transactions": f"{data_path}transactions.csv",  # Path for transactions data
        "oil": f"{data_path}oil.csv",  # Path for oil prices data
        "holidays_events": f"{data_path}holidays_events.csv",  # Path for holidays and events data
        "train": f"{data_path}train.csv"  # Path for training data
    }

    # Download the files if they don't already exist locally
    for key, file_path in files.items():
        download_file(file_path, GOOGLE_DRIVE_LINKS[key])

    # Load each downloaded CSV file into a pandas DataFrame
    df_stores = pd.read_csv(files["stores"])  # Stores data
    df_items = pd.read_csv(files["items"])  # Items data
    df_transactions = pd.read_csv(files["transactions"])  # Transactions data
    df_oil = pd.read_csv(files["oil"])  # Oil prices data
    df_holidays = pd.read_csv(files["holidays_events"])  # Holidays and events data

    # Filter store numbers for the 'Guayas' state
    # Extract the unique store numbers from the 'Guayas' state in the stores dataframe
    store_ids = df_stores[df_stores['state'] == 'Guayas']['store_nbr'].unique()

    # Define the item families we want to filter: 'GROCERY I', 'BEVERAGES', 'CLEANING'
    item_families = ['GROCERY I', 'BEVERAGES', 'CLEANING']

    # Get item numbers that belong to the specified item families
    items_ids = df_items[df_items['family'].isin(item_families)]

    # Select data before April'14
    max_date = '2014-04-01'

    # Chunk size
    chunk_size = 10 ** 6

    # Create an empty list to store filtered chunks of data
    filtered_chunks = []

    # Loop through each chunk of data (for large dataset processing)
    for chunk in pd.read_csv(files["train"], chunksize=chunk_size, parse_dates=['date'], low_memory=False):
        # Filter the chunk based on store numbers, item numbers
        # Conditions:
        # - Store numbers should be in 'Guayas' state
        # - Item numbers should belong to the selected item families
        chunk_filtered = chunk[(chunk['store_nbr'].isin(store_ids))]
        chunk_filtered = chunk_filtered[(chunk_filtered['date'] < max_date)]
        chunk_filtered = chunk_filtered.merge(items_ids, on="item_nbr", how="inner")

        # Append the filtered chunk to the list of filtered chunks
        filtered_chunks.append(chunk_filtered)

        # Delete the chunk to free up memory (important for large datasets)
        del chunk

    # Combine all filtered chunks into a single DataFrame
    df_train = pd.concat(filtered_chunks, ignore_index=True)

    # Clean up the memory by deleting the list of filtered chunks
    del filtered_chunks
    if df_train.empty:
        return None

    # Return all the loaded DataFrames
    return  df_train, df_stores, df_items, df_transactions, df_oil, df_holidays


def preprocess_input_data(inp_store_nbr, inp_item_nbr, inp_date,
                          df_train, df_stores, df_items, df_transactions, df_oil, df_holidays,
                          model_type="xgboost"):
    full_feature_df = preprocess_data(
        store_id=inp_store_nbr,
        item_id=inp_item_nbr,
        date=inp_date,
        df_train=df_train,
        df_stores=df_stores,
        df_items=df_items,
        df_transactions=df_transactions,
        df_oil=df_oil,
        df_holidays=df_holidays
    )
    if full_feature_df.empty:
        return None
    return prepare_input_for_model(full_feature_df, model_type)
   
    
def prepare_input_for_model(full_feature_df, model_type):
    """
    Select columns and cast dtypes for model.
    Includes error checking for missing columns.
    """
    if full_feature_df.empty:
        return None
    if model_type == "xgboost":
        cols = XGBOOST_FEATURES
        missing = set(cols) - set(full_feature_df.columns)
        if missing:
            raise ValueError(f"Missing columns for XGBoost: {missing}")

        df_xgb = full_feature_df[cols].copy()
        # Cast categorical features
        for c in ['store_nbr','item_nbr','family']:
            if c in df_xgb.columns:
                df_xgb[c] = df_xgb[c].astype('category')
        return df_xgb

    elif model_type == "lstm":
        cols = LSTM_FEATURES
        missing = set(cols) - set(full_feature_df.columns)
        if missing:
            raise ValueError(f"Missing columns for LSTM: {missing}")

        X = full_feature_df[cols].to_numpy()
        X_3d = X.reshape(1, len(X), len(cols))  # adjust timesteps if needed
        return X_3d

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def update_window(last_input, pred):
    """
    Update the input features for XGBoost autoregressive prediction.
    
    Parameters:
    - last_input: pd.DataFrame or np.array, shape (1, num_features)
                  Features used for predicting the previous day.
    - pred: float or np.array, prediction from the previous step.
    
    Returns:
    - new_input: same shape as last_input, updated with the new prediction.
    """
    # Convert to DataFrame if needed
    if isinstance(last_input, np.ndarray):
        last_input = pd.DataFrame(last_input)
    
    new_input = last_input.copy()
    
    # Example for lag features (adjust to your actual column names)
    lag_cols = [col for col in new_input.columns if "lag_" in col]
    
    # Shift lags down by 1 day
    for i in range(len(lag_cols)-1, 0, -1):
        new_input[lag_cols[i]] = new_input[lag_cols[i-1]]
    
    # Set lag_1 to the prediction of previous day
    new_input[lag_cols[0]] = pred
    
    # Update rolling features if you have them (simplified example)
    rolling_cols = [col for col in new_input.columns if "rolling_" in col]
    for col in rolling_cols:
        new_input[col] = new_input[lag_cols].mean(axis=1)  # simple rolling mean
    
    # Calendar features usually stay the same, no update needed
    
    return new_input.values  # return as np.array for XGBoost


import pandas as pd

def preprocess_data(store_id, item_id, date,
                    df_train, df_stores, df_items, df_transactions, df_oil, df_holidays):
    """
    Preprocess and feature-engineer sales data for a given store/item/date.
    Returns a feature-engineered DataFrame row for that specific store-item-date.
    """

    # --- Copy input ---
    df_train = df_train.copy()

    # --- Handle missing values ---
    if 'onpromotion' in df_train.columns:
        df_train['onpromotion'] = df_train['onpromotion'].fillna(False).astype(bool)
    if 'dcoilwtico' in df_oil.columns:
        df_oil['dcoilwtico'] = df_oil['dcoilwtico'].bfill()

    # --- Handle negative sales ---
    df_train['unit_sales'] = df_train['unit_sales'].apply(lambda x: max(x, 0))

    # --- Outlier removal (Z-score > 5) ---
    mean_sales = df_train.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform('mean')
    std_sales = df_train.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform('std').replace(0,1).fillna(1)
    df_train['z_score'] = (df_train['unit_sales'] - mean_sales) / std_sales
    df_train = df_train[df_train['z_score'] <= 5]

    # --- Merge with other data sources ---
    # items
    df_train = df_train.drop(columns=['family','class','perishable'], errors='ignore')
    df_train = pd.merge(df_train, df_items, on='item_nbr', how='left')

    # holidays (Guayaquil only in your notebook)
    df_holidays['date'] = pd.to_datetime(df_holidays['date'])
    df_holidays_guayas = df_holidays[df_holidays['locale_name'] == 'Guayaquil']
    df_train = pd.merge(df_train, df_holidays_guayas, on='date', how='left')

    # oil
    df_oil['date'] = pd.to_datetime(df_oil['date'])
    df_oil = df_oil.drop_duplicates(subset='date', keep='first')
    df_train = pd.merge(df_train, df_oil, on='date', how='left')

    # stores
    df_stores = df_stores.rename(columns={"type": "store_type"})
    df_train = pd.merge(df_train, df_stores, on='store_nbr', how='left')

    # transactions
    df_transactions['date'] = pd.to_datetime(df_transactions['date'])
    df_train = pd.merge(df_train, df_transactions, on=['date','store_nbr'], how='left')

    # --- Fill missing dates with 0 sales ---
    df_train['date'] = pd.to_datetime(df_train['date'])
    min_date, max_date = df_train['date'].min(), df_train['date'].max()
    full_date_range = pd.DataFrame({'date': pd.date_range(min_date, max_date, freq='D')})
    store_item_combos = df_train[['store_nbr','item_nbr']].drop_duplicates()
    all_combos = store_item_combos.merge(full_date_range, how='cross')
    df_filled = all_combos.merge(df_train, on=['store_nbr','item_nbr','date'], how='left')
    df_filled['unit_sales'] = df_filled['unit_sales'].fillna(0)

    # --- Feature engineering ---
    df_filled["year"] = df_filled["date"].dt.year
    df_filled["month"] = df_filled["date"].dt.month
    df_filled["day"] = df_filled["date"].dt.day
    df_filled["day_of_week"] = df_filled["date"].dt.dayofweek
    df_filled["is_weekend"] = df_filled["day_of_week"].isin([5,6]).astype(int)
    df_filled["is_holiday"] = df_filled['type'] == 'Holiday'

    # Lag features
    for lag in [1,7,14,30]:
        df_filled[f"lag_{lag}"] = df_filled.groupby(['store_nbr','item_nbr'])['unit_sales'].shift(lag)

    # Rolling features
    df_filled["rolling_avg_7"] = df_filled.groupby(['store_nbr','item_nbr'])['unit_sales'].transform(lambda x: x.rolling(7,min_periods=1).mean())
    df_filled["rolling_stdv_7"] = df_filled.groupby(['store_nbr','item_nbr'])['unit_sales'].transform(lambda x: x.rolling(7,min_periods=1).std())

    # --- Cleanup ---
    drop_cols = ['onpromotion','z_score','id','perishable','class','family','type','locale','locale_name',
                 'description','transferred','state','city','cluster','store_type','transactions']
    df_filled = df_filled.drop(columns=[c for c in drop_cols if c in df_filled.columns])
    df_filled = df_filled.fillna(method='ffill').dropna()

    # --- Filter for store/item/date ---
    try:
        inp_date = pd.to_datetime(date)
    except Exception:
        raise ValueError(f"Date {date} is not valid.")

    df_filtered = df_filled[(df_filled['store_nbr']==store_id) & 
                            (df_filled['item_nbr']==item_id) &
                            (df_filled['date']==inp_date)]

    return df_filtered

