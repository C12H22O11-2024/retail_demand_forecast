import os

# Project root (…/corporacion_favorita)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))



# --- MLflow store & model location ---
# Points MLflow at your local runs folder (copied from Colab/Drive).
# Must be an ABSOLUTE file URI (note the three slashes after file:).
MLFLOW_TRACKING_URI = "file:///Users/sugar/Documents/Github/retail_demand_forecast/mlflow_results"

# Model to load from MLflow. Add your own run_id here. We saw how to get it before.
MODEL_URI = "runs:/26a599ff4fe14c7fb552e9ed53534ecb/model"
"come back here"

# Directory paths for data and model files
DATA_PATH = os.path.join(os.getcwd(), "data/")

# JSONs for allowed IDs
STORES_JSON = os.path.join(DATA_PATH, "store_numbers.json")
ITEMS_JSON = os.path.join(DATA_PATH, "item_numbers.json")

# Pickles ofor lstm scaler
SCALER_X_PATH = os.path.join(DATA_PATH, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(DATA_PATH, "scaler_y.pkl")


XGB_MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model_0816.pkl")
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_model_0816.h5")
"""MODEL_FILES = {
    "xgboost_model": os.path.join(BASE_DIR, "models", "xgboost_model_0816.pkl"),
    "lstm_model": os.path.join(BASE_DIR, "models", "lstm_model_0816.h5")
}"""

#MODEL_PATH = os.path.join(BASE_DIR, "models","xgboost_model_0816.pkl")



#MODEL_PATH = "/Users/sugar/Documents/GitHub/retail_demand_forecast/"  # Path to the directory containing the model files
#MODEL_PATH = 'models/'  # Path to the directory containing the model files
#FEATURES_JSON = os.path.join(BASE_DIR, "models", "feature_cols.json")
XGBOOST_FEATURES = ['store_nbr','item_nbr','year','month','day','day_of_week',
                'lag_1','lag_7','lag_14','rolling_avg_7','rolling_stdv_7']
LSTM_FEATURES = ['lag_1','lag_7','rolling_avg_7','rolling_stdv_7','year','month','day','day_of_week']

# --- Model window length used in feature windowing ---
SEQ_LEN = 60

# Google Drive file IDs for each dataset
your_file_id_for_stores_csv = '1heKGso4BXMzi4PMY2A_Csb2m-KiuwRxu'  # ID for stores data CSV
your_file_id_for_items_csv = '1bTB2mV8WdK97zMgXK36hxSysThQ7yIvG'  # ID for items data CSV
your_file_id_for_transactions_csv = '1-2i5naktdTZn-EwEfurCQ-qjK1unAGPc'  # ID for transactions data CSV
your_file_id_for_oil_csv = '15JROHp1gVy9E5P_L_-oUUcx9xmMNQw6l'  # ID for oil prices data CSV
your_file_id_for_holidays_csv = '1yBky_3tA-oXUPC-2QNtDAKqk9bqPY16E'  # ID for holidays data CSV
your_file_id_for_train_csv = '1s4fhVpD1oEnixZSi1WtnexaYJJNjKyOc'  # ID for training data CSV

# Google Drive links for each dataset
GOOGLE_DRIVE_LINKS = {
    "stores": f"https://drive.google.com/uc?id={your_file_id_for_stores_csv}",  # Link for stores data
    "items": f"https://drive.google.com/uc?id={your_file_id_for_items_csv}",  # Link for items data
    "transactions": f"https://drive.google.com/uc?id={your_file_id_for_transactions_csv}",  # Link for transactions data
    "oil": f"https://drive.google.com/uc?id={your_file_id_for_oil_csv}",  # Link for oil prices data
    "holidays_events": f"https://drive.google.com/uc?id={your_file_id_for_holidays_csv}",  # Link for holidays data
    "train": f"https://drive.google.com/uc?id={your_file_id_for_train_csv}" # Link for training data
}

# Google Drive link for the model
your_file_id_for_xgboost_model = '1ctgpors0e_zbHSL_LZ0KyGEyyI_2zpbo'  # ID for the XGBoost model file
your_file_id_for_lstm_model = '1wpran1eQLIsNQkk3-GnYFtNdBDuxz1XH'  # ID for the LSTM model file



# Google Drive link for the model file
GOOGLE_DRIVE_LINKS_MODELS = {
    "xgboost": f"https://drive.google.com/uc?id={your_file_id_for_xgboost_model}",
    "lstm": f"https://drive.google.com/uc?id={your_file_id_for_lstm_model}"  # Link for the XGBoost model
}

 #Local Drive link for the model
local_file_id_for_xgboost_model = 'xgboost_model_0816.pkl'  # ID for the XGBoost model file
local_file_id_for_lstm_model = 'lstm_model_0816.h5'  # ID for the LSTM model file

# Local Drive link for the model file
LOCAL_MODEL_PATHS = {
    "xgboost": f"mlflow_results/models/{local_file_id_for_xgboost_model}",
    "lstm": f"mlflow_results/models/{local_file_id_for_lstm_model}"  # Link for the XGBoost model
}

