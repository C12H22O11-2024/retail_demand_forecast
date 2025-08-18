import os
import pandas as pd
import gdown  # For downloading from Google Drive
import xgboost as xgb
import pickle
from keras.models import load_model as keras_load_model
from app.config import LSTM_FEATURES, GOOGLE_DRIVE_LINKS_MODELS, LOCAL_MODEL_PATHS
from data.data_utils import get_lstm_scalers

# ----------- Download Functions -----------

def download_xgb_model(model_path=LOCAL_MODEL_PATHS["xgboost"], drive_link=GOOGLE_DRIVE_LINKS_MODELS["xgboost"]):
    """Download the XGBoost model if it doesn't exist locally."""
    if not os.path.exists(model_path):
        print("⬇️ Downloading XGBoost model from Google Drive...")
        gdown.download(drive_link, model_path, quiet=False)


def download_lstm_model(model_path=LOCAL_MODEL_PATHS["lstm"], drive_link=GOOGLE_DRIVE_LINKS_MODELS["lstm"]):
    """Download the LSTM model if it doesn't exist locally."""
    if not os.path.exists(model_path):
        print("⬇️ Downloading LSTM model from Google Drive...")
        gdown.download(drive_link, model_path, quiet=False)


# ----------- Load Functions -----------

def load_xgb_model(source="local"):
    """Load the XGBoost model (pickle)."""
    model_path = LOCAL_MODEL_PATHS["xgboost"]
    if source == "gdrive":
        download_xgb_model(model_path)
    with open(model_path, "rb") as f:
        xgb_model = pickle.load(f)
    print("✅ XGBoost model loaded successfully!")
    return xgb_model


def load_lstm_model(source="local"):
    """Load the LSTM model (Keras)."""
    model_path = LOCAL_MODEL_PATHS["lstm"]
    if source == "gdrive":
        download_lstm_model(model_path)
    lstm_model = keras_load_model(model_path)
    print("✅ LSTM model loaded successfully!")
    return lstm_model

# ----------- Dispatcher -----------

def load_model(preferred="xgboost", source="local"):
    """
    Load model by type and source.
    
    preferred: "xgboost" or "lstm"
    source: "local" or "gdrive"
    """
    if preferred == "xgboost":
        return load_xgb_model(source)
    elif preferred == "lstm":
        return load_lstm_model(source)
    else:
        raise ValueError(f"❌ Unsupported model type: {preferred}")
    

def predict(model, input_data, model_type="xgboost", look_back=7):
    """
    Predict sales for a single timestep.

    Parameters:
    - model: trained model object (XGBoost or LSTM)
    - input_data: input features (numpy array or pd.DataFrame)
    - model_type: "xgboost" or "lstm"

    Returns:
    - prediction: numpy array with predicted value(s)
    """
    if model_type == "xgboost":
        # Ensure input_data is 2D array
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.values
        return model.predict(input_data)  # shape: (1,) or (n_samples,)

    elif model_type == "lstm":
        scaler_X, scaler_y = get_lstm_scalers()
        # Ensure input_data is a DataFrame
        if isinstance(input_data, pd.DataFrame):
            # Take last 'look_back' rows
            X = input_data[LSTM_FEATURES].to_numpy()[-look_back:]
            # Scale using preloaded scaler
            X_scaled = scaler_X.transform(X)
            # Reshape to 3D (batch, seq_len, features)
            input_data = X_scaled.reshape(1, look_back, len(LSTM_FEATURES))

        # Predict
        y_pred_scaled = model.predict(input_data, verbose=0)
        # Inverse transform to original scale
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        return y_pred.ravel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict_working(model, input_data, model_type="xgboost"):
    if model_type == "xgboost":
        return model.predict(input_data)
    elif model_type == "lstm":
        return model.predict(input_data, verbose=0).ravel()
    else:
        raise ValueError("Unknown model type")