import os
import pandas as pd
#from data.data_utils import download_file
import gdown  # For downloading from Google Drive
import xgboost as xgb
import pickle
from keras.models import load_model as keras_load_model
from app.config import LSTM_FEATURES, XGBOOST_FEATURES, LSTM_MODEL_PATH, XGB_MODEL_PATH, GOOGLE_DRIVE_LINKS_MODELS, LOCAL_MODEL_PATHS, BASE_DIR
from data.data_utils import get_lstm_scalers

'''
def download_xgb_model(model_path, drive_link):
    """Downloads the model from Google Drive if it doesn't exist locally."""
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        gdown.download(drive_link, model_path, quiet=False)

def download_lstm_model(model_path, drive_link):
    """Downloads the model from Google Drive if it doesn't exist locally."""
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        gdown.download(drive_link, model_path, quiet=False)

def load_model(model_path=MODEL_PATH):
    """Downloads and loads a pre-trained XGBoost model."""
    # Define path to the model file
    model_file = f"{model_path}"
    print(model_file)
    # Download if not available
    #download_xgb_model(model_file, GOOGLE_DRIVE_LINKS_MODELS["xgboost_model"])
    download_xgb_model(model_file, LOCAL_DRIVE_LINKS_MODELS["xgboost_model"])
    
    # Load using pickle
    with open(model_path, 'rb') as f:
        xgboost_model = pickle.load(f)

    print("✅ Model loaded successfully!")
    return xgboost_model

    """# Load XGBoost model
    xgboost_model = xgb.XGBRegressor()
    xgboost_model.load_model(model_file)

    print("✅ Model loaded successfully!")
    return xgboost_model"""'''


# ----------- Download Functions -----------

def download_xgb_model(model_path=LOCAL_MODEL_PATHS["xgboost"], drive_link=GOOGLE_DRIVE_LINKS_MODELS["xgboost"]):
    """Download the XGBoost model if it doesn't exist locally."""
    if not os.path.exists(model_path):
        print("⬇️ Downloading XGBoost model from Google Drive...")
        gdown.download(drive_link, model_path, quiet=False)

#
#with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
#                tmp.write(requests.get(XGB_GDRIVE_PATH).content)
#                tmp_path = tmp.name
#            with open(tmp_path, "rb") as f:
#                st.session_state.model = pickle.load(f)
#
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
    #model_path = os.path.join(BASE_DIR, "models","xgboost_model_0816.pkl")
    #with open(model_path, 'rb') as f:
     #   xgboost_model = pickle.load(f)
    #return xgboost_model

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
        '''# Ensure input_data is 3D array (batch, seq_len, features)
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.values.reshape(1, input_data.shape[0], input_data.shape[1])
        pred = model.predict(input_data, verbose=0)
        # Return last timestep if sequence output, else flatten
        if pred.ndim == 3:  # (batch, seq_len, 1)
            pred = pred[:, -1, 0]
        else:
            pred = pred.ravel()'''
        '''# Assuming input_data is the latest look_back rows
        input_seq = input_data[feature_cols].values[-look_back:]  # shape: (look_back, num_features)
        input_seq_scaled = scaler_X.transform(input_seq)          # same scaler as training
        input_seq_scaled = input_seq_scaled.reshape(1, look_back, len(feature_cols))  # add batch dimension

        pred_scaled = st.session_state.model.predict(input_seq_scaled, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled).ravel()[0]  # inverse scale
        return pred'''
        '''# Ensure input_data is a 3D array of shape (1, look_back, num_features)
        if isinstance(input_data, pd.DataFrame):
            look_back = 7  # must match training
            X = input_data[LSTM_FEATURES].to_numpy()[-look_back:]
            X_scaled = st.session_state.scaler_X.transform(X)  # apply same scaler as training
            input_data = X_scaled.reshape(1, look_back, len(LSTM_FEATURES))
        pred_scaled = model.predict(input_data, verbose=0)
        pred = st.session_state.scaler_y.inverse_transform(pred_scaled)  # inverse scale
        return pred.ravel()'''

        

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict_working(model, input_data, model_type="xgboost"):
    if model_type == "xgboost":
        return model.predict(input_data)
    elif model_type == "lstm":
        return model.predict(input_data, verbose=0).ravel()
    else:
        raise ValueError("Unknown model type")