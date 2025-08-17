import sys
import os
# Prevent OpenMP/XGBoost thread deadlocks in Streamlit
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Optional: for older XGBoost versions
os.environ["XGBOOST_THREAD_LIMIT"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["KMP_WARNINGS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # disable GPU if present

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gc
import streamlit as st  # Import Streamlit for UI
import pandas as pd  # Data handling
import datetime  # For date handling
import matplotlib.pyplot as plt
import tensorflow as tf

from app.config import DATA_PATH
from data.data_utils import load_data, preprocess_input_data,get_allowed_ids, update_window  # Data functions
from model.model_utils import load_model, load_xgb_model, load_lstm_model, predict  # Model functions



# Initializations
#model_choice = "xgboost"

# Load data and model
@st.cache_data  # Cache to improve performance
def get_data():
    return load_data(DATA_PATH)



@st.cache_resource
def load_model(preferred: str, source: str):
    """Load model by type and source, cache per combination."""
    if preferred == "xgboost":
        return load_xgb_model(source)
    elif preferred == "lstm":
        return load_lstm_model(source)
    else:
        raise ValueError(f"❌ Unsupported model type: {preferred}")


def main():
    
    if "model" not in st.session_state:
        st.session_state.model = None

    # --- User selections ---
    model_choice = st.selectbox("Choose a model:", ["xgboost", "lstm"], index=0)
    source_choice = st.radio("Load model from:", ["local", "gdrive"], index=0)
    st.title(f"Corporación Favorita Sales Forecasting Using {model_choice.upper()} Model")

    if st.session_state.model is None:
        st.session_state.model = load_model(model_choice, source_choice)
        st.success(f"✅ Default {model_choice.upper()} loaded from {source_choice}")

    if st.button("Load Model"):
        # Clean up old model to avoid memory leaks
        if "model" in st.session_state and st.session_state.model is not None:
            del st.session_state.model
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass
            gc.collect()

        # Load new one from cache
        st.session_state.model = load_model(model_choice, source_choice)
        st.title(f"Corporación Favorita Sales Forecasting Using {model_choice.upper()} Model")
        st.success(f"✅ {model_choice.upper()} loaded from {source_choice}")

    # Load datasets and model
    try:
        #_stores, _items, _tx, _oil, _hol, df_train = load_data(FILE_IDS)
         df_train, df_stores, df_items, df_transactions, df_oil, df_holidays = get_data()
    except Exception as e:
        # If you see this: check Drive sharing/IDs; the 'train' file must allow "Anyone with the link can view".
        st.error(f"Failed to load data. Check Drive sharing and FILE_IDS. Details:\n{e}")
        st.stop()

    if df_train is None:
            st.warning("⚠️ Not enough data for this store/item/date. Please choose a different combination.")
    else:
        #st.success(f"✅ Loaded {model_choice.upper()} model from {source_choice}")
        # --- Load allowed store & item IDs from CSV ---
        store_ids, item_ids = get_allowed_ids()
     
        # Sidebar inputs
        st.sidebar.header("Select Inputs")

        # Store selection dropdown
        store_id = st.sidebar.selectbox("Select Store", store_ids)

        # Extract available items for the selected store and filter allowed items
        available_items = df_train[
            (df_train['store_nbr'] == store_id) & (df_train['item_nbr'].isin(item_ids))
            ]['item_nbr'].unique()

        # Item selection dropdown
        item_id = st.sidebar.selectbox("Select Item", available_items)

        # Date range selection
        default_date = datetime.date(2014, 1, 1)  # Default start date
        min_date = datetime.date(2014, 1, 1)  # Start date of the dataset
        max_date = datetime.date(2014, 3, 31)  # End date of the dataset
        date = st.sidebar.date_input("Select Date", value=default_date, min_value=min_date, max_value=max_date)

        horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 90, 1)

        # Filter for selected store and item
        historical_sales = df_train[
            (df_train['store_nbr'] == store_id) & 
            (df_train['item_nbr'] == item_id)
        ]

        # Ensure 'date' column is datetime
        historical_sales['date'] = pd.to_datetime(historical_sales['date'])

        # Determine start and end dates
        start_date = historical_sales['date'].min()  # earliest available
        end_date = pd.to_datetime("2013-12-31")     # fixed end date

        # Filter by date range
        historical_sales = historical_sales[
            (historical_sales['date'] >= start_date) & 
            (historical_sales['date'] <= end_date)
        ]

        # Ensure date column is set as index and sorted
        historical_sales = historical_sales.set_index("date").sort_index()

        # Ensure all dates are present and fill missing/negative sales with 0
        all_dates = pd.date_range(start=historical_sales.index.min(), end=end_date)
        historical_sales = historical_sales.reindex(all_dates)
        historical_sales["unit_sales"] = historical_sales["unit_sales"].clip(lower=0).fillna(0)

        # Display the line chart for unit sales
        st.subheader("Historical Sales Data")
        if not historical_sales.empty:
            st.line_chart(historical_sales["unit_sales"])
        else:
            st.write("No sales data available for the selected store and item.")

            # Predict button
        if st.sidebar.button("Get Forecast"):
            input_data = preprocess_input_data(store_id, item_id, date, df_train, df_stores, df_items, df_transactions, df_oil, df_holidays, model_choice)
            if input_data is None:
                st.warning("⚠️ Not enough data for this store/item/date. Please choose a different combination.")
            else:
                preds = []
                fcst_dates = []
                last_input = input_data.copy()

                for i in range(horizon):
                    pred = predict(st.session_state.model, input_data, model_choice)
                    preds.append(float(pred[0]))
                    fcst_dates.append(date + pd.Timedelta(days=i+1))

                    # Update last_input for autoregressive prediction (XGBoost)
                    if model_choice == "xgboost":
                        last_input = update_window(last_input, pred)  # Implement your shift/add logic

                # -------------------------------
                # Display Forecast
                # -------------------------------
                st.success(f"Predicted Sales for {date} using {model_choice.upper()}: {pred[0]:.2f}")
                st.success(f"Predicted sales for next {horizon} day(s).")

                fcst_df = pd.DataFrame({"date": fcst_dates, "prediction": preds}).set_index("date")

                # Plot overlay
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(historical_sales.index, historical_sales["unit_sales"], label="Actual")
                ax.plot(fcst_df.index, fcst_df["prediction"], marker="o", label="Forecast")
                ax.axvline(date, linestyle="--", color="grey", alpha=0.5)
                ax.set_xlabel("Date"); ax.set_ylabel("Unit Sales"); ax.legend()
                st.pyplot(fig)

                # CSV Download
                st.download_button(
                    "Download Forecast CSV",
                    fcst_df.to_csv(),
                    file_name=f"forecast_{store_id}_{item_id}_{date}.csv",
                    mime="text/csv"
                )




                #prediction = predict(st.session_state.model, input_data, model_choice)
                #st.success(f"Predicted Sales for {date}: {prediction[0]:.2f}")
        


def main_working():
    '''st.title("Corporación Favorita Sales Forecasting Using XGBoost Model")
    if "model" not in st.session_state:
        st.session_state.model = None

    # --- User selections ---
    model_choice = st.selectbox("Choose a model:", ["xgboost", "lstm"], index=0)
    source_choice = st.radio("Load model from:", ["local", "gdrive"], index=0)

    #if "model" not in st.session_state:
    #   st.session_state.model = None

    if st.session_state.model is None:
        st.session_state.model = load_model(model_choice, source_choice)
        st.success(f"✅ {model_choice.upper()} loaded from {source_choice}")

    if st.button("Load Model"):
        # Clean up old model to avoid memory leaks
        if "model" in st.session_state and st.session_state.model is not None:
            del st.session_state.model
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass
            gc.collect()

        # Load new one from cache
        st.session_state.model = load_model(model_choice, source_choice)
        st.success(f"✅ {model_choice.upper()} loaded from {source_choice}")

    # Load datasets and model
    try:
        #_stores, _items, _tx, _oil, _hol, df_train = load_data(FILE_IDS)
         df_train, df_stores, df_items, df_transactions, df_oil, df_holidays = get_data()
    except Exception as e:
        # If you see this: check Drive sharing/IDs; the 'train' file must allow "Anyone with the link can view".
        st.error(f"Failed to load data. Check Drive sharing and FILE_IDS. Details:\n{e}")
        st.stop()
    if df_train is None:
            st.warning("⚠️ Not enough data for this store/item/date. Please choose a different combination.")
    else:

        #model = get_model()
        #model = get_model(preferred=model_choice, source=source_choice)

        st.success(f"✅ Loaded {model_choice.upper()} model from {source_choice}")

        # Store and item numbers provided
        #store_numbers = [24, 28, 34, 51]  # Allowed store numbers
        #item_numbers = [257847, 315176, 1463862, 1463814, 1047679, 1074327]  # Allowed item numbers
        # --- Load allowed store & item IDs from CSV ---
        store_ids, item_ids = get_allowed_ids()

        st.write("Available store numbers:", store_ids)
        st.write("Available item numbers:", item_ids)

        # Sidebar inputs
        st.sidebar.header("Select Inputs")

        # Extract unique store numbers from df_train
        #store_ids = df_train[df_train['store_nbr'].isin(store_numbers)]['store_nbr'].unique()

        # Store selection dropdown
        store_id = st.sidebar.selectbox("Select Store", store_ids)

        # Extract available items for the selected store and filter allowed items
        available_items = df_train[
            (df_train['store_nbr'] == store_id) & (df_train['item_nbr'].isin(item_ids))
            ]['item_nbr'].unique()

        # Item selection dropdown
        item_id = st.sidebar.selectbox("Select Item", available_items)

        # Date range selection
        default_date = datetime.date(2014, 1, 1)  # Default start date
        min_date = datetime.date(2014, 1, 1)  # Start date of the dataset
        max_date = datetime.date(2014, 3, 31)  # End date of the dataset
        date = st.sidebar.date_input("Select Date", value=default_date, min_value=min_date, max_value=max_date)

        # Display historical sales trend
        historical_sales = df_train[
            (df_train["store_nbr"] == store_id) &
            (df_train["item_nbr"] == item_id) &
            (df_train["date"] >= "2013-01-02") &
            (df_train["date"] <= "2013-12-31")
            ]

        # Ensure date column is set as index and sorted
        historical_sales = historical_sales.set_index("date").sort_index()

        # Display the line chart for unit sales
        st.subheader("Historical Sales Data")
        if not historical_sales.empty:
            st.line_chart(historical_sales["unit_sales"])
        else:
            st.write("No sales data available for the selected store and item.")

            # Predict button
        if st.sidebar.button("Get Forecast"):
            input_data = preprocess_input_data(store_id, item_id, date, df_train, df_stores, df_items, df_transactions, df_oil, df_holidays, model_choice)
            if input_data is None:
                st.warning("⚠️ Not enough data for this store/item/date. Please choose a different combination.")
            else:
                prediction = predict(st.session_state.model, input_data, model_choice)
                st.success(f"Predicted Sales for {date}: {prediction[0]:.2f}")
        
        
    

    #df_train = get_df_train()'''



if __name__ == "__main__":
    main()

def junk():
    """
        # Load datasets and model
        try:
            #_stores, _items, _tx, _oil, _hol, df_train = load_data(FILE_IDS)
            _stores, _items, _transactions, _oil, _holidays, df_train = get_data()
        except Exception as e:
            # If you see this: check Drive sharing/IDs; the 'train' file must allow "Anyone with the link can view".
            st.error(f"Failed to load data. Check Drive sharing and FILE_IDS. Details:\n{e}")
            st.stop()
        
        #model = get_model()
        model = get_model(preferred=model_choice, source=source_choice)

        st.success(f"✅ Loaded {model_choice.upper()} model from {source_choice}")

        # Store and item numbers provided
        #store_numbers = [24, 28, 34, 51]  # Allowed store numbers
        #item_numbers = [257847, 315176, 1463862, 1463814, 1047679, 1074327]  # Allowed item numbers
        # --- Load allowed store & item IDs from CSV ---
        store_numbers, item_numbers = get_allowed_ids()

        st.write("Available store numbers:", store_numbers)
        st.write("Available item numbers:", item_numbers)

        # Sidebar inputs
        st.sidebar.header("Select Inputs")

        # Extract unique store numbers from df_train
        store_ids = df_train[df_train['store_nbr'].isin(store_numbers)]['store_nbr'].unique()

        # Store selection dropdown
        store_id = st.sidebar.selectbox("Select Store", store_ids)

        # Extract available items for the selected store and filter allowed items
        available_items = df_train[
            (df_train['store_nbr'] == store_id) & (df_train['item_nbr'].isin(item_numbers))
            ]['item_nbr'].unique()

        # Item selection dropdown
        item_id = st.sidebar.selectbox("Select Item", available_items)

        # Date range selection
        default_date = datetime.date(2014, 1, 1)  # Default start date
        min_date = datetime.date(2014, 1, 1)  # Start date of the dataset
        max_date = datetime.date(2014, 3, 31)  # End date of the dataset
        date = st.sidebar.date_input("Select Date", value=default_date, min_value=min_date, max_value=max_date)

        # Display historical sales trend
        historical_sales = df_train[
            (df_train["store_nbr"] == store_id) &
            (df_train["item_nbr"] == item_id) &
            (df_train["date"] >= "2013-01-16") &
            (df_train["date"] <= "2014-03-31")
            ]

        # Ensure date column is set as index and sorted
        historical_sales = historical_sales.set_index("date").sort_index()

        # Display the line chart for unit sales
        st.subheader("Historical Sales Data")
        if not historical_sales.empty:
            st.line_chart(historical_sales["unit_sales"])
        else:
            st.write("No sales data available for the selected store and item.")

            # Predict button
        if st.sidebar.button("Get Forecast"):
            input_data = preprocess_input_data(store_id, item_id, date, df_train)
            prediction = predict(model, input_data)
            st.success(f"Predicted Sales for {date}: {prediction[0]:.2f}")
            
            
            
            
            st.write("Press a button to load a specific model/source combination:")
    

    # --- 1️⃣ XGBoost + Local ---
    if st.button("Load XGBoost + Local"):
        if st.session_state.model is not None:
            st.write(f"session_state at click was: {st.session_state.model}")
            del st.session_state.model
            st.write(f"sessions state cleared to: {st.session_state.model}")
            
        try:
            with open(XGB_LOCAL_PATH, 'rb') as f:
                xgb_local = pickle.load(f)
            st.success("✅ XGBoost + Local loaded successfully")
        except Exception as e:
            st.error(f"❌ XGBoost + Local failed: {e}")

    # --- 2️⃣ XGBoost + GDrive ---
    if st.button("Load XGBoost + GDrive"):
        if st.session_state.model is not None:
            st.write(f"session_state at click was: {st.session_state.model}")
            del st.session_state.model
            st.write(f"sessions state cleared to: {st.session_state.model}")
        try:
            response = requests.get(XGB_GDRIVE_PATH)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name

            # Load the pickle model
            with open(tmp_path, "rb") as f:
                xgb_gdrive = pickle.load(f)
            st.success("✅ XGBoost + GDrive loaded successfully")
        except Exception as e:
            st.error(f"❌ XGBoost + GDrive failed: {e}")

    # --- 3️⃣ LSTM + Local ---
    if st.button("Load LSTM + Local"):
        if st.session_state.model is not None:
            st.write(f"session_state at click was: {st.session_state.model}")
            del st.session_state.model
            st.write(f"sessions state cleared to: {st.session_state.model}")
        try:
            lstm_local = load_model(LSTM_LOCAL_PATH)
            st.success("✅ LSTM + Local loaded successfully")
        except Exception as e:
            st.error(f"❌ LSTM + Local failed: {e}")

    # --- 4️⃣ LSTM + GDrive ---
    if st.button("Load LSTM + GDrive"):
        if st.session_state.model is not None:
            st.write(f"session_state at click was: {st.session_state.model}")
            del st.session_state.model
            st.write(f"sessions state cleared to: {st.session_state.model}")
        try:
            response = requests.get(LSTM_GDRIVE_PATH)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name

            # Load the Keras model
            lstm_gdrive = load_model(tmp_path)
            st.success("✅ LSTM + GDrive loaded successfully")
        except Exception as e:
            st.error(f"❌ LSTM + GDrive failed: {e}")



            
"""


def main_testing_buttons():
    """st.title("Corporación Favorita Sales Forecasting")

    if "model" not in st.session_state:
        st.session_state.model = None

    st.write("Press a button to load a specific model/source combination:")

    # --- 1️⃣ XGBoost + Local ---
    if st.button("Load XGBoost + Local"):
        if st.session_state.model is not None:
            del st.session_state.model
            gc.collect()
        try:
            with open(XGB_LOCAL_PATH, "rb") as f:
                st.session_state.model = pickle.load(f)
            st.success("✅ XGBoost + Local loaded successfully")
        except Exception as e:
            st.error(f"❌ XGBoost + Local failed: {e}")

    # --- 2️⃣ XGBoost + GDrive ---
    if st.button("Load XGBoost + GDrive"):
        if st.session_state.model is not None:
            del st.session_state.model
            gc.collect()
        try:
            import requests, tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
                tmp.write(requests.get(XGB_GDRIVE_PATH).content)
                tmp_path = tmp.name
            with open(tmp_path, "rb") as f:
                st.session_state.model = pickle.load(f)
            st.success("✅ XGBoost + GDrive loaded successfully")
        except Exception as e:
            st.error(f"❌ XGBoost + GDrive failed: {e}")

    # --- 3️⃣ LSTM + Local ---
    if st.button("Load LSTM + Local"):
        if st.session_state.model is not None:
            del st.session_state.model
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except ImportError:
                pass
            gc.collect()
        try:
            import tensorflow as tf   # lazy import
            st.session_state.model = tf.keras.models.load_model(LSTM_LOCAL_PATH)
            st.success("✅ LSTM + Local loaded successfully")
        except Exception as e:
            st.error(f"❌ LSTM + Local failed: {e}")

    # --- 4️⃣ LSTM + GDrive ---
    if st.button("Load LSTM + GDrive"):
        if st.session_state.model is not None:
            del st.session_state.model
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except ImportError:
                pass
            gc.collect()
        try:
            import requests, tempfile, tensorflow as tf
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
                tmp.write(requests.get(LSTM_GDRIVE_PATH).content)
                tmp_path = tmp.name
            st.session_state.model = tf.keras.models.load_model(tmp_path)
            st.success("✅ LSTM + GDrive loaded successfully")
        except Exception as e:
            st.error(f"❌ LSTM + GDrive failed: {e}")"""



    