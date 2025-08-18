# retail_demand_forecast
Corporación Favorita Sales Forecasting App
📌 Project Overview

This project tackles retail demand forecasting for Corporación Favorita, a major Ecuadorian grocery retailer. The focus is on predicting daily unit sales for stores located in Guayas province.

The interactive Streamlit app allows demand planners to:

Select a store–item combination

Explore historical sales data

Generate demand forecasts using a trained model



⚙️ Model Choice & Performance

Two models were considered and both are available in the app:

XGBoost – selected as the best model during Sprint 3 based on validation results.

LSTM – also supported in the app and provides faster results at inference time.

The deployed app gives planners flexibility to use either model.

🏗️ Project Structure
retail_demand_forecast/
│
├── app/
│   ├── main.py          # Streamlit app (UI + interaction)
│   ├── config.py        # Paths, constants
│   └── __init__.py
│
├── model/
│   ├── model_utils.py   # Model loading, prediction logic
│   └── __init__.py
│
├── models/              # Trained models
│   ├── xgboost_model_0816.pkl
│   └── lstm_model_0816.h5
│
├── data/
│   ├── data_utils.py    # Data loading & preprocessing
│   ├── <input files exported from Colab>
│   └── __init__.py
│
├── notebooks/           # Colab notebooks (EDA, training, experiments)
│
├── mlflow_results/      # MLflow tracking folder (local store, empty in repo)
│
├── requirements.txt
└── README.md

🚀 How to Run

Clone this repo:

git clone https://github.com/C12H22O11-2024/retail_demand_forecast.git
cd retail_demand_forecast


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app locally:

streamlit run app/main.py


In the sidebar, select:

Store number

Item number

Forecast start date (Jan–Mar 2014)

Model type (XGBoost or LSTM)

View historical sales + forecast overlay.

Optionally, download the forecast as a CSV.

✅ Next Steps / Known Issues

Horizon slider is present but not functional — forecasts are generated only for a value of 1. Export forecasts as CSV files for further analysis can be deployed after this.

XGBoost was chosen as the best model in Sprint 3, but LSTM currently runs faster in the app, need to further look into this.

Streamlit integration is local-only; could be extended to a remote server.

Preprocessing can be further optimized for scalability.

Clarification of whether output is what was expected . There was some unclarity due to changinging requirements..