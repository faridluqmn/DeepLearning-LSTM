# ==============================================================
# 🌦️ LSTM Weather Forecasting – Surabaya (Multivariate Version)
# Dataset: weather_surabaya.csv
# Fitur:
#  - Input: Temperature & Rainfall
#  - Target: Pilih salah satu (Temperature / Rainfall)
#  - Prediksi jangka pendek 1–30 hari
#  - Training, evaluasi, dan forecasting via Streamlit UI
# ==============================================================

import os
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --------------------------------------------------------------
# Konfigurasi Streamlit
# --------------------------------------------------------------
st.set_page_config(page_title="LSTM Cuaca Surabaya", layout="wide")

# --------------------------------------------------------------
# Fungsi bantu
# --------------------------------------------------------------
def create_sequences_multivariate(data, window_size, target_index):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i, target_index])
    return np.array(X), np.array(y)

def prepare_data(df, target_col, window_size, train_ratio=0.8):
    features = ["Temperature", "Rainfall"]
    data = df[features].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    target_index = features.index(target_col)
    X, y = create_sequences_multivariate(data_scaled, window_size, target_index)
    dates_seq = df["Date"].iloc[window_size:].reset_index(drop=True)

    split_idx = int(len(X) * train_ratio)
    return (
        scaler,
        X[:split_idx], X[split_idx:],
        y[:split_idx], y[split_idx:],
        dates_seq[split_idx:], data_scaled
    )

def get_latest_model_path(models_dir="models"):
    if not os.path.exists(models_dir):
        return None
    files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith(".keras")]
    if not files:
        return None
    return max(files, key=os.path.getctime)

# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------
st.sidebar.title("⚙️ Konfigurasi")

dataset_path = st.sidebar.text_input("Nama file dataset", value="weather_surabaya.csv")
target_col = st.sidebar.selectbox("Pilih variabel target", ["Temperature", "Rainfall"], index=0)
window_size = st.sidebar.slider("Window Size (hari ke belakang)", 5, 90, 45)
epochs = st.sidebar.slider("Epochs", 10, 200, 50)
batch_size = st.sidebar.slider("Batch Size", 8, 128, 32)
forecast_days = st.sidebar.slider("Prediksi ke depan (hari)", 1, 30, 15)

st.title("🌦️ LSTM Cuaca Surabaya – Prediksi Suhu & Curah Hujan")

# --------------------------------------------------------------
# Load dataset
# --------------------------------------------------------------
if not os.path.exists(dataset_path):
    st.error(f"File dataset '{dataset_path}' tidak ditemukan.")
    st.stop()

df = pd.read_csv(dataset_path)
if "Date" not in df.columns:
    st.error("Dataset harus memiliki kolom 'Date'.")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"])

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🧠 Training", "📈 Evaluasi", "🔮 Prediksi ke Depan"])

# =======================
# TAB 1 – DATA
# =======================
with tab1:
    st.subheader("📊 Ringkasan Dataset Cuaca – Surabaya")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Jumlah Data", len(df))
        st.metric("Periode", f"{df['Date'].min().date()} ➜ {df['Date'].max().date()}")
    with col_b:
        st.metric("Rata-rata Suhu (°C)", f"{df['Temperature'].mean():.2f}")
        st.metric("Rata-rata Hujan (mm)", f"{df['Rainfall'].mean():.2f}")

    st.dataframe(df.head(), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df["Date"], df["Temperature"], label="Temperature", color="orange")
    ax2 = ax.twinx()
    ax2.plot(df["Date"], df["Rainfall"], label="Rainfall", color="blue", alpha=0.6)
    ax.set_title("Tren Cuaca Harian Surabaya")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    st.pyplot(fig, clear_figure=True)

# =======================
# TAB 2 – TRAINING
# =======================
with tab2:
    st.subheader("🧠 Training Model LSTM")

    scaler, X_train, X_test, y_train, y_test, _, _ = prepare_data(df, target_col, window_size)

    if st.button("🚀 Mulai Training Model"):
        with st.spinner("Sedang melatih model..."):
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(window_size, X_train.shape[2])),
                Dropout(0.2),
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                LSTM(32),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")

            os.makedirs("models", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            model_path = f"models/lstm_weather_{target_col}_{timestamp}.keras"

            checkpoint_cb = ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss", mode="min", verbose=1)
            earlystop_cb = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)

            t0 = time.time()
            history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs,
                                batch_size=batch_size, callbacks=[checkpoint_cb, earlystop_cb], verbose=1)
            t1 = time.time()

        st.success(f"Training selesai dalam {round(t1 - t0, 2)} detik")
        st.info(f"Model tersimpan di: `{model_path}`")

        st.session_state["latest_model_path"] = model_path
        st.session_state["latest_target"] = target_col
        st.session_state["latest_window_size"] = window_size

        fig_loss, ax_loss = plt.subplots(figsize=(7, 3))
        ax_loss.plot(history.history["loss"], label="Train Loss")
        ax_loss.plot(history.history["val_loss"], label="Val Loss")
        ax_loss.set_title("Training vs Validation Loss")
        ax_loss.legend()
        st.pyplot(fig_loss, clear_figure=True)

# =======================
# TAB 3 – EVALUASI
# =======================
with tab3:
    st.subheader("📈 Evaluasi Model")

    latest_model = st.session_state.get("latest_model_path")
    if not latest_model or not os.path.exists(latest_model):
        st.warning("Belum ada model tersimpan.")
    else:
        trained_target = st.session_state["latest_target"]
        trained_ws = st.session_state["latest_window_size"]

        scaler_eval, X_train_e, X_test_e, y_train_e, y_test_e, dates_test, _ = prepare_data(df, trained_target, trained_ws)
        model = load_model(latest_model)
        y_pred_scaled = model.predict(X_test_e)
        y_pred = scaler_eval.inverse_transform(np.repeat(y_pred_scaled, 2, axis=-1))[:, 0]
        y_true = scaler_eval.inverse_transform(np.repeat(y_test_e.reshape(-1, 1), 2, axis=-1))[:, 0]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        st.success("Evaluasi selesai ✅")
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{rmse:.3f}")
        col2.metric("MAE", f"{mae:.3f}")
        col3.metric("R²", f"{r2:.3f}")

        fig_eval, ax_eval = plt.subplots(figsize=(10, 4))
        ax_eval.plot(dates_test, y_true, label="Aktual", color="blue")
        ax_eval.plot(dates_test, y_pred, label="Prediksi", color="red", linestyle="dashed")
        ax_eval.legend()
        st.pyplot(fig_eval, clear_figure=True)

# =======================
# TAB 4 – FORECASTING
# =======================
with tab4:
    st.subheader("🔮 Prediksi ke Depan")

    latest_model = st.session_state.get("latest_model_path")
    if not latest_model or not os.path.exists(latest_model):
        st.warning("Latih model dulu di tab 'Training'.")
    else:
        trained_target = st.session_state["latest_target"]
        trained_ws = st.session_state["latest_window_size"]

        scaler_fc, X_train_fc, X_test_fc, y_train_fc, y_test_fc, _, data_scaled = prepare_data(df, trained_target, trained_ws)
        model = load_model(latest_model)

        seq = data_scaled[-trained_ws:]
        preds = []
        for _ in range(forecast_days):
            pred = model.predict(seq.reshape(1, trained_ws, 2), verbose=0)
            preds.append(pred[0, 0])
            new_entry = np.array([pred[0, 0], seq[-1, 1]])  # suhu/hujan terakhir ikut update
            seq = np.vstack([seq[1:], new_entry])

        preds = np.array(preds).reshape(-1, 1)
        future_full = np.concatenate([preds, np.zeros_like(preds)], axis=1)
        preds_inv = scaler_fc.inverse_transform(future_full)[:, 0]

        last_date = df["Date"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            f"Predicted_{trained_target}": preds_inv
        })

        st.markdown(f"#### Prediksi {forecast_days} Hari ke Depan ({trained_target})")
        st.dataframe(forecast_df.style.format({f"Predicted_{trained_target}": "{:.2f}"}))

        fig_f, ax_f = plt.subplots(figsize=(10, 4))
        ax_f.plot(df["Date"], df[trained_target], label="Data Historis", color="blue")
        ax_f.plot(forecast_df["Date"], forecast_df[f"Predicted_{trained_target}"], label="Prediksi", color="red", marker="o")
        ax_f.legend()
        st.pyplot(fig_f, clear_figure=True)