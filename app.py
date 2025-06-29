
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import random
import os

# Set seed
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

# Judul App
st.title("📈 Prediksi Emisi Karbon Global Menggunakan LSTM")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_excel("Global_Carbon_Budget_2024_v1.0.xlsx", sheet_name="Global Carbon Budget")
    df = df.iloc[21:, [0, 1]]
    df.columns = ['Year', 'Fossil_Emissions']
    df = df.dropna()
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Fossil_Emissions'] = pd.to_numeric(df['Fossil_Emissions'], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    return df

df = load_data()

# Normalisasi
scaler = MinMaxScaler()
scaled_emissions = scaler.fit_transform(df[['Fossil_Emissions']])

# Membuat data sekuensial
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

window_size = 5
X, y = create_sequences(scaled_emissions, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Bangun dan latih model
@st.cache_resource
def build_and_train_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=100, batch_size=8, verbose=0)
    return model

model = build_and_train_model()

# Prediksi seluruh data historis
y_pred = model.predict(X)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_true_rescaled = scaler.inverse_transform(y)

# Plot data aktual vs prediksi
st.subheader("📊 Visualisasi Prediksi Historis")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['Year'][window_size:], y_true_rescaled, label='Aktual')
ax.plot(df['Year'][window_size:], y_pred_rescaled, label='Prediksi')
ax.set_xlabel("Tahun")
ax.set_ylabel("Emisi Karbon Fosil (GtC/tahun)")
ax.set_title("Prediksi Historis Emisi Karbon Global")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Forecasting
st.subheader("📅 Prediksi Masa Depan")

future_steps = st.slider("Jumlah Tahun ke Depan", 1, 20, 5)
last_sequence = scaled_emissions[-window_size:].reshape(1, window_size, 1)
future_predictions = []

for _ in range(future_steps):
    next_pred = model.predict(last_sequence)[0]
    future_predictions.append(next_pred)
    last_sequence = np.append(last_sequence[:, 1:, :], [[next_pred]], axis=1)

future_predictions_rescaled = scaler.inverse_transform(future_predictions)
future_years = np.arange(df['Year'].iloc[-1] + 1, df['Year'].iloc[-1] + 1 + future_steps)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(df['Year'], df['Fossil_Emissions'], label='Data Historis')
ax2.plot(future_years, future_predictions_rescaled, label='Prediksi Masa Depan')
ax2.set_xlabel("Tahun")
ax2.set_ylabel("Emisi Karbon Fosil (GtC/tahun)")
ax2.set_title("Forecasting Emisi Karbon Global")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)
