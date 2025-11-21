🧠 LSTM Model – Deep Learning Project
Project ini merupakan implementasi Long Short-Term Memory (LSTM), sebuah arsitektur Recurrent Neural Network (RNN) yang dirancang untuk mempelajari pola pada data berurutan seperti time series, teks, atau sequence data lainnya.
Model dilatih untuk melakukan prediksi berkelanjutan, klasifikasi, atau forecasting, tergantung dataset yang digunakan.

📌 Fitur Utama
Implementasi LSTM layer (single atau stacked)
Preprocessing otomatis (normalisasi, sequence windowing)
Train–validation–test split
Optimizer Adam + Loss sesuai tugas (MSE/CE)
Visualization loss & metric selama training
Prediksi dan evaluasi model
Menyimpan model dalam format .h5

📁 Struktur Folder
Struktur dasar project:
project/
│── dataset/
│     ├── weather_surabaya.csv.csv
│── model/
│     ├── lstm_model.h5
│── app.py
│── README.md
Folder `models/` dibuat otomatis saat training.

🚀 Cara Menjalankan Aplikasi
1. Install dependencies
Jika belum ada `requirements.txt`, install manual:
pip install streamlit numpy pandas matplotlib scikit-learn tensorflow
2. Jalankan aplikasi
streamlit run app.py
3. Pastikan dataset tersedia
Format minimal kolom:
Date, Temperature, Rainfall

🧩 Alur Proses
Load dataset
Normalisasi MinMaxScaler
Windowing sebanyak window_size hari (default 45)
LSTM training
Validasi (val_loss)
Evaluasi model : RMSE, MAE, R²
Forecast ke depan 1–30 hari
Visualisasi hasil

🛠 Teknologi yang Digunakan
Python
TensorFlow / Keras
Streamlit
NumPy
Pandas
scikit-learn
Matplotlib
