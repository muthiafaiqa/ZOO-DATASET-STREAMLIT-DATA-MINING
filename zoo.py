import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("🐾 Data Mining - Zoo Dataset (Versi Lokal)")

file_path = r"C:\Users\MSI PC\Downloads\tugas data mining 9\zoo.tab"
data = pd.read_csv(file_path, sep='\t')

# Hapus baris pertama kalau berisi 'string' atau 'meta'
if data.iloc[0, 0] == 'string':
    data = data.drop(index=0).reset_index(drop=True)

st.subheader("📋 Data Awal")
st.dataframe(data.head())

# Kolom fitur
fitur = ['name', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic',
         'predator', 'toothed', 'backbone', 'breathes', 'venomous',
         'fins', 'legs', 'tail', 'domestic', 'catsize']

# Pastikan kolom lengkap
missing_cols = [col for col in fitur if col not in data.columns]
if missing_cols:
    st.error(f"Kolom berikut tidak ditemukan di dataset: {missing_cols}")
else:
    # Pisahkan fitur dan target
    X = data[fitur]
    y = data['type']

    # Hapus kolom 'name'
    X = X.drop(columns=['name'])

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    st.subheader("✅ Informasi Dataset")
    st.write(f"Jumlah data total: {len(data)}")
    st.write(f"Data latih: {len(X_train)} | Data uji: {len(X_test)}")

    st.subheader("🎯 Target (type) - Label Encoding")
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    st.write(label_map)

    st.subheader("🧩 Nama Kolom Fitur")
    st.write(list(X.columns))

    st.success("Data siap digunakan untuk training model!")
