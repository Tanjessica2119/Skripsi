import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os

# Fungsi label warna slider
def get_color_label(value):
    if value <= 25:
        color = 'red'
        label = 'Low (Red)'
    elif value <= 50:
        color = 'orange'
        label = 'Medium (Orange)'
    elif value <= 75:
        color = 'yellow'
        label = 'High (Yellow)'
    else:
        color = 'lightgreen'
        label = 'Very High (Green)'
    return f"<span style='color:{color}; font-weight:bold'>⬤ {label}</span>"

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model dan encoder
knn = joblib.load('knn_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Load data
df = pd.read_csv('hero_data.csv')
df_raw = pd.read_csv('hero_data.csv')  # Simpan data asli untuk ambil role asli

# Judul
st.title("Hero Recommendation System")

# Input user
hero_role_input = st.selectbox("Select Hero Role", label_encoders['hero_role'].classes_)
hero_specially_input = st.selectbox("Select Hero Specialty", label_encoders['hero_specially'].classes_)

# Slider dengan label warna
hero_durability_input = st.slider("Hero Durability", 0, 100, 50)
st.markdown(get_color_label(hero_durability_input), unsafe_allow_html=True)

hero_offence_input = st.slider("Hero Offence", 0, 100, 50)
st.markdown(get_color_label(hero_offence_input), unsafe_allow_html=True)

hero_ability_input = st.slider("Hero Ability", 0, 100, 50)
st.markdown(get_color_label(hero_ability_input), unsafe_allow_html=True)

hero_difficulty_input = st.slider("Hero Difficulty", 0, 100, 50)
st.markdown(get_color_label(hero_difficulty_input), unsafe_allow_html=True)

# Tombol rekomendasi
if st.button("Recommend Heroes"):
    if all(val < 18 for val in [hero_durability_input, hero_offence_input, hero_ability_input, hero_difficulty_input]):
        st.warning("Input tidak valid. Harap isi minimal salah satu atribut di atas.")
    else:
        # Transform input
        hero_role_encoded = label_encoders['hero_role'].transform([hero_role_input])[0]
        hero_specially_encoded = label_encoders['hero_specially'].transform([hero_specially_input])[0]

        new_data = [[hero_role_encoded, hero_specially_encoded, hero_durability_input,
                     hero_offence_input, hero_ability_input, hero_difficulty_input]]

        distances, indices = knn.kneighbors(new_data)

        # Tampilkan hasil
        st.markdown("### Recommended Heroes:")
        for i, index in enumerate(indices[0]):
            try:
                hero_name = df['hero_name'].iloc[index]
                st.markdown(f"#### {i+1}. {hero_name}")

                # Gambar hero
                hero_img_path = f"images/hero/{hero_name}.png"
                if os.path.exists(hero_img_path):
                    st.image(hero_img_path, width=150)
                else:
                    st.warning(f"Gambar hero '{hero_name}' tidak ditemukan.")

                # Ambil role asli (bisa lebih dari 1)
                hero_row = df_raw[df_raw['hero_name'] == hero_name]
                if not hero_row.empty:
                    hero_roles = hero_row['hero_role'].values[0].split(',')
                    hero_roles = [r.strip() for r in hero_roles]
                else:
                    hero_roles = ["Unknown"]

                # Tampilkan icon dan nama role secara rapi
                for role in hero_roles:
                    icon_path = f"images/role/{role}.png"
                    cols = st.columns([0.1, 0.9])
                    with cols[0]:
                        if os.path.exists(icon_path):
                            st.image(icon_path, width=30)
                        else:
                            st.write("🛑")
                    with cols[1]:
                        st.markdown(f"<div style='padding-top:6px'>{role}</div>", unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"Gagal menampilkan hero ke-{i+1}: {str(e)}")

# End
if __name__ == "__main__":
    pass
