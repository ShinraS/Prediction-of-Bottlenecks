import streamlit as st
import numpy as np
import joblib
import os

# Indispensable pour lire les vieux modèles .h5
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import keras

@st.cache_resource
def load_resources():
    # On change l'extension ici en .h5
    model_path = 'models/model_multi_task.h5'
    
    # On charge en forçant le moteur H5
    model = keras.models.load_model(model_path, compile=False)
    
    le_act = joblib.load('models/le_act.joblib')
    scaler_time = joblib.load('models/scaler_time.joblib')
    X_test = np.load('models/X_test.npy')
    return model, le_act, scaler_time, X_test

model, le_act, scaler_time, X_test = load_resources()

st.title("🎲 Démo : Diagnostic Aléatoire de Goulots")

if st.button("🔄 Piocher un dossier au hasard et analyser"):
    idx = np.random.randint(0, len(X_test))
    sample_input = X_test[idx : idx + 1] 
    preds = model.predict(sample_input)
    pred_act_dist = preds[0]
    pred_time_norm = preds[1]
    idx_pred = np.argmax(pred_act_dist)
    nom_act_predite = le_act.inverse_transform([idx_pred])[0]
    
    time_log = scaler_time.inverse_transform(pred_time_norm.reshape(-1, 1))
    time_hours = max(0, np.expm1(time_log).flatten()[0])
    st.subheader(f"🕵️ Analyse du Dossier n°{idx}")
    
    with st.expander("📜 Historique des 5 dernières étapes (Input)", expanded=True):
        past_indices = sample_input[0, :, 0].astype(int)
        past_names = le_act.inverse_transform(past_indices)
        
        for i, name in enumerate(past_names, 1):
            st.write(f"Étape {i} : **{name}**")

    st.divider()
    col1, col2 = st.columns(2)
    SEUIL = 21.0 
    
    with col1:
        st.markdown("### 📍 Prochaine étape prédite")
        st.info(f"**{nom_act_predite}**")
        
    with col2:
        st.markdown("### ⏳ Délai estimé")
        if time_hours > SEUIL:
            st.error(f"**{time_hours:.2f} Heures**")
            st.warning("🚨 **ALERTE : GOULOT DÉTECTÉ**")
        else:
            st.success(f"**{time_hours:.2f} Heures**")
            st.write("✅ Flux normal")
    st.markdown("---")
    if time_hours > SEUIL:
        st.subheader("💡 Pourquoi ce diagnostic ?")
        st.write(f"L'intelligence artificielle a détecté que le passage vers l'activité **{nom_act_predite}** présente un risque élevé de ralentissement compte tenu de l'historique récent de ce dossier. Un dépassement du seuil de **21h** est critique pour le respect des SLA (Service Level Agreements) de la banque.")
    else:
        st.subheader("💡 Analyse de fluidité")
        if time_hours < 1.0:
            st.write(f"Bien que des répétitions puissent être présentes dans l'historique, l'IA prédit que la prochaine étape (**{nom_act_predite}**) sera traitée quasi-instantanément. Il s'agit d'une activité de routine qui ne constitue pas un point de blocage pour le processus.")
        else:
            st.write(f"Le délai prédit pour l'activité **{nom_act_predite}** est inférieur au seuil critique. Le dossier suit une trajectoire standard et ne nécessite pas d'intervention prioritaire pour le moment.")