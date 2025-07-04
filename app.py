import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="KI-Vorhersage für Lackrezepturen", layout="wide")
st.title("\U0001F3A8 KI-Vorhersage für Lackrezepturen")

# --- Datei-Upload ---
uploaded_file = st.file_uploader("\U0001F4C1 CSV-Datei hochladen (mit ; getrennt)", type=["csv"])
if uploaded_file is None:
    st.warning("Bitte lade eine CSV-Datei hoch.")
    st.stop()

# --- CSV einlesen mit Fehlerbehandlung ---
try:
    df = pd.read_csv(uploaded_file, sep=";", decimal=",", on_bad_lines='skip')
    st.success("\u2705 Datei erfolgreich geladen.")
except Exception as e:
    st.error(f"\u274C Fehler beim Einlesen der Datei: {e}")
    st.stop()

st.write("\U0001F9FE Gefundene Spalten:", df.columns.tolist())

# --- Viskositätskurven plotten ---
scherraten = [0.1, 0.209, 0.436, 1, 1.9, 3.28, 10, 17.3, 36.2, 53, 100, 329, 687, 1000, 3010]
scherraten_cols = [str(s).replace('.', ',') for s in scherraten]
vorhandene_cols = [c for c in scherraten_cols if c in df.columns]

if vorhandene_cols:
    st.subheader("\U0001F4C9 Gemessene Viskositätskurven")
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, row in df.iterrows():
        ax.plot(scherraten, row[vorhandene_cols].values, alpha=0.3, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel("Scherrate [1/s]")
    ax.set_ylabel("Viskosität")
    ax.set_title("Viskosität vs. Scherrate (gemessene Kurven)")
    st.pyplot(fig)

# --- Spaltenauswahl ---
alle_spalten = df.columns.tolist()
vorgeschlagene_rohstoffe = alle_spalten[:6]
vorgeschlagene_zielgroessen = alle_spalten[6:]

st.subheader("\U0001F527 Spaltenauswahl")
rohstoff_spalten = st.multiselect(
    "\U0001F9EA Wähle die Rohstoffspalten (Einflussgrößen)",
    options=alle_spalten,
    default=vorgeschlagene_rohstoffe
)

zielspalten_options = [s for s in alle_spalten if s not in rohstoff_spalten]
default_zielspalten = [s for s in vorgeschlagene_zielgroessen if s in zielspalten_options]

zielspalten = st.multiselect(
    "\U0001F3AF Wähle die Zielgrößen (Kennwerte)",
    options=zielspalten_options,
    default=default_zielspalten
)

if not rohstoff_spalten or not zielspalten:
    st.error("Bitte sowohl Rohstoff- als auch Zielspalten auswählen.")
    st.stop()

# --- Modelltraining vorbereiten ---
X = df[rohstoff_spalten].copy()
y = df[zielspalten].copy()

kategorisch = X.select_dtypes(include="object").columns.tolist()
numerisch = X.select_dtypes(exclude="object").columns.tolist()

X_encoded = pd.get_dummies(X)
df_encoded = X_encoded.copy()
df_encoded[y.columns] = y
df_encoded = df_encoded.dropna()

X_clean = df_encoded[X_encoded.columns]
y_clean = df_encoded[y.columns]

if X_clean.empty or y_clean.empty:
    st.error("\u274C Keine gültigen Daten zum Trainieren.")
    st.stop()

# --- Modelltraining ---
modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_clean, y_clean)

# --- Benutzer-Eingabeformular ---
st.sidebar.header("\U0001F527 Parameter anpassen")
user_input = {}

for col in numerisch:
    try:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        if min_val == max_val:
            user_input[col] = st.sidebar.number_input(col, value=mean_val)
        else:
            user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)
    except:
        continue

for col in kategorisch:
    options = sorted(df[col].dropna().unique())
    user_input[col] = st.sidebar.selectbox(col, options)

input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)

for col in X_clean.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_clean.columns]

# --- Vorhersage ---
prediction = modell.predict(input_encoded)[0]

st.subheader("\U0001F52E Vorhergesagte Zielgrößen")
for i, ziel in enumerate(zielspalten):
    st.metric(label=ziel, value=round(prediction[i], 2))

# --- Vorhersage Viskositätskurve ---
if all(col in zielspalten for col in vorhandene_cols):
    st.subheader("\U0001F4C8 Vorhergesagte Viskositätskurve")
    pred_dict = dict(zip(zielspalten, prediction))
    pred_visko = [pred_dict[col] for col in vorhandene_cols]

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(scherraten, pred_visko, marker='o', color='red', label='Vorhersage')
    ax2.set_xscale('log')
    ax2.set_xlabel("Scherrate [1/s]")
    ax2.set_ylabel("Viskosität")
    ax2.set_title("Vorhergesagte Viskositätskurve")
    ax2.legend()
    st.pyplot(fig2)

# --- Lokales Frage-Modul (Variante 1) ---
st.subheader("\U0001F9E0 Frage-Modul (lokal, einfache Logik)")
frage = st.text_input("\U0001F50D Stelle eine Frage zu Rohstoffen, Kosten oder Eigenschaften:")

if frage:
    antwort = ""
    lower_frage = frage.lower()

    bekannte_ziele = ['kratzschutz', 'glanz', 'kosten', 'viskosität']
    erkannte_ziele = [ziel for ziel in bekannte_ziele if ziel in lower_frage]

    bekannte_rohstoffe = [c.lower() for c in rohstoff_spalten]
    erkannte_rohstoffe = [r for r in bekannte_rohstoffe if r in lower_frage]

    if not erkannte_rohstoffe or not erkannte_ziele:
        antwort = "❓ Frage konnte nicht eindeutig interpretiert werden. Bitte nenne einen bekannten Rohstoff und eine Zielgröße."
    else:
        try:
            korrelations = df.corr(numeric_only=True)
            ziel_kandidat = erkannte_ziele[0].capitalize()
            rohstoff_kandidat = [r for r in df.columns if r.lower() == erkannte_rohstoffe[0]][0]

            if ziel_kandidat in korrelations.columns and rohstoff_kandidat in korrelations.index:
                korrelation = korrelations.loc[rohstoff_kandidat, ziel_kandidat]
                antwort = f"🔍 Die Korrelation zwischen **{rohstoff_kandidat}** und **{ziel_kandidat}** beträgt {korrelation:.2f}."
                if abs(korrelation) > 0.5:
                    antwort += " → Das ist ein starker Zusammenhang."
                elif abs(korrelation) > 0.3:
                    antwort += " → Das ist ein mäßiger Zusammenhang."
                else:
                    antwort += " → Der Zusammenhang ist eher schwach."
            else:
                antwort = f"⚠️ Leider liegen keine Korrelationsdaten für {rohstoff_kandidat} oder {ziel_kandidat} vor."
        except Exception as e:
            antwort = f"❌ Fehler bei der Analyse: {e}"

    st.markdown(antwort)
