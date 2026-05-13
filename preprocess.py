import csv
import json
import math
import random
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

LEGEND = """
╔══════════════════════════════════════════════════════════╗
║  DETECCION TEMPRANA DE ASTEROIDES PELIGROSOS            ║
║  Pipeline: CSV → Coordenadas 3D → Modelo ML → JSON     ║
╚══════════════════════════════════════════════════════════╝
"""

def orbital_to_cartesian(a, e, i_deg, om_deg, w_deg, mean_anomaly):
    i = math.radians(i_deg)
    om = math.radians(om_deg)
    w = math.radians(w_deg)
    M = mean_anomaly
    E = M
    for _ in range(20):
        E_next = M + e * math.sin(E)
        if abs(E_next - E) < 1e-10:
            break
        E = E_next
    nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))
    r = a * (1 - e * e) / (1 + e * math.cos(nu))
    x_orb = r * math.cos(nu)
    y_orb = r * math.sin(nu)
    cos_om = math.cos(om); sin_om = math.sin(om)
    cos_w = math.cos(w); sin_w = math.sin(w)
    cos_i = math.cos(i); sin_i = math.sin(i)
    x = x_orb * (cos_om * cos_w - sin_om * sin_w * cos_i) + y_orb * (-cos_om * sin_w - sin_om * cos_w * cos_i)
    y = x_orb * (sin_om * cos_w + cos_om * sin_w * cos_i) + y_orb * (-sin_om * sin_w + cos_om * cos_w * cos_i)
    z = x_orb * (sin_w * sin_i) + y_orb * (cos_w * sin_i)
    return x, y, z

print(LEGEND)

# ====== STEP 1: Leer CSV y procesar datos ======
print("[1/5] Leyendo datos orbitales...")
rows = []
with open('asteroids_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)
print(f"  Registros totales: {len(rows)}")

# ====== STEP 2: Calcular posiciones 3D ======
print("[2/5] Calculando posiciones orbitales 3D...")
asteroids = []
for row in rows:
    try:
        name = row['full_name'].strip()
        H_str = row['H'].strip()
        moid_str = row['moid'].strip()
        pha = row['pha'].strip()
        if not H_str or not moid_str or pha not in ('Y', 'N'):
            continue
        H = float(H_str)
        moid = float(moid_str)
        a = float(row['a'])
        e = float(row['e'])
        i = float(row['i'])
        om = float(row['om'])
        w = float(row['w'])
        if any(math.isnan(v) for v in (a, e, i, om, w, H, moid)):
            continue
        mean_anomaly = random.uniform(0, 2 * math.pi)
        x, y, z = orbital_to_cartesian(a, e, i, om, w, mean_anomaly)
        if any(math.isnan(v) for v in (x, y, z)):
            continue
        if abs(x) > 8 or abs(y) > 8 or abs(z) > 6:
            continue
        asteroids.append({
            'name': name,
            'x': round(x, 6),
            'y': round(y, 6),
            'z': round(z, 6),
            'a': round(a, 4),
            'e': round(e, 4),
            'i': round(i, 2),
            'H': round(H, 2),
            'moid': round(moid, 6),
            'pha': 1 if pha == 'Y' else 0
        })
    except (ValueError, KeyError):
        continue

hazardous = sum(1 for a in asteroids if a['pha'] == 1)
safe = len(asteroids) - hazardous
print(f"  {len(asteroids)} asteroides procesados ({hazardous} peligrosos, {safe} seguros)")

# ====== STEP 3: Entrenar modelos ML ======
print("[3/5] Preprocesando datos para modelos ML...")
df = pd.read_csv('asteroids_data.csv')
cols_numeric = ['a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y', 'n_obs_used', 'H', 'moid', 'condition_code']
for col in cols_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df['pha_binary'] = df['pha'].map({'Y': 1, 'N': 0}).fillna(0).astype(int)
df = df.dropna(subset=['moid', 'H', 'a', 'e', 'i'])
imputer = SimpleImputer(strategy='median')
if cols_numeric:
    df[cols_numeric] = imputer.fit_transform(df[cols_numeric])

features = ['a', 'e', 'i', 'om', 'w', 'q', 'H', 'moid', 'condition_code', 'n_obs_used']
X = df[features]
y = df['pha_binary']
y_true_all = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"  Entrenando con {len(X_train)} muestras...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

print("  Entrenando Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

print("  Entrenando Red Neuronal...")
model_nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_nn.fit(X_train_scaled, y_train, epochs=50, batch_size=64,
             validation_split=0.2, callbacks=[early_stop], verbose=0,
             class_weight=class_weights_dict)

# ====== STEP 4: Predecir probabilidades ======
print("[4/5] Prediciendo probabilidades para todos los asteroides...")
X_all_scaled = scaler.transform(X)
y_prob_rf_all = rf_model.predict_proba(X_all_scaled)[:, 1]
y_prob_nn_all = model_nn.predict(X_all_scaled, verbose=0).ravel()

rf_pred = (y_prob_rf_all > 0.5).astype(int)
nn_pred = (y_prob_nn_all > 0.5).astype(int)

rf_recall = (rf_pred[y_true_all == 1].sum() / y_true_all.sum() * 100)
nn_recall = (nn_pred[y_true_all == 1].sum() / y_true_all.sum() * 100)
print(f"  RF - Predice {rf_pred.sum()} peligrosos | Recall: {rf_recall:.1f}%")
print(f"  NN - Predice {nn_pred.sum()} peligrosos | Recall: {nn_recall:.1f}%")

# ====== STEP 4.5: Emparejar predicciones con asteroides 3D ======
print("  Emparejando predicciones con datos 3D...")
prob_map = {}
for i in range(len(df)):
    prob_map[df['full_name'].iloc[i].strip()] = (
        round(float(y_prob_rf_all[i]), 4),
        round(float(y_prob_nn_all[i]), 4)
    )

matched = 0
for ast in asteroids:
    if ast['name'] in prob_map:
        ast['prob_rf'] = prob_map[ast['name']][0]
        ast['prob_nn'] = prob_map[ast['name']][1]
        matched += 1
    else:
        ast['prob_rf'] = 0.5
        ast['prob_nn'] = 0.5

print(f"  {matched}/{len(asteroids)} asteroides emparejados con predicciones")

# ====== STEP 5: Exportar JSON ======
print("[5/5] Exportando JSON...")
data = {
    'asteroids': asteroids,
    'total': len(asteroids),
    'hazardous': hazardous,
    'safe': safe,
    'model_info': {
        'rf_recall': round(rf_recall, 1),
        'nn_recall': round(nn_recall, 1),
        'rf_precision': round(float(rf_pred[y_true_all == 1].sum()) / max(float(rf_pred.sum()), 1) * 100, 1),
        'nn_precision': round(float(nn_pred[y_true_all == 1].sum()) / max(float(nn_pred.sum()), 1) * 100, 1)
    }
}

with open('asteroides_data.json', 'w') as f:
    json.dump(data, f)

file_size = os.path.getsize('asteroides_data.json')
print(f"\n  JSON exportado: {len(asteroids)} asteroides ({file_size/1024/1024:.1f} MB)")
print(f"  Ready! Abre http://localhost:8000/asteroides.html")
