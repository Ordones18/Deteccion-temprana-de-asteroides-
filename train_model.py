import json
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

print("Cargando datos...")
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

print(f"Entrenando Random Forest sobre {len(X_train)} muestras...")
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced'
)
rf_model.fit(X_train_scaled, y_train)

print("Entrenando Red Neuronal...")
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
model_nn.fit(
    X_train_scaled, y_train,
    epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stop],
    verbose=0, class_weight=class_weights_dict
)

print("Evaluando modelos...")
X_all_scaled = scaler.transform(X)

y_prob_rf_all = rf_model.predict_proba(X_all_scaled)[:, 1]
y_prob_nn_all = model_nn.predict(X_all_scaled, verbose=0).ravel()

rf_pred = (y_prob_rf_all > 0.5).astype(int)
nn_pred = (y_prob_nn_all > 0.5).astype(int)

print(f"RF - Hazardous predictions: {rf_pred.sum()} | Recall: { (rf_pred[y==1].sum() / y.sum() * 100):.1f}%")
print(f"NN - Hazardous predictions: {nn_pred.sum()} | Recall: { (nn_pred[y==1].sum() / y.sum() * 100):.1f}%")

predictions = {}
for i in range(len(df)):
    predictions[i] = {
        'prob_rf': round(float(y_prob_rf_all[i]), 4),
        'prob_nn': round(float(y_prob_nn_all[i]), 4)
    }

with open('predictions.json', 'w') as f:
    json.dump(predictions, f)

import joblib
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
model_nn.save('nn_model.keras', save_format='keras')

print("Modelos guardados: rf_model.pkl, nn_model.keras, scaler.pkl")
print("Predicciones guardadas: predictions.json")
