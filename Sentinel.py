import requests
import sseclient
import json
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import threading
import time
from flask import Flask, Response
import questionary
from questionary import Style
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ====== CONFIGURACIÓN ======
SSE_URL = os.getenv("SSE_URL", "http://host.docker.internal:8080/stream-csv")
MODEL_PATH = "modelo_sensores.keras"
SCALER_PATH = "scaler_sensores.gz"
LOOKBACK = 5
N_FUTURO = 2

# ====== CARGA DEL MODELO Y SCALER ======
scaler = joblib.load(SCALER_PATH)
model = load_model(MODEL_PATH)

all_features = scaler.feature_names_in_
sensor_cols = [col for col in all_features if not col.endswith('_delta')]

# ====== MENÚ DE SELECCIÓN DE SENSOR ======
custom_style = Style([
    ('pointer', 'fg:#00ff00 bold'),
    ('highlighted', 'fg:#0080ff bold'),
    ('selected', 'fg:#00ff00 bold'),
    ('question', 'bold'),
])

def seleccionar_sensor():
    return questionary.select(
        "Selecciona el sensor que quieres analizar:",
        choices=sensor_cols,
        style=custom_style
    ).ask()

SENSOR = seleccionar_sensor()

if SENSOR not in sensor_cols:
    raise ValueError(f"El sensor '{SENSOR}' no está entre los sensores válidos: {sensor_cols}")

sensor_idx = sensor_cols.index(SENSOR)

# ====== BUFFER Y PREDICCIONES ======
historico = []
predicciones_en_tiempo_real = []

def convertir_a_input(valores):
    entradas = []
    for i in range(len(valores)):
        fila = np.zeros(len(all_features))
        sensor_value = valores[i]
        delta = valores[i] - valores[i-1] if i > 0 else 0
        fila[all_features.tolist().index(SENSOR)] = sensor_value
        fila[all_features.tolist().index(f"{SENSOR}_delta")] = delta
        entradas.append(fila)
    return np.array(entradas)

def procesar_evento(event_data):
    global historico, predicciones_en_tiempo_real

    try:
        bloque = json.loads(event_data)
    except Exception as e:
        print("Error decodificando JSON:", e, flush=True)
        return

    valores_sensor = [d["sensor_dato"] for d in bloque if d["sensor_name"] == SENSOR]
    if not valores_sensor:
        print(f"No hay datos para el sensor {SENSOR}", flush=True)
        return

    nuevo_valor = float(valores_sensor[0])
    print(f"[{SENSOR}] Recibido: {nuevo_valor}", flush=True)
    historico.append(nuevo_valor)

    if len(historico) < LOOKBACK:
        return

    if len(historico) > LOOKBACK:
        historico = historico[-LOOKBACK:]

    entrada = convertir_a_input(historico)
    entrada_df = pd.DataFrame(entrada, columns=all_features)
    entrada_esc = scaler.transform(entrada_df).reshape(1, LOOKBACK, -1)

    pred = model.predict(entrada_esc, verbose=0)[0].reshape(N_FUTURO, -1)
    full_pred = np.zeros((N_FUTURO, len(all_features)))
    full_pred[:, all_features.tolist().index(SENSOR)] = pred[:, sensor_idx]
    pred_inv = scaler.inverse_transform(full_pred)

    resultado = {
        "predicciones": [
            {"paso": i+1, "valor": float(pred_inv[i, all_features.tolist().index(SENSOR)])}
            for i in range(N_FUTURO)
        ]
    }

    predicciones_en_tiempo_real.append(json.dumps(resultado))
    if len(predicciones_en_tiempo_real) > 10:
        predicciones_en_tiempo_real = predicciones_en_tiempo_real[-10:]

    print("Predicción enviada:", resultado, flush=True)

# ====== STREAMING SSE ======
app = Flask(__name__)

@app.route("/predicciones")
def sse():
    def event_stream():
        last_index = 0
        while True:
            if last_index < len(predicciones_en_tiempo_real):
                data = predicciones_en_tiempo_real[last_index]
                yield f"data: {data}\n\n"
                last_index += 1
            time.sleep(1)
    return Response(event_stream(), mimetype="text/event-stream")

# ====== HILO PARA RECIBIR EVENTOS Y HACER PREDICCIONES ======
def loop_sse():
    response = requests.get(SSE_URL, stream=True)
    client = sseclient.SSEClient(response)
    for event in client.events():
        procesar_evento(event.data)

def start_background_thread():
    hilo_pred = threading.Thread(target=loop_sse)
    hilo_pred.daemon = True
    hilo_pred.start()

# Llamar al inicio del hilo en importación (requerido para Gunicorn)
start_background_thread()
