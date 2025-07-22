FROM python:3.10-slim

WORKDIR /app

# ====== ENTORNO ======
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=""

# ====== COPIA E INSTALACIÓN DE TENSORFLOW ======
COPY tensorflow/tensorflow-2.19.0-cp310-cp310-manylinux_2_17_x86_64.whl .
RUN pip install --no-cache-dir tensorflow-2.19.0-cp310-cp310-manylinux_2_17_x86_64.whl
RUN pip install --no-cache-dir tensorflow-cpu==2.19.0

# ====== DEPENDENCIAS ======
COPY requirements.txt .
RUN pip install --no-cache-dir --no-build-isolation --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Gunicorn es necesario para producción
RUN pip install gunicorn

# ====== ARCHIVOS DE APLICACIÓN ======
COPY Sentinel.py .
COPY modelo_sensores.keras .
COPY scaler_sensores.gz .

# ====== INICIO CON GUNICORN ======
CMD ["gunicorn", "-b", "0.0.0.0:5000", "Sentinel:app"]