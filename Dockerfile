FROM python:3.12-slim

WORKDIR /app

# freetds es la dependencia de sistema de pymssql
RUN apt-get update && apt-get install -y --no-install-recommends \
    freetds-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# batch_prediccion_jnc.py implementa el modelo validado en la tesis.
# batch_runner.py es el job de la prueba de concepto (Prophet sobre los SKUs
# de Perfil A, escala diaria); queda en el repositorio como referencia pero
# no se copia a la imagen ni se ejecuta.
COPY batch_prediccion_jnc.py .

CMD ["python", "batch_prediccion_jnc.py"]
