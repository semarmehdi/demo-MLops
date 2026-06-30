FROM python:3.11-slim

WORKDIR /app

# Logs en temps réel + pas de .pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Comportement "prod" par défaut : l'image tourne en continu toutes les 30 s
# et reste résiliente aux erreurs. La CI surcharge ces valeurs pour borner
# le nombre de cycles et faire échouer le build au premier incident.
ENV ETL_LOOP_ITERATIONS=0 \
    ETL_LOOP_INTERVAL_SECONDS=30 \
    ETL_FAIL_FAST=false

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app files
COPY etl.py .
COPY utils/ ./utils/

CMD ["python", "etl.py"]