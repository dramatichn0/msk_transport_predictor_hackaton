FROM python:3.12-slim
WORKDIR /app
COPY ml_service/requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY ml_service/app ./app
COPY transit_common ./transit_common
COPY dataset ./dataset
# Keep a baked-in fallback so an old named Docker volume cannot hide the
# trained artifact while a container is being recreated.
COPY models /app/bundled-models
ENV DATASET_DIR=/app/dataset MODEL_DIR=/app/models PYTHONPATH=/app
EXPOSE 8001
CMD ["sh", "-c", "mkdir -p /app/models && if [ ! -f /app/models/delay_model.joblib ]; then cp -f /app/bundled-models/* /app/models/; fi && exec uvicorn app.main:app --host 0.0.0.0 --port 8001"]
