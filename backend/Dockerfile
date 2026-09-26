FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app ./app
COPY transit_common ./transit_common
COPY dashboard ./dashboard
COPY dataset ./dataset
ENV DATASET_DIR=/app/dataset DASHBOARD_DIR=/app/dashboard ML_URL=http://ml:8001 PYTHONPATH=/app
EXPOSE 8000 9201
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
