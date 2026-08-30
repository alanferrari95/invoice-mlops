FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY --from=builder /install/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /install/bin /usr/local/bin
COPY src/ /app/src/
COPY models/model.joblib /app/models/model.joblib

EXPOSE 8080
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
