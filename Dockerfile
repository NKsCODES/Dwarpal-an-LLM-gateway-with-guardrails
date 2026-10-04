FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv/gateway

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home gateway && chown -R gateway /srv/gateway
USER gateway

EXPOSE 8000 8501
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
