# Dockerfile: Secure container for Sigma Security
# Isolates bug-catching and spirit-taming with sigma discipline

FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "src/main.py"]
