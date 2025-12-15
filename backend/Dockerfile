FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN wget "https://github.com/DYcoded/skin-ai-app/releases/download/v1.0.0/final_resnet_oversampled_weights.weights.h5" -O final_resnet_oversampled_weights.weights.h5

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python", "testapi.py"]