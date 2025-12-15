FROM python:3.10-slim


RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY . /app

RUN git lfs pull

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "testapi.py"]