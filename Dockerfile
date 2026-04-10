FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Fix: Install pkg_resources first
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install setuptools-scm  # Add this line
RUN pip install -r requirements.txt

COPY main.py .
RUN mkdir -p uploads outputs

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
