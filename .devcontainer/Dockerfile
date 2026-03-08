FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Instalamos Python 3.10 y herramientas de compilación
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Instalamos PyTorch primero (Versión CPU)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copiamos el archivo de requisitos directamente a /app/
WORKDIR /app
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt