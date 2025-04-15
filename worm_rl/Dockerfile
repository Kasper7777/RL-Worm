# Use an Ubuntu base image with Python
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    xvfb \
    patchelf \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a script to run with xvfb (for rendering support)
RUN echo '#!/bin/bash\nxvfb-run -s "-screen 0 1400x900x24" python3 "$@"' > /usr/local/bin/xvfb-run-python \
    && chmod +x /usr/local/bin/xvfb-run-python

# Default command
CMD ["xvfb-run-python", "train.py"] 