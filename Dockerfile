# Base image with Python and CUDA support
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the code into the container
COPY object_detection.py .

# Run the code when the container starts
CMD ["python3", "object_detection.py"]
