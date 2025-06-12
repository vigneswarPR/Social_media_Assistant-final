# Use a lightweight official Python image as the base
FROM python:3.10-slim-buster

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

# Create a working directory inside the container
WORKDIR ${APP_HOME}

# Install system dependencies
# ffmpeg is crucial for video processing (used by OpenCV and potentially for tasks.py)
# libgl1 and libsm6, libxrender1 are often needed for OpenCV to run correctly in a headless environment
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxrender1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the working directory
COPY . .

# Expose the port that Streamlit runs on (default is 8501)
EXPOSE 8501

# Command to run the Streamlit application
# The '--server.address 0.0.0.0' makes the app accessible from outside the container
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]