# Use a slim, official Python base image compatible with the required architecture
FROM --platform=linux/amd64 python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your agent code into the container
COPY agents/ ./agents/

# Copy your main script and the trained model
COPY main.py .
COPY model.txt .

# Specify the command to run when the container starts
CMD ["python", "main.py"]