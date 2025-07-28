# Use a consistent Python base image
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# --- FIX: Install system-level dependencies required by lightgbm ---
RUN apt-get update && apt-get install -y libgomp1

# Now, copy and install Python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of your application code
COPY . .

# Set the command to run when the container starts
CMD ["python", "main.py"]