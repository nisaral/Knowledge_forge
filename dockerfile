# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Expose the port the app runs on (Gunicorn will use this)
EXPOSE 8000

# Set the command to run the app using a production server
# Ensure GEMINI_API_KEY is set as a Secret in your deployment environment
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]