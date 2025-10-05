# Start with official Python 3.11 image (creates base layer)
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies (creates dependency layer)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (creates application layer)
COPY server.py .

# Create directory for database
RUN mkdir -p ./data

# Expose port 8000 for the API
EXPOSE 8000

# Command to run when container starts
CMD ["python", "server.py"]
