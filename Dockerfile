FROM tensorflow/tensorflow:2.12.0

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project files (including Save Model folder)
COPY . /app

# Expose port for FastAPI
EXPOSE 8000

# Start uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]