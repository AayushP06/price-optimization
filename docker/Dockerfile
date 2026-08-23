# Use official Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./backend/

# Copy the static HTML frontend (served by Flask at /)
COPY price-optimization-website.html ./price-optimization-website.html

# Copy stored_data.json if it exists (optional)
COPY stored_data.json* ./

# Expose Flask port
EXPOSE 5000

# Environment defaults (override via docker run -e or .env)
ENV PORT=5000
ENV DEBUG=False

# Run the Flask server
CMD ["python", "backend/app/main.py"]
