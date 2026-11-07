# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY src/api/requirements.txt ./src/api/

# Install Python dependencies
RUN pip install --no-cache-dir -r src/api/requirements.txt

# ✅ Copy complete project structure
COPY src/ ./src/
COPY models/ ./models/

# Create __init__.py files to make packages work
RUN touch src/__init__.py
RUN touch src/api/__init__.py

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ✅ Run with same command as local
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]