FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy project files
COPY --chown=user . .

# Create necessary directories for models and data
RUN mkdir -p models/l1 models/l2 data/storage data/datasets

# Expose the port
EXPOSE 7860

# Command to run the application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "app:app"]
