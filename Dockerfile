# === Stage 1: Build React Frontend ===
FROM node:18-alpine AS frontend-builder

# Set working directory
WORKDIR /frontend

# Copy frontend dependency files
COPY frontend/package.json frontend/package-lock.json ./

# Install frontend dependencies
RUN npm install

# Copy frontend source code
COPY frontend/ ./

# Build the React app for production
RUN npm run build

# === Stage 2: Setup Python Backend ===
FROM python:3.12-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install SpaCy model
RUN python -m spacy download en_core_web_sm

# Install SentenceTransformer model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L6-v2')"

# Set environment variable for SpaCy data path
ENV SPACY_DATA=/usr/local/lib/python3.12/site-packages

# Create a non-root user for security
RUN addgroup --system appgroup && adduser --system appuser --ingroup appgroup

# Copy backend source code, including connect.py and db.py
COPY . .

# Copy the built frontend from the previous stage
COPY --from=frontend-builder /frontend/build ./frontend/build

# Change ownership to non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Set the PORT environment variable (default to 5000)
ENV PORT=5000
EXPOSE $PORT

# Health check (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:$PORT/ || exit 1

# Command to run the application using Waitress
CMD waitress-serve --port=$PORT chat:app