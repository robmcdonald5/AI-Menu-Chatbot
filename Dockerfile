# Step 1: Use an official Python runtime as a base image
FROM python:3.10-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container
COPY . /app

# Step 4: Install system dependencies (you might need some basic tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 5: Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Step 6: Download necessary NLTK data (punkt tokenizer)
RUN python -m nltk.downloader punkt

# Step 7: Expose the port on which your application runs (if using a web interface)
EXPOSE 5000

# Step 8: Run the application
CMD ["python", "app.py"]  # Adjust this command to match your entry point
