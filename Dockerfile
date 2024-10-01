# Step 1: Use a lightweight Python base image
FROM python:3.10-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the requirements file into the container
COPY requirements.txt .

# Step 4: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of the application code into the container
COPY . .

# Step 6: Download NLTK data (like 'punkt')
RUN python -m nltk.downloader punkt

# Step 7: Expose the port your application runs on (if applicable)
EXPOSE 5000

# Step 8: Run the application (assuming chat.py is the entry point)
CMD ["python", "chat.py"]
