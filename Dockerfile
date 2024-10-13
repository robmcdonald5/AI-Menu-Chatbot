# Step 1: Use a lightweight Python base image
FROM python:3.12

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the requirements file into the container
COPY requirements.txt .

# Step 4: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of the application code into the container
COPY . .

# Step 6: Expose the port your application runs on (if applicable)
EXPOSE 5000

# Step 7: Run the application (assuming chat.py is the entry point)
CMD ["python", "chat.py"]
