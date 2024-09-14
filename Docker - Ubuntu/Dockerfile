# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data 
RUN python -m nltk.downloader punkt 
RUN python -m nltk.downloader punkt_tab 

# Make port 80 available to the world outside this container
EXPOSE 80

# Run chat.py when the container launches
CMD ["python", "chat.py"]
