import os
import time  # Don't forget to import time if you're using sleep
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

class Database:
    def __init__(self, db_name):
        self.uri = os.getenv('MONGODB_URI')  # Load MongoDB URI from environment variables
        print(f"MongoDB URI: {self.uri}")  # Verify URI loading
        self.client = None
        self.db = None
        self.db_name = db_name

    def connect(self):
        for attempt in range(3):  # Retry 3 times
            try:
                print(f"Attempting to connect to MongoDB (Attempt {attempt + 1})")
                self.client = MongoClient(self.uri)
                self.db = self.client[self.db_name]
                print("MongoDB connection successful!")
                break
            except ServerSelectionTimeoutError as e:
                print(f"MongoDB connection failed on attempt {attempt + 1}: {e}")
                time.sleep(5)  # Wait before retrying
            except Exception as e:
                print(f"MongoDB connection failed: {e}")
        
        # Correctly check if self.db is not None after connection attempts
        if self.db is not None:
            print("MongoDB connection successful!")
            try:
                print(f"Databases: {self.client.list_database_names()}")
            except Exception as e:
                print(f"Failed to list databases: {e}")
        else:
            print("Failed to connect to MongoDB after 3 attempts")

    def get_db(self):
        return self.db