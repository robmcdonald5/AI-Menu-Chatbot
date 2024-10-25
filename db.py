import os
import time
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

class Database:
    def __init__(self, db_name):
        self.uri = os.getenv('MONGODB_URI')  # Ensure this includes tls=true
        print("Attempting to connect to MongoDB...")
        self.client = None
        self.db = None
        self.db_name = db_name

    def connect(self):
        for attempt in range(3):
            try:
                print(f"Attempting to connect to MongoDB (Attempt {attempt + 1})")
                self.client = MongoClient(
                    self.uri,
                    serverSelectionTimeoutMS=30000,
                    socketTimeoutMS=30000,
                    tls=True  # Ensure TLS is enabled
                )
                self.db = self.client[self.db_name]
                # Force a connection to verify settings
                self.client.admin.command('ping')
                print("MongoDB connection successful!")
                break
            except ServerSelectionTimeoutError as e:
                print(f"MongoDB connection failed on attempt {attempt + 1}: {e}")
                time.sleep(5)
            except Exception as e:
                print(f"MongoDB connection failed: {e}")

        if self.db is not None:
            print("MongoDB connection confirmed.")
            try:
                print("Databases connected successfully.")
            except Exception as e:
                print(f"Failed to list databases: {e}")
        else:
            print("Failed to connect to MongoDB after 3 attempts")