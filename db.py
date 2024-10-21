from dotenv import load_dotenv
import os
from pymongo.mongo_client import MongoClient

class Database:
    def __init__(self, db_name):
        load_dotenv()  # Load the .env file
        self.uri = os.getenv('MONGODB_URI')  # Correctly load the URI from the .env file
        print(f"MongoDB URI: {self.uri}")  # <-- Add this to verify that the URI is being loaded correctly
        self.client = None
        self.db = None
        self.db_name = db_name

    def connect(self):
        try:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            print("Connection successful!")
        except Exception as e:
            print("Connection failed:", e)

    def get_db(self):
        return self.db
