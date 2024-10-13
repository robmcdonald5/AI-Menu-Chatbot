import os
from pymongo.mongo_client import MongoClient

class Database:
    def __init__(self, db_name):
        self.uri = os.environ.get('MONGODB_URI')
        self.client = None
        self.db = None
        self.db_name = db_name

    def connect(self):
        try:
            self.client = MongoClient(self.uri)
            self.db = self.client['ChipotleMenu']
            print("Connection successful!")
        except Exception as e:
            print("Connection failed", e)

    def get_db(self):
        return self.db
