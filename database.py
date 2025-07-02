import os
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import time
import socks  # Import PySocks
import socket
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Database:
    def __init__(self, db_name):
        self.uri = os.getenv('MONGODB_URI')  # Ensure this includes tls=true and the database name
        self.client = None
        self.db = None
        self.db_name = db_name

    def connect(self):
        for attempt in range(3):
            try:
                print(f"Attempting to connect to MongoDB (Attempt {attempt + 1})")

                # Set up SOCKS5 proxy if QUOTAGUARDSTATIC_SOCKS5_URL is set
                quotaguard_url = os.getenv('QUOTAGUARDSTATIC_SOCKS5_URL')
                if quotaguard_url:
                    parsed = urlparse(quotaguard_url)
                    if parsed.scheme != 'socks5':
                        raise ValueError(f"Unsupported proxy scheme {parsed.scheme}")
                    socks.setdefaultproxy(
                        socks.PROXY_TYPE_SOCKS5,
                        parsed.hostname,
                        parsed.port,
                        True,  # rdns: Set to True to resolve DNS names through the proxy
                        parsed.username,
                        parsed.password
                    )
                    socket.socket = socks.socksocket  # Monkey patch socket module

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

    def get_db(self):
        if self.db is None:
            self.connect()
        return self.db