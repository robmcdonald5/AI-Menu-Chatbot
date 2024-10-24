import os
from pymongo import MongoClient
import socks
import socket
from urllib.parse import urlparse

class Database:
    def __init__(self, db_name):
        self.uri = os.getenv('MONGODB_URI')  # Load MongoDB URI from environment variables
        print(f"MongoDB URI: {self.uri}")  # Verify URI loading
        self.client = None
        self.db = None
        self.db_name = db_name

    def connect(self):
        try:
            # Check if PROXIMO_URL is set
            prox_url = os.getenv('PROXIMO_URL')
            if prox_url:
                parsed = urlparse(prox_url)
                proxy_host = parsed.hostname
                proxy_port = parsed.port
                proxy_username = parsed.username
                proxy_password = parsed.password

                # Configure SOCKS5 proxy with authentication
                socks.set_default_proxy(
                    socks.SOCKS5,
                    proxy_host,
                    proxy_port,
                    username=proxy_username,
                    password=proxy_password
                )
                socket.socket = socks.socksocket

                print(f"Configured to use Proximo SOCKS5 proxy at {proxy_host}:{proxy_port}")
            else:
                print("No Proximo proxy configured.")

            # Initialize MongoDB client
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            print("MongoDB connection successful!")
        except Exception as e:
            print("MongoDB connection failed:", e)

    def get_db(self):
        return self.db