import os
from pymongo import MongoClient
import socks
import socket
import urllib.parse

class Database:
    def __init__(self, db_name):
        # Retrieve environment variables
        self.uri = os.getenv('MONGODB_URI')
        self.proxy_url = os.getenv('PROXIMO_URL')

        if not self.uri:
            raise ValueError("MONGODB_URI is not set in environment variables.")
        
        if not self.proxy_url:
            raise ValueError("PROXIMO_URL is not set in environment variables.")

        print(f"MongoDB URI: {self.uri}")         # For debugging; remove in production
        print(f"Proximo URL: {self.proxy_url}")   # For debugging; remove in production

        self.client = None
        self.db = None
        self.db_name = db_name

    def connect(self):
        try:
            # Parse the PROXIMO_URL
            parsed = urllib.parse.urlparse(self.proxy_url)
            proxy_scheme = parsed.scheme.lower()

            if proxy_scheme not in ['socks5', 'socks4']:
                raise ValueError("PROXIMO_URL must start with socks5:// or socks4://")

            # Map scheme to PySocks proxy type
            if proxy_scheme == 'socks5':
                proxy_type = socks.SOCKS5
            elif proxy_scheme == 'socks4':
                proxy_type = socks.SOCKS4

            # Set default proxy
            socks.set_default_proxy(
                proxy_type,
                parsed.hostname,
                parsed.port,
                username=parsed.username,
                password=parsed.password
            )
            socket.socket = socks.socksocket

            # Initialize MongoDB client
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            print("Connection to MongoDB successful via Proximo proxy!")

        except Exception as e:
            print("Connection to MongoDB failed:", e)
            raise e

    def get_db(self):
        if not self.db:
            self.connect()
        return self.db
