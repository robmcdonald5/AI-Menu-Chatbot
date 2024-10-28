import os
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import time
import socks
import socket
from urllib.parse import urlparse

def create_socks5_socket_factory(proxy_url):
    parsed = urlparse(proxy_url)

    def socks5_socket(*args):
        sock = socks.socksocket()
        sock.set_proxy(
            socks.SOCKS5,
            addr=parsed.hostname,
            port=parsed.port,
            username=parsed.username,
            password=parsed.password,
            rdns=True
        )
        sock.connect(args[0])
        return sock

    return socks5_socket

class Database:
    def __init__(self, db_name):
        self.uri = os.getenv('MONGODB_URI')
        self.client = None
        self.db = None
        self.db_name = db_name

    def connect(self):
        for attempt in range(3):
            try:
                print(f"Attempting to connect to MongoDB (Attempt {attempt + 1})")

                # Set up custom socket factory if QUOTAGUARDSTATIC_URL is set
                quotaguard_url = os.getenv('QUOTAGUARDSTATIC_URL')
                if quotaguard_url:
                    socket_factory = create_socks5_socket_factory(quotaguard_url)
                    self.client = MongoClient(
                        self.uri,
                        serverSelectionTimeoutMS=30000,
                        socketTimeoutMS=30000,
                        tls=True,
                        socketFactory=socket_factory
                    )
                else:
                    self.client = MongoClient(
                        self.uri,
                        serverSelectionTimeoutMS=30000,
                        socketTimeoutMS=30000,
                        tls=True
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
        else:
            print("Failed to connect to MongoDB after 3 attempts")

    def get_db(self):
        if self.db is None:
            self.connect()
        return self.db