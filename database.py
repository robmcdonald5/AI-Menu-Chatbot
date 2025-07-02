import os
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConfigurationError, OperationFailure
import time
import socks  # Import PySocks
import socket
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Database:
    def __init__(self, db_name):
        self.uri = os.getenv('MONGODB_URI')
        self.client = None
        self.db = None
        self.db_name = db_name
        
        # Debug: Print the URI (without password for security)
        if self.uri:
            # Mask the password in the URI for logging
            masked_uri = self.uri.replace(self.uri.split(':')[2].split('@')[0], '****')
            print(f"Using MongoDB URI: {masked_uri}")
        else:
            print("ERROR: MONGODB_URI not found in environment variables")

    def connect(self):
        if not self.uri:
            print("ERROR: No MongoDB URI provided")
            return False
            
        for attempt in range(3):
            try:
                print(f"Attempting to connect to MongoDB (Attempt {attempt + 1})")
                
                # Reset socket in case it was modified by previous attempts
                if hasattr(socket, '_original_socket'):
                    socket.socket = socket._original_socket
                
                # Set up SOCKS5 proxy if QUOTAGUARDSTATIC_SOCKS5_URL is set
                quotaguard_url = os.getenv('QUOTAGUARDSTATIC_SOCKS5_URL')
                if quotaguard_url:
                    print("Setting up SOCKS5 proxy...")
                    # Store original socket for reset
                    if not hasattr(socket, '_original_socket'):
                        socket._original_socket = socket.socket
                        
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
                else:
                    print("No proxy configuration found")

                # Create MongoDB client with more detailed configuration
                self.client = MongoClient(
                    self.uri,
                    serverSelectionTimeoutMS=30000,
                    socketTimeoutMS=30000,
                    connectTimeoutMS=30000,
                    tls=True,  # Ensure TLS is enabled
                    tlsAllowInvalidCertificates=False,  # Keep secure
                    retryWrites=True,
                    w='majority'
                )
                
                print("Client created, attempting to ping...")
                
                # Force a connection to verify settings
                result = self.client.admin.command('ping')
                print(f"Ping result: {result}")
                
                # Set the database
                self.db = self.client[self.db_name]
                print(f"Connected to database: {self.db_name}")
                
                # Test database access
                collections = self.db.list_collection_names()
                print(f"Available collections: {collections}")
                
                print("MongoDB connection successful!")
                return True
                
            except ServerSelectionTimeoutError as e:
                print(f"MongoDB connection timeout on attempt {attempt + 1}: {e}")
                print("This usually means the server is unreachable or credentials are wrong")
                time.sleep(5)
                
            except ConfigurationError as e:
                print(f"MongoDB configuration error on attempt {attempt + 1}: {e}")
                print("Check your connection string format")
                break  # Don't retry configuration errors
                
            except OperationFailure as e:
                print(f"MongoDB operation failed on attempt {attempt + 1}: {e}")
                print("This usually means authentication failed")
                break  # Don't retry auth failures
                
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {type(e).__name__}: {e}")
                time.sleep(5)

        print("Failed to connect to MongoDB after all attempts")
        return False

    def get_db(self):
        if self.db is None:
            success = self.connect()
            if not success:
                return None
        return self.db

    def close(self):
        if self.client:
            self.client.close()
            print("MongoDB connection closed")