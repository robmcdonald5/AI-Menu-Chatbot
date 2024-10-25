import os
from db import Database

# Get environment variables
db_host = os.getenv('DB_HOST', 'localhost')
db_user = os.getenv('DB_USER', 'user')
db_pass = os.getenv('DB_PASS', 'password')
database_name = 'ChipotleMenu'

# Initialize the Database object with connection parameters
database = Database(database_name, host=db_host, user=db_user, password=db_pass)