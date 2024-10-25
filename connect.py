from db import Database

# Define the database name
database_name = 'ChipotleMenu'

# Initialize the Database object
database = Database(database_name)

# Establish the connection to MongoDB
database.connect()

# Retrieve the connected database object
db = database.get_db()