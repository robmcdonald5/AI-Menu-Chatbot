from database import Database

# Initialize the Database object with only the database name
database = Database('ChipotleMenu')

# Connect to MongoDB
database.connect()