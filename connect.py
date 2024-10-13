from db import Database

database_name = 'ChipotleMenu'
database = Database(database_name)

database.connect()

db = database.get_db()