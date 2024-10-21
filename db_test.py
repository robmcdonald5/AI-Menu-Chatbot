# db_test.py

from connect import db  # Assuming connect.py is in the same directory

def fetch_menu_data():
    # Fetch and print all unique categories
    all_items = list(db.MenuItem.find({}))
    categories_set = set()
    for item in all_items:
        if 'category' in item:
            categories = item['category']
            if isinstance(categories, list):
                categories_set.update([cat.lower() for cat in categories])
            else:
                categories_set.add(item['category'].lower())
    print(f"All categories in database: {categories_set}\n")

    # Define categories to test
    categories_to_test = ['Protein', 'Rice', 'Beans', 'Toppings']

    # For each category, fetch and print items
    for category in categories_to_test:
        # Since 'category' is an array in your database, we need to adjust the query
        items_cursor = db.MenuItem.find({'category': category})
        items = list(items_cursor)
        print(f"Items in category '{category}':")
        if items:
            for item in items:
                print(f"- {item['name']}")
        else:
            print("No items found.")
        print()

if __name__ == '__main__':
    fetch_menu_data()
