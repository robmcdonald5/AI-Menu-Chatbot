import random
import json
import torch
import re
import spacy
import numpy as np
import uuid
from sklearn.cluster import KMeans
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer, util
from collections import Counter

# Import the database connection
from connect import db  # Make sure connect.py is in the same directory

# Load SpaCy model and Sentence-BERT model
nlp = spacy.load('en_core_web_sm')
#sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Toggleable debug mode
DEBUG = False  # Set to True to enable debug output

# For visual clarity keep this function high in the stack
def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence

with open('intents.json', 'r') as f:
    intents = json.load(f)

# Precompute intent embeddings
all_patterns = []
pattern_tags = []

# Pattern intent recog main loop
for intent in intents['intents']:
    tag = intent['tag']
    patterns = intent['patterns']
    for pattern in patterns:
        cleaned_pattern = clean_sentence(pattern)
        all_patterns.append(cleaned_pattern)
        pattern_tags.append(tag)

# Compute embeddings for all patterns
pattern_embeddings = sentence_model.encode(all_patterns)

# Apply KMeans clustering
num_clusters = len(set(pattern_tags))
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(pattern_embeddings)

# Map clusters to intents
cluster_labels = kmeans.labels_
cluster_to_intent = {}

for cluster_id in range(num_clusters):
    indices = np.where(cluster_labels == cluster_id)[0]
    cluster_intents = [pattern_tags[i] for i in indices]
    most_common_intent = Counter(cluster_intents).most_common(1)[0][0]
    cluster_to_intent[cluster_id] = most_common_intent

if DEBUG:
    print(f"[DEBUG] Cluster to Intent Mapping: {cluster_to_intent}")

is_fixing = False

# Generate a unique session ID
session_id = str(uuid.uuid4())

# Initialize order_id for this session
order_id = 1

# Fetch menu data from the database
def fetch_menu_data():
    # Fetch main menu items
    main_items_cursor = db.MenuItem.find({'category': 'Main'})
    main_items = list(main_items_cursor)

    # Build the menu dictionary with item names and prices
    menu = {}
    for item in main_items:
        name = item['name'].lower()
        if 'size_details' in item and item['size_details']:
            price = item['size_details'][0].get('price', 0)
            menu[name] = price
        else:
            menu[name] = 0  # Default price if not available

    # Fetch addons and normalize names to lowercase
    # Since 'category' is an array, we can query directly with the category name
    meats = [item['name'].lower() for item in db.MenuItem.find({'category': 'Protein'})]
    rice = [item['name'].lower() for item in db.MenuItem.find({'category': 'Rice'})]
    beans = [item['name'].lower() for item in db.MenuItem.find({'category': 'Beans'})]
    toppings = [item['name'].lower() for item in db.MenuItem.find({'category': 'Toppings'})]

    if DEBUG:
        print(f"[DEBUG] Meats list: {meats}")
        print(f"[DEBUG] Rice list: {rice}")
        print(f"[DEBUG] Beans list: {beans}")
        print(f"[DEBUG] Toppings list: {toppings}")

        # Fetch all items and collect categories
        all_items = list(db.MenuItem.find({}))
        categories_set = set()
        for item in all_items:
            if 'category' in item:
                categories = item['category']
                if isinstance(categories, list):
                    categories_set.update([cat.lower() for cat in categories])
                else:
                    categories_set.add(item['category'].lower())
        print(f"[DEBUG] All categories in database: {categories_set}")

    return menu, meats, rice, beans, toppings

menu, meats, rice, beans, toppings = fetch_menu_data()

# Create addons list
addons_list = meats + rice + beans + toppings

missing_field_context = {}

field_prompts = {
    "meats": "What kind of meat would you like?",
    "rice": "What type of rice would you like?",
    "beans": "Would you like black beans, pinto beans, or none?",
    "toppings": "What toppings would you like?"
}

bot_name = "Chipotle"
print("Hi I am an automated Chipotle AI menu, what would you like to order! (type 'quit' to exit)")

def check_missing_fields():
    global is_fixing
    orders = list(db.Orders.find({"session_id": session_id}))
    for order in orders:
        for field in ["meats", "rice", "beans", "toppings"]:
            if not order.get(field):  # Checks if the list is empty or missing
                missing_field_context["order_id"] = order["order_id"]
                missing_field_context["field"] = field
                prompt_user_for_missing_field(order["order_id"], field)
                is_fixing = True
                return  # Stop after finding the first missing field
    is_fixing = False  # No missing fields left
    print(f"{bot_name}: Anything else I can help with?")

def prompt_user_for_missing_field(order_id, field):
    print(f"{bot_name}: For order {order_id}, {field_prompts[field]}")

def update_order(order_id, field, value):
    db.Orders.update_one(
        {"session_id": session_id, "order_id": order_id},
        {"$set": {field: value}}
    )
    print(f"{bot_name}: Updated order {order_id} with {field}: {', '.join(value)}")
    missing_field_context.clear()
    check_missing_fields()  # Check if there are more missing fields

def text2int(textnum):
    num_words = {
        "one": 1, "two":2, "three":3, "four":4, "five":5,
        "six":6, "seven":7, "eight":8, "nine":9, "ten":10
    }
    return num_words.get(textnum.lower(), 1)

def process_order_spacy(input_sentence):
    global order_id
    doc = nlp(input_sentence)
    items = []
    meats_op = []
    rice_op = []
    beans_op = []
    toppings_op = []
    quantity = 1

    # Extract quantities
    for ent in doc.ents:
        if ent.label_ == 'CARDINAL':
            try:
                quantity = int(ent.text)
            except ValueError:
                quantity = text2int(ent.text)

    # Extract items
    for token in doc:
        if token.lemma_ in menu:
            items.append(token.lemma_)

    # Extract addons
    doc_text = doc.text.lower()
    for meat in meats:
        if meat.lower() in doc_text:
            meats_op.append(meat)
    for rice_type in rice:
        if rice_type.lower() in doc_text:
            rice_op.append(rice_type)
    for bean in beans:
        if bean.lower() in doc_text:
            beans_op.append(bean)
    for topping in toppings:
        if topping.lower() in doc_text:
            toppings_op.append(topping)

    if items:
        for item in items:
            price = menu[item]
            for _ in range(quantity):
                # Insert the order into the database
                db.Orders.insert_one({
                    "session_id": session_id,
                    "order_id": order_id,
                    "item": item,
                    "price": price,
                    "meats": meats_op if meats_op else [],
                    "rice": rice_op if rice_op else [],
                    "beans": beans_op if beans_op else [],
                    "toppings": toppings_op if toppings_op else []
                })
                order_id += 1  # Increment order_id for the session
        return f"Added {quantity} {', '.join(items)} to your order."
    else:
        return "Sorry, I didn't understand the items you want to order."

def extract_field_value(field, user_input):
    # Create a PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    # Add field-specific patterns
    if field == "meats":
        patterns = [nlp.make_doc(text) for text in meats]
    elif field == "rice":
        patterns = [nlp.make_doc(text) for text in rice]
    elif field == "beans":
        patterns = [nlp.make_doc(text) for text in beans]
    elif field == "toppings":
        patterns = [nlp.make_doc(text) for text in toppings]
    else:
        return None
    matcher.add("FIELD_VALUE", patterns)

    doc = nlp(user_input)
    if DEBUG:
        print(f"[DEBUG] Field: {field}")
        print(f"[DEBUG] Patterns: {[doc.text for doc in patterns]}")
        print(f"[DEBUG] User Input: {user_input}")
    matches = matcher(doc)
    if matches:
        # Collect all matching phrases
        values = set()
        for match_id, start, end in matches:
            span = doc[start:end]
            values.add(span.text.lower())
        if DEBUG:
            print(f"[DEBUG] Matches found: {values}")
        return list(values)
    else:
        if DEBUG:
            print("[DEBUG] No matches found.")
        return None

def display_current_order():
    orders = list(db.Orders.find({"session_id": session_id}))
    if orders:
        print(f"{bot_name}: Here is your current order:")
        for order in orders:
            meats = ', '.join(order['meats']) if order['meats'] else 'None'
            rice = ', '.join(order['rice']) if order['rice'] else 'None'
            beans = ', '.join(order['beans']) if order['beans'] else 'None'
            toppings = ', '.join(order['toppings']) if order['toppings'] else 'None'
            print(f"Order ID: {order['order_id']}, Item: {order['item']}, Meats: {meats}, Rice: {rice}, Beans: {beans}, Toppings: {toppings}")
    else:
        print(f"{bot_name}: Your order is currently empty.")

def extract_order_ids(input_sentence):
    doc = nlp(input_sentence)
    order_ids = []
    for ent in doc.ents:
        if ent.label_ == 'CARDINAL':
            try:
                order_ids.append(int(ent.text))
            except ValueError:
                order_ids.append(text2int(ent.text))
    return order_ids

def remove_items_by_ids(order_ids):
    removed_items = []
    for oid in order_ids:
        order = db.Orders.find_one({"session_id": session_id, "order_id": oid})
        if order:
            removed_items.append(f"{order['item']} (Order ID {oid})")
            db.Orders.delete_one({"session_id": session_id, "order_id": oid})
    return removed_items

def extract_features(input_sentence):
    doc = nlp(input_sentence)
    features = {
        "item": None,
        "meats": [],
        "rice": [],
        "beans": [],
        "toppings": []
    }

    # Extract items
    for token in doc:
        if token.lemma_ in menu:
            features["item"] = token.lemma_

    # Extract addons
    doc_text = doc.text.lower()
    for meat in meats:
        if meat.lower() in doc_text:
            features["meats"].append(meat)
    for rice_type in rice:
        if rice_type.lower() in doc_text:
            features["rice"].append(rice_type)
    for bean in beans:
        if bean.lower() in doc_text:
            features["beans"].append(bean)
    for topping in toppings:
        if topping.lower() in doc_text:
            features["toppings"].append(topping)

    return features

def remove_items_by_features(features):
    removed_items = []
    query = {"session_id": session_id}

    if features["item"]:
        query["item"] = features["item"]
    if features["meats"]:
        query["meats"] = {"$in": features["meats"]}
    if features["rice"]:
        query["rice"] = {"$in": features["rice"]}
    if features["beans"]:
        query["beans"] = {"$in": features["beans"]}
    if features["toppings"]:
        query["toppings"] = {"$in": features["toppings"]}

    orders_to_remove = list(db.Orders.find(query))

    for order in orders_to_remove:
        removed_items.append(f"{order['item']} (Order ID {order['order_id']})")
        db.Orders.delete_one({"session_id": session_id, "order_id": order["order_id"]})

    return removed_items

def predict_intent_with_clustering(user_input):
    cleaned_input = clean_sentence(user_input)
    input_embedding = sentence_model.encode([cleaned_input])
    cluster_id = kmeans.predict(input_embedding)[0]
    predicted_intent = cluster_to_intent.get(cluster_id, None)
    return predicted_intent

chat_length = 0

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    cleaned_sentence = clean_sentence(sentence)

    if DEBUG:
        print(f"[DEBUG] Cleaned sentence: {cleaned_sentence}")

    # If fixing, first try to extract the missing field value
    if is_fixing:
        # User is providing missing field info
        order_id_fix = missing_field_context["order_id"]
        field = missing_field_context["field"]

        # Extract the value using the updated function
        value = extract_field_value(field, cleaned_sentence)

        if DEBUG:
            print(f"[DEBUG] Extracted value: {value}")

        if value:
            update_order(order_id_fix, field, value)
            # After updating, check if there are more missing fields
            if is_fixing:
                order_id_fix = missing_field_context["order_id"]
                field = missing_field_context["field"]
            continue
        else:
            # If value not found, prompt again
            pass

    # Compute embedding with clustering
    predicted_tag = predict_intent_with_clustering(sentence)

    if DEBUG:
        print(f"[DEBUG] Predicted intent: {predicted_tag}")

    if predicted_tag:
        for intent in intents['intents']:
            if predicted_tag == intent["tag"]:
                if predicted_tag == "order" and not is_fixing:
                    response = process_order_spacy(sentence)
                    chat_length += 1
                    print(f"{bot_name}: {response}")
                    if DEBUG:
                        print(f"[DEBUG] Current orders in DB")
                    check_missing_fields()

                elif predicted_tag == "remove_id" and not is_fixing:
                    order_ids = extract_order_ids(sentence)
                    if order_ids:
                        removed_items = remove_items_by_ids(order_ids)
                        if removed_items:
                            print(f"{bot_name}: I've removed the following items from your order: {', '.join(removed_items)}.")
                            if DEBUG:
                                print(f"[DEBUG] Current orders in DB")
                        else:
                            print(f"{bot_name}: I couldn't find any items with the specified IDs.")
                    else:
                        print(f"{bot_name}: I couldn't detect any order IDs in your request.")

                elif predicted_tag == "remove_desc" and not is_fixing:
                    # Extract features and remove items based on features
                    features = extract_features(sentence)
                    removed_items = remove_items_by_features(features)
                    if removed_items:
                        print(f"{bot_name}: I've removed the following items from your order: {', '.join(removed_items)}.")
                        if DEBUG:
                            print(f"[DEBUG] Current orders in DB")
                    else:
                        print(f"{bot_name}: I couldn't find any items matching the specified features.")

                elif predicted_tag == "check_order":
                    # Display the current order
                    display_current_order()

                    if is_fixing:
                        print(f"{bot_name}: {field_prompts[field]}")

                elif predicted_tag == "modify_order" and not is_fixing:
                    # Implement modification logic here
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
                    # Modification code would go here

                else:
                    # If still fixing, prompt for the missing field
                    if is_fixing:
                        order_id_fix = missing_field_context["order_id"]
                        field = missing_field_context["field"]
                        print(f"{bot_name}: Sorry, I don't understand what you're saying. {field_prompts[field]}")
                    else:
                        print(f"{bot_name}: {random.choice(intent['responses'])}")

    else:
        print(f"{bot_name}: I do not understand...")

    if chat_length > 0 and not is_fixing and predicted_tag != "checkout":
        print(f"{bot_name}: Anything else I can help with?")