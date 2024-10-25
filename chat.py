from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import random
import json
import torch
import re
import spacy
import numpy as np
import uuid
from sklearn.cluster import KMeans
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer
from collections import Counter
import os  # Import os module

# Import the database connection
from connect import database as db  # Make sure connect.py is in the same directory

app = Flask(__name__, static_folder='frontend/build')  # Set static_folder to frontend/build
CORS(app)

# Load SpaCy model and Sentence-BERT model
nlp = spacy.load('en_core_web_sm')
#sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Toggleable debug mode
DEBUG = True  # Set to True to enable debug output

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

# Pattern intent recognition main loop
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

# Session data storage
session_data = {}

# Fetch menu data from the database
def fetch_menu_data():
    # Fetch main menu items with case-insensitive matching and handle arrays
    main_items_cursor = db.MenuItem.find({
        '$or': [
            {'category': {'$regex': '^main$', '$options': 'i'}},
            {'category': {'$elemMatch': {'$regex': '^main$', '$options': 'i'}}}
        ]
    })
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

    # Print out menu dictionary to verify "item" is included
    if DEBUG:
        print(f"[DEBUG] Menu items: {list(menu.keys())}")

    # Fetch addons and normalize names to lowercase
    meats = [item['name'].lower() for item in db.MenuItem.find({'category': {'$regex': '^protein$', '$options': 'i'}})]
    rice = [item['name'].lower() for item in db.MenuItem.find({'category': {'$regex': '^rice$', '$options': 'i'}})]
    beans = [item['name'].lower() for item in db.MenuItem.find({'category': {'$regex': '^beans$', '$options': 'i'}})]
    toppings = [item['name'].lower() for item in db.MenuItem.find({'category': {'$regex': '^toppings$', '$options': 'i'}})]

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

field_prompts = {
    "meats": "What kind of meat would you like?",
    "rice": "What type of rice would you like?",
    "beans": "Would you like black beans, pinto beans, or none?",
    "toppings": "What toppings would you like?"
}

bot_name = "Chipotle"
print("Hi I am an automated Chipotle AI menu, what would you like to order! (type 'quit' to exit)")

# Pushes Orders forward to check or null for next value
def get_next_order_id(session_id):
    last_order = db.Orders.find_one({"session_id": session_id}, sort=[("order_id", -1)])
    if last_order and "order_id" in last_order:
        return last_order["order_id"] + 1
    else:
        return 1

def check_missing_fields(session_id):
    session = session_data[session_id]
    # Only consider orders for this session that are not completed
    orders = list(db.Orders.find({"session_id": session_id, "completed": False}))
    if DEBUG:
        print(f"[DEBUG] Orders with missing fields: {orders}")
    for order in orders:
        for field in ["meats", "rice", "beans", "toppings"]:
            if not order.get(field):  # Checks if the list is empty or missing
                if DEBUG:
                    print(f"[DEBUG] Missing field '{field}' in order: {order}")
                session["missing_field_context"]["order_id"] = order["order_id"]
                session["missing_field_context"]["field"] = field
                session["is_fixing"] = True
                return f"For order {order['order_id']}, {field_prompts[field]}"
    session["is_fixing"] = False  # No missing fields left
    return None

def update_order(session_id, order_id, field, value):
    if DEBUG:
        order_before = db.Orders.find_one({"session_id": session_id, "order_id": order_id})
        print(f"[DEBUG] Order before update: {order_before}")
    result = db.Orders.update_one(
        {"session_id": session_id, "order_id": order_id},
        {"$set": {field: value}}
    )
    if DEBUG:
        print(f"[DEBUG] Update result: matched {result.matched_count}, modified {result.modified_count}")
    # Check if order is now complete
    order = db.Orders.find_one({"session_id": session_id, "order_id": order_id})
    if DEBUG:
        print(f"[DEBUG] Updated order: {order}")
    required_fields = ["meats", "rice", "beans", "toppings"]
    if all(order.get(f) for f in required_fields):
        db.Orders.update_one(
            {"session_id": session_id, "order_id": order_id},
            {"$set": {"completed": True}}
        )
    return f"Updated order {order_id} with {field}: {', '.join(value)}"

def text2int(textnum):
    num_words = {
        "one": 1, "two":2, "three":3, "four":4, "five":5,
        "six":6, "seven":7, "eight":8, "nine":9, "ten":10
    }
    return num_words.get(textnum.lower(), 1)

def process_order_spacy(session_id, input_sentence):
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
                # Generate the next order_id for this session
                order_id = get_next_order_id(session_id)
                # Insert the order into the database
                db.Orders.insert_one({
                    "session_id": session_id,
                    "order_id": order_id,
                    "item": item,
                    "price": price,
                    "meats": meats_op if meats_op else [],
                    "rice": rice_op if rice_op else [],
                    "beans": beans_op if beans_op else [],
                    "toppings": toppings_op if toppings_op else [],
                    "completed": False  # Add completed flag
                })
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

def display_current_order(session_id):
    orders = list(db.Orders.find({"session_id": session_id}))
    if orders:
        response_lines = [f"Here is your current order:"]
        for order in orders:
            meats = ', '.join(order['meats']) if order['meats'] else 'None'
            rice = ', '.join(order['rice']) if order['rice'] else 'None'
            beans = ', '.join(order['beans']) if order['beans'] else 'None'
            toppings = ', '.join(order['toppings']) if order['toppings'] else 'None'
            response_lines.append(f"Order ID: {order['order_id']}, Item: {order['item']}, Meats: {meats}, Rice: {rice}, Beans: {beans}, Toppings: {toppings}")
        return '\n'.join(response_lines)
    else:
        return "Your order is currently empty."

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

def remove_items_by_ids(session_id, order_ids):
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

def remove_items_by_features(session_id, features):
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

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"response": "No message provided"}), 400

    sentence = data.get("message")
    session_id = data.get("session_id")

    if not session_id or session_id not in session_data:
        old_session_id = session_id  # Save old session_id if any
        # Generate a new session_id and initialize session data
        session_id = str(uuid.uuid4())
        session_data[session_id] = {
            "is_fixing": False,
            "missing_field_context": {},
            "chat_length": 0
        }

        # Delete orders associated with the old session_id
        if old_session_id:
            db.Orders.delete_many({"session_id": old_session_id})
            if DEBUG:
                print(f"[DEBUG] Deleted orders for old session_id: {old_session_id}")

        if DEBUG:
            print(f"[DEBUG] New session created with session_id: {session_id}")
    else:
        if DEBUG:
            print(f"[DEBUG] Existing session found with session_id: {session_id}")

    session = session_data[session_id]
    is_fixing = session["is_fixing"]
    missing_field_context = session["missing_field_context"]
    chat_length = session["chat_length"]

    cleaned_sentence = clean_sentence(sentence)

    if DEBUG:
        print(f"[DEBUG] Session ID: {session_id}")
        print(f"[DEBUG] Cleaned sentence: {cleaned_sentence}")

    responses = []

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
            update_msg = update_order(session_id, order_id_fix, field, value)
            responses.append(update_msg)
            # After updating, check if there are more missing fields
            missing_fields_response = check_missing_fields(session_id)
            if missing_fields_response:
                responses.append(missing_fields_response)
            else:
                responses.append("Anything else I can help with?")

            # Update is_fixing after checking for missing fields
            is_fixing = session_data[session_id]["is_fixing"]

            if not is_fixing:
                responses.append("Anything else I can help with?")
            return jsonify({"response": "\n".join(responses), "session_id": session_id})
        else:
            # Try to predict intent even when is_fixing is True
            predicted_tag = predict_intent_with_clustering(sentence)

            if DEBUG:
                print(f"[DEBUG] Predicted intent while fixing: {predicted_tag}")

            if predicted_tag in ["check_order"]:
                # Handle these intents even when fixing
                if predicted_tag == "check_order":
                    display_details = display_current_order(session_id)
                    responses.append(display_details)
                    # Prompt again for the missing field
                    responses.append(field_prompts[field])
                else:
                    # For other intents, keep prompting for the missing field
                    responses.append(f"Sorry, I didn't understand what you're saying. {field_prompts[field]}")

                return jsonify({"response": "\n".join(responses), "session_id": session_id})
            else:
                responses.append(f"Sorry, I didn't understand what you're saying. {field_prompts[field]}")
                return jsonify({"response": "\n".join(responses), "session_id": session_id})

    # Compute embedding with clustering
    predicted_tag = predict_intent_with_clustering(sentence)

    if DEBUG:
        print(f"[DEBUG] Predicted intent: {predicted_tag}")

    if predicted_tag:
        for intent in intents['intents']:
            if predicted_tag == intent["tag"]:
                if predicted_tag == "order" and not is_fixing:
                    response = process_order_spacy(session_id, sentence)
                    chat_length += 1
                    responses.append(response)
                    if DEBUG:
                        print(f"[DEBUG] Current orders in DB")
                    missing_fields_response = check_missing_fields(session_id)
                    if missing_fields_response:
                        responses.append(missing_fields_response)
                    else:
                        responses.append("Anything else I can help with?")

                    # Update is_fixing after checking for missing fields
                    is_fixing = session_data[session_id]["is_fixing"]

                elif predicted_tag == "remove_id" and not is_fixing:
                    order_ids = extract_order_ids(sentence)
                    if order_ids:
                        removed_items = remove_items_by_ids(session_id, order_ids)
                        if removed_items:
                            responses.append(f"I've removed the following items from your order: {', '.join(removed_items)}.")
                            if DEBUG:
                                print(f"[DEBUG] Current orders in DB")
                        else:
                            responses.append("I couldn't find any items with the specified IDs.")
                    else:
                        responses.append("I couldn't detect any order IDs in your request.")

                elif predicted_tag == "remove_desc" and not is_fixing:
                    # Extract features and remove items based on features
                    features = extract_features(sentence)
                    removed_items = remove_items_by_features(session_id, features)
                    if removed_items:
                        responses.append(f"I've removed the following items from your order: {', '.join(removed_items)}.")
                        if DEBUG:
                            print(f"[DEBUG] Current orders in DB")
                    else:
                        responses.append("I couldn't find any items matching the specified features.")

                elif predicted_tag == "check_order":
                    # Display the current order
                    display_details = display_current_order(session_id)
                    responses.append(display_details)

                    # Update is_fixing after displaying the order
                    is_fixing = session_data[session_id]["is_fixing"]

                    if is_fixing:
                        field = session["missing_field_context"]["field"]
                        responses.append(field_prompts[field])
                    else:
                        responses.append("Anything else I can help with?")

                elif predicted_tag == "modify_order" and not is_fixing:
                    # Implement modification logic here
                    responses.append(random.choice(intent['responses']))
                    # Modification code would go here

                else:
                    if is_fixing:
                        field = session["missing_field_context"]["field"]
                        responses.append(f"Sorry, I don't understand what you're saying. {field_prompts[field]}")
                    else:
                        responses.append(random.choice(intent['responses']))

                break  # Intent found and processed
    else:
        responses.append("I do not understand...")

    # Update session data
    session_data[session_id]["is_fixing"] = is_fixing
    session_data[session_id]["chat_length"] = chat_length

    return jsonify({"response": "\n".join(responses), "session_id": session_id})

# Serve React frontend
@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

# Serve other static files
@app.route('/<path:path>')
def serve_static(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=False)