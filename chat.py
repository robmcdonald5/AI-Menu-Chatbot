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
import os
from datetime import datetime, timedelta

# Import the database connection
from connect import database as db  # Ensure connect.py is correctly set up with get_db()

app = Flask(__name__, static_folder='frontend/build')  # Set static_folder to frontend/build
# CORS(app, resources={r"/*": {"origins": ["https://chipotleaimenu.app"]}}, supports_credentials=True)

# Load SpaCy model and Sentence-BERT model
nlp = spacy.load('en_core_web_sm')
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

if DEBUG:
    print(f"[DEBUG] Loaded {len(all_patterns)} patterns for intent recognition.")

# Fetch menu data from the database
def fetch_menu_data():
    db_instance = db.get_db()

    # Access the MenuItem collection
    menu_item_collection = db_instance['MenuItem']

    # Fetch main menu items with case-insensitive matching and handle arrays
    main_items_cursor = menu_item_collection.find({
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

    if DEBUG:
        print(f"[DEBUG] Menu items: {list(menu.keys())}")

    # Fetch addons and normalize names to lowercase
    meats = [item['name'].lower() for item in menu_item_collection.find({'category': {'$regex': '^protein$', '$options': 'i'}})]
    rice = [item['name'].lower() for item in menu_item_collection.find({'category': {'$regex': '^rice$', '$options': 'i'}})]
    beans = [item['name'].lower() for item in menu_item_collection.find({'category': {'$regex': '^beans$', '$options': 'i'}})]
    toppings = [item['name'].lower() for item in menu_item_collection.find({'category': {'$regex': '^toppings$', '$options': 'i'}})]

    if DEBUG:
        print(f"[DEBUG] Meats list: {meats}")
        print(f"[DEBUG] Rice list: {rice}")
        print(f"[DEBUG] Beans list: {beans}")
        print(f"[DEBUG] Toppings list: {toppings}")

        # Fetch all items and collect categories
        all_items = list(menu_item_collection.find({}))
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
    last_order = db.get_db().Orders.find_one({"session_id": session_id}, sort=[("order_id", -1)])
    if last_order and "order_id" in last_order:
        return last_order["order_id"] + 1
    else:
        return 1

def text2int(textnum):
    num_words = {
        "one": 1, "two":2, "three":3, "four":4, "five":5,
        "six":6, "seven":7, "eight":8, "nine":9, "ten":10
    }
    return num_words.get(textnum.lower(), 1)

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
                db.get_db().Orders.insert_one({
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

def update_order(session_id, order_id, field, value):
    if DEBUG:
        order_before = db.get_db().Orders.find_one({"session_id": session_id, "order_id": order_id})
        print(f"[DEBUG] Order before update: {order_before}")
    result = db.get_db().Orders.update_one(
        {"session_id": session_id, "order_id": order_id},
        {"$set": {field: value}}
    )
    if DEBUG:
        print(f"[DEBUG] Update result: matched {result.matched_count}, modified {result.modified_count}")
    # Check if order is now complete
    order = db.get_db().Orders.find_one({"session_id": session_id, "order_id": order_id})
    if DEBUG:
        print(f"[DEBUG] Updated order: {order}")
    required_fields = ["meats", "rice", "beans", "toppings"]
    if all(order.get(f) for f in required_fields):
        db.get_db().Orders.update_one(
            {"session_id": session_id, "order_id": order_id},
            {"$set": {"completed": True}}
        )
    return f"Updated order {order_id} with {field}: {', '.join(value)}"

def display_current_order(session_id):
    orders = list(db.get_db().Orders.find({"session_id": session_id}))
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
        order = db.get_db().Orders.find_one({"session_id": session_id, "order_id": oid})
        if order:
            removed_items.append(f"{order['item']} (Order ID {oid})")
            db.get_db().Orders.delete_one({"session_id": session_id, "order_id": oid})
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

    orders_to_remove = list(db.get_db().Orders.find(query))

    for order in orders_to_remove:
        removed_items.append(f"{order['item']} (Order ID {order['order_id']})")
        db.get_db().Orders.delete_one({"session_id": session_id, "order_id": order["order_id"]})

    return removed_items

def predict_intent(user_input):
    cleaned_input = clean_sentence(user_input)
    input_embedding = sentence_model.encode([cleaned_input])[0]

    # Compute cosine similarities with all pattern embeddings
    similarities = np.dot(pattern_embeddings, input_embedding) / (np.linalg.norm(pattern_embeddings, axis=1) * np.linalg.norm(input_embedding))
    max_similarity_index = np.argmax(similarities)
    max_similarity = similarities[max_similarity_index]
    predicted_tag = pattern_tags[max_similarity_index]

    if DEBUG:
        print(f"[DEBUG] Max similarity: {max_similarity}, Predicted intent: {predicted_tag}")

    threshold = 0.45  # Adjusted threshold
    if max_similarity >= threshold:
        return predicted_tag
    else:
        return None

def check_missing_fields(session):
    session_id = session['session_id']
    # Only consider orders for this session that are not completed
    orders = list(db.get_db().Orders.find({"session_id": session_id, "completed": False}))
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

# Intent handler functions

def process_order(session_id, session, sentence):
    response = process_order_spacy(session_id, sentence)
    session['chat_length'] += 1
    responses = [response]
    missing_fields_response = check_missing_fields(session)
    if missing_fields_response:
        responses.append(missing_fields_response)
    else:
        responses.append("Anything else I can help with?")
    return "\n".join(responses)

def remove_order_by_id(session_id, session, sentence):
    order_ids = extract_order_ids(sentence)
    responses = []
    if order_ids:
        removed_items = remove_items_by_ids(session_id, order_ids)
        if removed_items:
            responses.append(f"I've removed the following items from your order: {', '.join(removed_items)}.")
        else:
            responses.append("I couldn't find any items with the specified IDs.")
    else:
        responses.append("I couldn't detect any order IDs in your request.")
    return "\n".join(responses)

def remove_order_by_description(session_id, session, sentence):
    features = extract_features(sentence)
    removed_items = remove_items_by_features(session_id, features)
    if removed_items:
        response = f"I've removed the following items from your order: {', '.join(removed_items)}."
    else:
        response = "I couldn't find any items matching the specified features."
    return response

def modify_order(session_id, session, sentence):
    return "Sure, what would you like to modify in your order?"

def checkout_order(session_id, session, sentence):
    # Delete the session's orders
    db.get_db().Orders.delete_many({"session_id": session_id})
    if DEBUG:
        print(f"[DEBUG] Orders for session {session_id} have been deleted upon checkout.")
    response = "Your order is complete and has been submitted. Thank you!"
    # Reset session data
    session['is_fixing'] = False
    session['missing_field_context'] = {}
    session['chat_length'] = 0
    session['last_activity'] = datetime.utcnow()
    return response

def check_order(session_id, session, sentence):
    return display_current_order(session_id)

def restart_order(session_id, session, sentence):
    db.get_db().Orders.delete_many({"session_id": session_id})
    if DEBUG:
        print(f"[DEBUG] Orders for session {session_id} have been reset by user request.")
    # Reset session data but keep the same session_id
    session['is_fixing'] = False
    session['missing_field_context'] = {}
    session['chat_length'] = 0
    session['last_activity'] = datetime.utcnow()
    return "Your order has been reset. You can start a new order."

def check_menu(session_id, session, sentence):
    menu_items = ', '.join(menu.keys())
    return f"Our main items are: {menu_items}"

def provide_options(session_id, session, sentence):
    is_fixing = session.get("is_fixing", False)
    missing_field_context = session.get("missing_field_context", {})
    if is_fixing:
        field = missing_field_context.get("field")
        if field and field in field_prompts:
            if field == "meats":
                options = ', '.join(meats)
            elif field == "rice":
                options = ', '.join(rice)
            elif field == "beans":
                options = ', '.join(beans)
            elif field == "toppings":
                options = ', '.join(toppings)
            else:
                options = "I'm sorry, I don't have options for that."
            return f"Available {field} options are: {options}"
        else:
            return "I'm sorry, I don't know what options you're asking about."
    else:
        return "You can order burritos, bowls, tacos, salads, and more."

# Mapping of intent tags to handler functions
intent_handlers = {
    "order": process_order,
    "remove_id": remove_order_by_id,
    "remove_desc": remove_order_by_description,
    "remove_item": remove_order_by_description,  # Updated to handle 'remove_item' intent
    "modify_order": modify_order,
    "checkout": checkout_order,
    # Interruption intents
    "check_order": check_order,
    "restart": restart_order,
    "restart_order": restart_order,  # Added to handle 'restart_order' intent
    "menu": check_menu,
    "show_menu": check_menu,  # Added to handle 'show_menu' intent
    "ask_options": provide_options,
}

# Define the inactivity timeout
INACTIVITY_TIMEOUT = timedelta(minutes=5)

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"response": "No message provided"}), 400

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"response": "An internal error occurred. Please try again later."}), 500

    sentence = data.get("message")
    session_id = data.get("session_id")

    db_instance = db.get_db()
    sessions_collection = db_instance['Sessions']

    # Retrieve or create session data
    if session_id:
        session = sessions_collection.find_one({'session_id': session_id})
        if session:
            # Existing session
            last_activity = session.get('last_activity', datetime.utcnow())
            if datetime.utcnow() - last_activity > INACTIVITY_TIMEOUT:
                # Session has been inactive for more than 5 minutes
                db_instance.Orders.delete_many({'session_id': session_id})
                if DEBUG:
                    print(f"[DEBUG] Session {session_id} has been inactive for over 5 minutes. Orders deleted.")
                # Reset session data
                session = {
                    'session_id': session_id,
                    'is_fixing': False,
                    'missing_field_context': {},
                    'chat_length': 0,
                    'last_activity': datetime.utcnow()
                }
                sessions_collection.replace_one({'session_id': session_id}, session, upsert=True)
            else:
                # Update last_activity
                sessions_collection.update_one(
                    {'session_id': session_id},
                    {'$set': {'last_activity': datetime.utcnow()}}
                )
                session['last_activity'] = datetime.utcnow()
                if DEBUG:
                    print(f"[DEBUG] Existing session found with session_id: {session_id}")
        else:
            # Session data not found in database, create new session
            session = {
                'session_id': session_id,
                'is_fixing': False,
                'missing_field_context': {},
                'chat_length': 0,
                'last_activity': datetime.utcnow()
            }
            sessions_collection.insert_one(session)
            if DEBUG:
                print(f"[DEBUG] Session data not found, created new session with session_id: {session_id}")
    else:
        # No session_id provided, create new session
        session_id = str(uuid.uuid4())
        session = {
            'session_id': session_id,
            'is_fixing': False,
            'missing_field_context': {},
            'chat_length': 0,
            'last_activity': datetime.utcnow()
        }
        sessions_collection.insert_one(session)
        if DEBUG:
            print(f"[DEBUG] New session created with session_id: {session_id}")

    is_fixing = session.get("is_fixing", False)
    missing_field_context = session.get("missing_field_context", {})
    chat_length = session.get("chat_length", 0)

    cleaned_sentence = clean_sentence(sentence)

    if DEBUG:
        print(f"[DEBUG] Session ID: {session_id}")
        print(f"[DEBUG] Cleaned sentence: {cleaned_sentence}")

    responses = []

    # Predict intent
    predicted_tag = predict_intent(sentence)

    if DEBUG:
        print(f"[DEBUG] Predicted intent: {predicted_tag}")

    # Handle the intent using the intent_handlers mapping
    if predicted_tag in intent_handlers:
        handler_function = intent_handlers[predicted_tag]
        # Call the handler function
        response = handler_function(session_id, session, sentence)
        responses.append(response)

        # Update is_fixing from session in case it was modified
        is_fixing = session.get("is_fixing", False)
        missing_field_context = session.get("missing_field_context", {})

        # If we were in the middle of fixing, and the intent was an interruption, prompt again
        interruption_intents = ["check_order", "restart", "menu", "ask_options", "show_menu", "restart_order"]
        if is_fixing and predicted_tag in interruption_intents:
            field = missing_field_context.get("field")
            if field and field in field_prompts:
                responses.append(field_prompts[field])

    elif is_fixing:
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
            missing_fields_response = check_missing_fields(session)
            if missing_fields_response:
                responses.append(missing_fields_response)
            else:
                responses.append("Anything else I can help with?")

            # Update is_fixing after checking for missing fields
            is_fixing = session.get("is_fixing", False)

            # Update session data in the database
            sessions_collection.replace_one({'session_id': session_id}, session, upsert=True)

        else:
            responses.append(f"Sorry, I didn't understand what you're saying. {field_prompts[field]}")

    else:
        responses.append("I do not understand...")

    # Update session data
    session['last_activity'] = datetime.utcnow()
    sessions_collection.replace_one({'session_id': session_id}, session, upsert=True)

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

# Uncomment the following lines if you want to run the Flask app locally
if __name__ == '__main__':
   port = int(os.environ.get("PORT", 5000))
   app.run(host='0.0.0.0', port=port, debug=True)