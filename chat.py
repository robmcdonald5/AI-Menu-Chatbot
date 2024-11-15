from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import random
import json
import re
import spacy
import numpy as np
import uuid
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import os
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import MinMaxScaler
from word2number import w2n
import logging
from logging.handlers import RotatingFileHandler
import sys

# Import the database connection
from connect import database as db  # Ensure connect.py is correctly set up with get_db()

# Import the MenuFuzzer
from fuzzer import MenuFuzzer  # Ensure fuzzer.py is in the same directory or in Python path

app = Flask(__name__, static_folder='frontend/build')  # Set static_folder to frontend/build
CORS(app, resources={r"/*": {"origins": ["http://localhost:5001"]}}, supports_credentials=True)
#CORS(app, resources={r"/*": {"origins": ["https://chipotleaimenu.app"]}}, supports_credentials=True)

## Logging Configuration ##

# Toggleable debug mode
DEBUG = True  # Set to False in production

# Create the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)  # Default to INFO to reduce verbosity

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create and configure console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)  # Console shows INFO and above
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# Create and configure rotating file handler
file_handler = RotatingFileHandler("chatbot.log", maxBytes=5*1024*1024, backupCount=5)  # 5MB per file, keep last 5
file_handler.setLevel(logging.DEBUG if DEBUG else logging.INFO)  # File logs DEBUG if DEBUG=True
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Suppress verbose logs from external libraries
logging.getLogger('pymongo').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('spacy').setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.WARNING)  # Flask's built-in server
# Add any other external libraries as needed

# Create a logger for your application
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)  # Application logger

## End of Logging Configuration ##

# Load SpaCy model and Sentence-BERT model
nlp = spacy.load('en_core_web_sm')
sentence_model = SentenceTransformer('all-mpnet-base-v2')

# Define similarity thresholds
SIMILARITY_THRESHOLD_HIGH = 0.7  # High confidence
SIMILARITY_THRESHOLD_MEDIUM = 0.45  # Medium confidence

# Define weights for similarity metrics
WEIGHT_COSINE = 0.5
WEIGHT_EUCLIDEAN = 0.3
WEIGHT_JACCARD = 0.2

# Function to clean sentences
def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence

# Load intents from intents.json
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Create a dictionary mapping tags to their respective intents for easier access
intents_dict = {intent['tag']: intent for intent in intents['intents']}

# Precompute intent embeddings
all_patterns = []
pattern_tags = []

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
    logger.debug(f"Loaded {len(all_patterns)} patterns for intent recognition.")

## Helper Functions ##

# Function to update order
def update_order(session_id, order_id, field, value):
    if DEBUG:
        order_before = db.get_db().Orders.find_one({"session_id": session_id, "order_id": order_id})
        logger.debug(f"Order before update: {order_before}")
    result = db.get_db().Orders.update_one(
        {"session_id": session_id, "order_id": order_id},
        {"$set": {field: value}}
    )
    if DEBUG:
        logger.debug(f"Update result: matched {result.matched_count}, modified {result.modified_count}")
    # Check if order is now complete
    order = db.get_db().Orders.find_one({"session_id": session_id, "order_id": order_id})
    if DEBUG:
        logger.debug(f"Updated order: {order}")
    required_fields = ["meats", "rice", "beans", "toppings"]
    if all(order.get(f) for f in required_fields):
        db.get_db().Orders.update_one(
            {"session_id": session_id, "order_id": order_id},
            {"$set": {"completed": True}}
        )
    return f"Updated order {order_id} with {field}: {', '.join(value)}"

def replace_spelled_numbers(text):
    words = text.split()
    result = []

    for word in words:
        try:
            # Convert spelled-out number to numeric equivalent
            number = w2n.word_to_num(word)
            result.append(str(number))
        except ValueError:
            result.append(word)

    return " ".join(result)

def segment_input(input_sentence):
    input_sentence = replace_spelled_numbers(input_sentence)
    articles = ["a", "an", "the"]
    pattern = r"(?=\b(?:\d+|" + "|".join(articles) + r")\b)"
    substrings = re.split(pattern, input_sentence)
    substrings = [s.strip() for s in substrings if s.strip()]
    return substrings

def is_entree(category):
    entree_categories = ['main', 'entree']
    if isinstance(category, list):
        # Convert all categories to lowercase for case-insensitive comparison
        category = [c.lower() for c in category]
        return any(cat in entree_categories for cat in category)
    elif isinstance(category, str):
        return category.lower() in entree_categories
    else:
        return False

# Fetch menu data from the database
def fetch_menu_data():
    db_instance = db.get_db()

    # Access the MenuItem collection
    menu_item_collection = db_instance['MenuItem']

    # Fetch all menu items
    all_items_cursor = menu_item_collection.find({})
    all_items = list(all_items_cursor)

    # Build the menu dictionary with item details
    menu = {}
    for item in all_items:
        name = item['name'].lower()
        price = item['size_details'][0].get('price', 0) if 'size_details' in item and item['size_details'] else 0
        category = item.get('category', 'other')
        if isinstance(category, list):
            category = [c.lower() for c in category]
        elif isinstance(category, str):
            category = category.lower()
        else:
            category = 'other'
        menu[name] = {
            'price': price,
            'category': category
        }

    logger.debug(f"Loaded {len(menu)} menu items.")

    # Fetch addons and normalize names to lowercase
    meats = [item['name'].lower() for item in menu_item_collection.find({'category': {'$regex': '^protein$', '$options': 'i'}})]
    rice = [item['name'].lower() for item in menu_item_collection.find({'category': {'$regex': '^rice$', '$options': 'i'}})]
    beans = [item['name'].lower() for item in menu_item_collection.find({'category': {'$regex': '^beans$', '$options': 'i'}})]
    toppings = [item['name'].lower() for item in menu_item_collection.find({'category': {'$regex': '^toppings$', '$options': 'i'}})]

    logger.debug(f"Meats list: {meats}")
    logger.debug(f"Rice list: {rice}")
    logger.debug(f"Beans list: {beans}")
    logger.debug(f"Toppings list: {toppings}")

    # Define add-ons
    addons = set(meats + rice + beans + toppings)

    # Define main items: all menu items not in add-ons
    main_items = [name for name in menu.keys() if name not in addons]

    logger.debug(f"Main items: {main_items}")

    return menu, meats, rice, beans, toppings, main_items

# Initial fetch of menu data
menu, meats, rice, beans, toppings, main_items = fetch_menu_data()

# Initialize the MenuFuzzer with menu data
fuzzer = MenuFuzzer(menu_items=main_items, meats=meats, rice=rice, beans=beans, toppings=toppings, debug=DEBUG)

# Initialize PhraseMatcher for main menu items only
menu_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
menu_patterns = [nlp.make_doc(name) for name in main_items]
menu_matcher.add("MENU_ITEM", menu_patterns)

# Define add-on categories
addon_categories = {
    "meats": meats,
    "rice": rice,
    "beans": beans,
    "toppings": toppings
}

# Initialize a single PhraseMatcher for add-ons
addon_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
for category, addons_list in addon_categories.items():
    patterns = [nlp.make_doc(addon) for addon in addons_list]
    addon_matcher.add(category.upper(), patterns)

# Create addons list
addons_list = meats + rice + beans + toppings

field_prompts = {
    "meats": "What kind of meat would you like?",
    "rice": "What type of rice would you like?",
    "beans": "Would you like black beans, pinto beans, or none?",
    "toppings": "What toppings would you like?"
}

bot_name = "Chipotle"
logger.info("Hi, I am an automated Chipotle AI menu. What would you like to order! (type 'quit' to exit)")

# Function to get next order_id
def get_next_order_id(session_id):
    last_order = db.get_db().Orders.find_one({"session_id": session_id}, sort=[("order_id", -1)])
    if last_order and "order_id" in last_order:
        return last_order["order_id"] + 1
    else:
        return 1

# Function to convert text numbers to integers
def text2int(textnum):
    num_words = {
        "one": 1, "two":2, "three":3, "four":4, "five":5,
        "six":6, "seven":7, "eight":8, "nine":9, "ten":10
    }
    return num_words.get(textnum.lower(), 1)

# Function to extract field values using PhraseMatcher
def extract_field_value(field, user_input):
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
        logger.debug(f"Field: {field}")
        logger.debug(f"Patterns: {[pattern.text for pattern in patterns]}")
        logger.debug(f"User Input: {user_input}")
    matches = matcher(doc)
    if matches:
        # Collect all matching phrases
        values = set()
        for match_id, start, end in matches:
            span = doc[start:end]
            values.add(span.text.lower())
        if DEBUG:
            logger.debug(f"Matches found: {values}")
        return list(values)
    else:
        if DEBUG:
            logger.debug("No matches found.")
        return None

# Function to extract menu items using PhraseMatcher
def extract_menu_items(user_input):
    doc = nlp(user_input)
    matches = menu_matcher(doc)
    items = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        items.add(span.text.lower())
    if DEBUG:
        logger.debug(f"Extracted menu items: {items}")
    return list(items)

# Function to process order using SpaCy with improved association of add-ons
def process_order_spacy(session_id, input_sentence):
    # Segment input sentence
    segments = segment_input(input_sentence)

    if DEBUG:
        logger.debug(f"Segments: {segments}")

    # Initialize a dictionary to hold add-ons for each item
    item_addons = defaultdict(lambda: {"meats": [], "rice": [], "beans": [], "toppings": []})
    items = []

    # Iterate through each segment to extract and associate add-ons
    for seg in segments:
        doc = nlp(seg)
        main_items_in_seg = extract_menu_items(seg)

        # Extract add-ons only if a main item is present in the segment
        if main_items_in_seg:
            main_item = main_items_in_seg[0]
            items.append(main_item)

            # Use the single addon_matcher to find all add-ons
            matches = addon_matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                category = nlp.vocab.strings[match_id]  # Retrieve the category label
                addon = span.text.lower()
                # Map the category label back to lowercase for consistency
                category_key = category.lower()
                if addon not in item_addons[main_item][category_key]:
                    item_addons[main_item][category_key].append(addon)

    if DEBUG:
        logger.debug(f"Items extracted: {items}")
        logger.debug(f"Item addons: {dict(item_addons)}")

    # Extract quantity (assuming total quantity applies to all items)
    quantity = 1
    for ent in nlp(input_sentence).ents:
        if ent.label_ == 'CARDINAL':
            try:
                quantity = int(ent.text)
                break  # Use the first numerical value found
            except ValueError:
                quantity = text2int(ent.text)
                break

    # Fetch current max order_id
    last_order = db.get_db().Orders.find_one({"session_id": session_id}, sort=[("order_id", -1)])
    current_max_order_id = last_order["order_id"] if last_order else 0

    # Define required_addons mapping
    required_addons = {
        'burrito': ['meats', 'rice', 'beans', 'toppings'],
        'bowl': ['rice', 'beans', 'toppings'],
        # Add other items as needed
    }

    # Create and insert orders
    if items:
        orders_to_insert = []
        confirmation_messages = []
        for item in items:
            item_details = menu.get(item, {})
            price = item_details.get('price', 0)
            category = item_details.get('category', 'other')

            # Determine if the item is an entree
            if is_entree(category):
                category_type = "entree"
            else:
                category_type = "non_entree"

            for _ in range(quantity):
                current_max_order_id += 1
                order_id = current_max_order_id
                order_document = {
                    "session_id": session_id,
                    "order_id": order_id,
                    "item": item,
                    "price": price,
                    "category": category_type,  # standardized category
                    "completed": False
                }
                if category_type == "entree":
                    order_document["meats"] = item_addons[item]["meats"] if item_addons[item]["meats"] else []
                    order_document["rice"] = item_addons[item]["rice"] if item_addons[item]["rice"] else []
                    order_document["beans"] = item_addons[item]["beans"] if item_addons[item]["beans"] else []
                    order_document["toppings"] = item_addons[item]["toppings"] if item_addons[item]["toppings"] else []
                # Non-entrees do not have add-ons

                orders_to_insert.append(order_document)

        # Insert all orders at once for efficiency
        if orders_to_insert:
            db.get_db().Orders.insert_many(orders_to_insert)

            # Prepare confirmation messages
            for order in orders_to_insert:
                if order["category"] == "entree":
                    addons = []
                    addons.extend(order["meats"])
                    addons.extend(order["rice"])
                    addons.extend(order["beans"])
                    addons.extend(order["toppings"])
                    if addons:
                        confirmation_messages.append(f"{order['item'].capitalize()} with {', '.join(addons)}")
                    else:
                        confirmation_messages.append(f"{order['item'].capitalize()}")
                else:
                    confirmation_messages.append(f"{order['item'].capitalize()}")

            return f"Added {', '.join(confirmation_messages)} to your order."
    else:
        return "Sorry, I didn't understand the items you want to order."

# Unified Removal Function
def handle_removal(session_id, session, sentence):
    # Determine the removal method
    removal_method = determine_removal_method(sentence)

    if DEBUG:
        logger.debug(f"Removal Method Determined: {removal_method}")

    if removal_method == "reset_order":
        # Prompt for confirmation
        session["pending_action"] = {"action": "reset_order"}
        return "Are you sure you want to reset your entire order? (yes/no)"

    elif removal_method == "by_order_id":
        order_ids = extract_order_ids(sentence)
        if order_ids:
            removed_items = remove_items_by_ids(session_id, order_ids)
            if removed_items:
                return f"I've removed the following items from your order: {', '.join(removed_items)}."
            else:
                return "I couldn't find any items with the specified IDs."
        else:
            return "I couldn't detect any order IDs in your request."

    else:
        # Default to removal by features/descriptions
        features = extract_features(sentence)
        if DEBUG:
            logger.debug(f"Features extracted for removal: {features}")
        removed_items = remove_items_by_features(session_id, features)
        if removed_items:
            return f"I've removed the following items from your order: {', '.join(removed_items)}."
        else:
            return "I couldn't find any items matching the specified features."

# Function to determine removal method
def determine_removal_method(sentence):
    # Define keywords for each removal method
    # Updated regex to include 'remove order \d+' patterns
    if re.search(r'\b(reset|start over|clear my order|delete my order|restart my order|erase my order)\b', sentence, re.IGNORECASE):
        return "reset_order"
    elif re.search(r'\b(order id|item id|order number|delete \d+|remove(?: order)? \d+|cancel \d+)\b', sentence, re.IGNORECASE):
        return "by_order_id"
    else:
        # Default to removal by features/descriptions
        return "by_features"

# Function to extract order IDs
def extract_order_ids(input_sentence):
    doc = nlp(input_sentence)
    order_ids = []
    for ent in doc.ents:
        if ent.label_ == 'CARDINAL':
            try:
                order_ids.append(int(ent.text))
            except ValueError:
                order_ids.append(text2int(ent.text))
    # Additionally, use regex to find standalone numbers
    regex_ids = re.findall(r'\b\d+\b', input_sentence)
    for rid in regex_ids:
        try:
            order_ids.append(int(rid))
        except ValueError:
            continue
    # Remove duplicates
    return list(set(order_ids))

# Function to remove items by IDs
def remove_items_by_ids(session_id, order_ids):
    removed_items = []
    for oid in order_ids:
        order = db.get_db().Orders.find_one({"session_id": session_id, "order_id": oid})
        if order:
            removed_items.append(f"{order['item'].capitalize()} (Order ID {oid})")
            db.get_db().Orders.delete_one({"session_id": session_id, "order_id": oid})
    return removed_items

# Function to extract features
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

    if DEBUG:
        logger.debug(f"Features extracted: {features}")

    return features

# Function to remove items by features
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

    if DEBUG:
        logger.debug(f"Query for removal by features: {query}")

    orders_to_remove = list(db.get_db().Orders.find(query))

    if DEBUG:
        logger.debug(f"Orders to remove based on features: {orders_to_remove}")

    for order in orders_to_remove:
        removed_items.append(f"{order['item'].capitalize()} (Order ID {order['order_id']})")
        db.get_db().Orders.delete_one({"session_id": session_id, "order_id": order["order_id"]})

    return removed_items

# Function to predict intent
def predict_intent(user_input):
    cleaned_input = clean_sentence(user_input)
    input_embedding = sentence_model.encode([cleaned_input])[0]

    # Compute Cosine Similarity
    cosine_similarities = np.dot(pattern_embeddings, input_embedding) / (np.linalg.norm(pattern_embeddings, axis=1) * np.linalg.norm(input_embedding))

    # Compute Euclidean Distance
    euclidean_distances = np.linalg.norm(pattern_embeddings - input_embedding, axis=1)

    # Compute Jaccard Similarity
    input_tokens = set(cleaned_input.split())
    jaccard_similarities = np.array([
        len(input_tokens.intersection(set(pattern.split()))) / len(input_tokens.union(set(pattern.split()))) 
        if len(input_tokens.union(set(pattern.split()))) > 0 else 0 
        for pattern in all_patterns
    ])

    # Normalize Euclidean Distances and Jaccard Similarities
    scaler_euclidean = MinMaxScaler()
    normalized_euclidean = scaler_euclidean.fit_transform(euclidean_distances.reshape(-1, 1)).flatten()
    normalized_jaccard = jaccard_similarities  # Already in [0,1]

    # Combine similarities with weights
    combined_scores = (WEIGHT_COSINE * cosine_similarities) + \
                      (WEIGHT_EUCLIDEAN * (1 - normalized_euclidean)) + \
                      (WEIGHT_JACCARD * normalized_jaccard)

    # Find the best match
    max_score_index = np.argmax(combined_scores)
    max_score = combined_scores[max_score_index]
    predicted_tag = pattern_tags[max_score_index]

    if DEBUG:
        logger.debug(f"Max combined score: {max_score}, Predicted intent: {predicted_tag}")

    # Determine intent based on combined score thresholds
    if max_score >= SIMILARITY_THRESHOLD_HIGH:
        confidence = "high"
        return predicted_tag, confidence
    elif SIMILARITY_THRESHOLD_MEDIUM <= max_score < SIMILARITY_THRESHOLD_HIGH:
        confidence = "medium"
        return predicted_tag, confidence
    else:
        confidence = "low"
        return None, confidence

# Function to check missing fields and set slot-filling context
def check_missing_fields(session):
    if session.get("pending_action"):
        return None  # Skip slot-filling if a critical action is pending

    session_id = session['session_id']
    # Only consider orders for this session that are not completed
    orders = list(db.get_db().Orders.find({"session_id": session_id, "completed": False}))
    if DEBUG:
        logger.debug(f"Orders with missing fields: {orders}")
    for order in orders:
        category = order.get("category", "other")
        if category == "entree":
            # Iterate over each required add-on field
            for field in ["meats", "rice", "beans", "toppings"]:
                if not order.get(field):  # Checks if the list is empty or missing
                    if DEBUG:
                        logger.debug(f"Missing field '{field}' in order: {order}")
                    session["missing_field_context"]["order_id"] = order["order_id"]
                    session["missing_field_context"]["field"] = field
                    session["is_fixing"] = True
                    return f"For order {order['order_id']}, {field_prompts[field]}"
        else:
            # Non-entrees do not require add-ons; mark as completed
            if not order.get("completed"):
                db.get_db().Orders.update_one(
                    {"session_id": session_id, "order_id": order["order_id"]},
                    {"$set": {"completed": True}}
                )
    session["is_fixing"] = False  # No missing fields left
    return None

# Intent handler functions
def process_order(session_id, session, sentence):
    response = process_order_spacy(session_id, sentence)
    session['chat_length'] += 1
    return response

def modify_order_handler(session_id, session, sentence):
    return "Sure, what would you like to modify in your order?"

def checkout_order(session_id, session, sentence):
    # Delete the session's orders
    db.get_db().Orders.delete_many({"session_id": session_id})
    if DEBUG:
        logger.debug(f"Orders for session {session_id} have been deleted upon checkout.")
    # Reset session data
    session['is_fixing'] = False
    session['missing_field_context'] = {}
    session['chat_length'] = 0
    session['last_activity'] = datetime.now(timezone.utc)
    return "Your order is complete and has been submitted. Thank you!"

def check_order(session_id, session, sentence):
    return display_current_order(session_id)

def restart_order(session_id, session, sentence):
    db.get_db().Orders.delete_many({"session_id": session_id})
    if DEBUG:
        logger.debug(f"Orders for session {session_id} have been reset by user request.")
    # Reset session data but keep the same session_id
    session['is_fixing'] = False
    session['missing_field_context'] = {}
    session['chat_length'] = 0
    session['last_activity'] = datetime.now(timezone.utc)
    return "Your order has been reset. You can start a new order."

def check_menu(session_id, session, sentence):
    menu_items = ', '.join([item.capitalize() for item in main_items])
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

# Function to handle confirmation of critical actions
def handle_confirm(session_id, session, sentence):
    if session.get("pending_action") and session["pending_action"].get("action") == "reset_order":
        # Proceed with resetting the order
        db.get_db().Orders.delete_many({"session_id": session_id})
        if DEBUG:
            logger.debug(f"Orders for session {session_id} have been reset upon user confirmation.")
        # Reset session data
        session['is_fixing'] = False
        session['missing_field_context'] = {}
        session['chat_length'] = 0
        session['last_activity'] = datetime.now(timezone.utc)
        # Clear pending action
        session.pop("pending_action", None)
        return "Your order has been reset. You can start a new order."
    else:
        return "Great! Let's continue with your order."

# Function to handle denial of critical actions
def handle_deny(session_id, session, sentence):
    if session.get("pending_action") and session["pending_action"].get("action") == "reset_order":
        # Cancel the reset action
        session.pop("pending_action", None)
        return "Alright, your order remains unchanged."
    else:
        return "Okay, let's clarify your request. What would you like to do?"

# Mapping of intent tags to handler functions
intent_handlers = {
    "goodbye": lambda sid, s, sen: random.choice(intents_dict['goodbye']['responses']),
    "order": process_order,
    "remove_order": handle_removal,
    "modify_order": modify_order_handler,
    "reset_order": handle_removal,
    "checkout": checkout_order,
    "check_order": check_order,
    "restart_order": handle_removal,
    "show_menu": check_menu,
    "ask_options": provide_options,
    "confirm": handle_confirm,
    "deny": handle_deny,
    "vegan_options": lambda sid, s, sen: random.choice(intents_dict['vegan_options']['responses']),
    "fallback": lambda sid, s, sen: random.choice(intents_dict['fallback']['responses'])
}

# Define the inactivity timeout
INACTIVITY_TIMEOUT = timedelta(minutes=5)

# Function to display current order
def display_current_order(session_id):
    orders = list(db.get_db().Orders.find({"session_id": session_id}))
    if orders:
        response_lines = [f"Here is your current order:"]
        for order in orders:
            category = order.get("category", "other")
            if category == "entree":
                meats = ', '.join(order['meats']) if order.get('meats') else 'None'
                rice = ', '.join(order['rice']) if order.get('rice') else 'None'
                beans = ', '.join(order['beans']) if order.get('beans') else 'None'
                toppings = ', '.join(order['toppings']) if order.get('toppings') else 'None'
                response_lines.append(
                    f"Order ID: {order['order_id']}, Item: {order['item'].capitalize()}, "
                    f"Meats: {meats}, Rice: {rice}, Beans: {beans}, Toppings: {toppings}"
                )
            else:
                # For non-entrees, just display item
                response_lines.append(f"Order ID: {order['order_id']}, Item: {order['item'].capitalize()}")
        return '\n'.join(response_lines)
    else:
        return "Your order is currently empty."

# Function to update session data
def update_session(sessions_collection, session, session_id):
    session['last_activity'] = datetime.now(timezone.utc)
    sessions_collection.replace_one({'session_id': session_id}, session, upsert=True)

@app.route('/get_order', methods=['GET'])
def get_order():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    order_details = list(db.get_db().Orders.find({"session_id": session_id}))

    for order in order_details:
        order['_id'] = str(order['_id'])
    if DEBUG:
        logger.debug(order_details)
    return jsonify({"order_details": order_details})

@app.route('/get_menu_items', methods=['GET'])
def get_menu_items():
    menu_items = list(db.get_db().MenuItem.find({}, {"name": 1, "category": 1, "_id": 0}))

    if DEBUG:
        logger.debug(menu_items)
    return jsonify({"menu_items": menu_items})

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"response": "No message provided"}), 400

    except Exception as e:
        logger.exception("An unexpected error occurred.")
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
            last_activity = session.get('last_activity', datetime.now(timezone.utc))
            if last_activity.tzinfo is None:  # Make datetime timezone-aware if it's naive
                last_activity = last_activity.replace(tzinfo=timezone.utc)

            if datetime.now(timezone.utc) - last_activity > INACTIVITY_TIMEOUT:
                # Session has been inactive for more than 5 minutes
                db_instance.Orders.delete_many({'session_id': session_id})
                if DEBUG:
                    logger.debug(f"Session {session_id} has been inactive for over 5 minutes. Orders deleted.")
                # Reset session data
                session = {
                    'session_id': session_id,
                    'is_fixing': False,
                    'missing_field_context': {},
                    'chat_length': 0,
                    'last_activity': datetime.now(timezone.utc)
                }
                sessions_collection.replace_one({'session_id': session_id}, session, upsert=True)
            else:
                # Update last_activity
                sessions_collection.update_one(
                    {'session_id': session_id},
                    {'$set': {'last_activity': datetime.now(timezone.utc)}}
                )
                session['last_activity'] = datetime.now(timezone.utc)
                if DEBUG:
                    logger.debug(f"Existing session found with session_id: {session_id}")
        else:
            # Session data not found in database, create new session
            session = {
                'session_id': session_id,
                'is_fixing': False,
                'missing_field_context': {},
                'chat_length': 0,
                'last_activity': datetime.now(timezone.utc)
            }
            sessions_collection.insert_one(session)
            if DEBUG:
                logger.debug(f"Session data not found, created new session with session_id: {session_id}")
    else:
        # No session_id provided, create new session
        session_id = str(uuid.uuid4())
        session = {
            'session_id': session_id,
            'is_fixing': False,
            'missing_field_context': {},
            'chat_length': 0,
            'last_activity': datetime.now(timezone.utc)
        }
        sessions_collection.insert_one(session)
        if DEBUG:
            logger.debug(f"New session created with session_id: {session_id}")

    is_fixing = session.get("is_fixing", False)
    missing_field_context = session.get("missing_field_context", {})
    chat_length = session.get("chat_length", 0)

    # Apply fuzzing to correct minor typos
    corrected_sentence, corrections = fuzzer.correct_text(sentence)

    if DEBUG and corrections:
        logger.debug(f"Fuzzing corrections: {corrections}")

    cleaned_sentence = clean_sentence(corrected_sentence)

    if DEBUG:
        logger.debug(f"Session ID: {session_id}")
        logger.debug(f"Original sentence: {sentence}")
        logger.debug(f"Corrected sentence: {corrected_sentence}")
        logger.debug(f"Cleaned sentence: {cleaned_sentence}")

    responses = []

    # Predict intent
    predicted_tag, confidence = predict_intent(sentence)  # Consider using corrected_sentence if appropriate

    if DEBUG:
        logger.debug(f"Predicted intent: {predicted_tag}, Confidence: {confidence}")

    # Define interrupt and confirmation intents
    confirmation_intents = ["confirm", "deny"]
    interrupt_intents = ["remove_order", "modify_order", "restart_order", "show_menu", "check_order", "ask_options", "reset_order"]

    # Handle confirmation intents if a pending action exists
    if session.get("pending_action") and predicted_tag in confirmation_intents and confidence != "low":
        if predicted_tag == "confirm":
            response = intent_handlers["confirm"](session_id, session, sentence)
            responses.append(response)
        elif predicted_tag == "deny":
            response = intent_handlers["deny"](session_id, session, sentence)
            responses.append(response)
        
        # After handling confirmation, check if there are missing fields
        missing_fields_response = check_missing_fields(session)
        if missing_fields_response:
            responses.append(missing_fields_response)
        else:
            responses.append("Anything else I can help with?")

    elif predicted_tag in interrupt_intents and confidence != "low":
        # Handle interrupt intents immediately
        handler_function = intent_handlers.get(predicted_tag, intent_handlers["fallback"])
        response = handler_function(session_id, session, sentence)
        responses.append(response)
        
        # After handling interrupt, check if there are missing fields
        missing_fields_response = check_missing_fields(session)
        if missing_fields_response:
            responses.append(missing_fields_response)

    elif is_fixing and confidence != "low":
        # Handle slot filling
        order_id_fix = missing_field_context.get("order_id")
        field = missing_field_context.get("field")

        if not order_id_fix or not field:
            # Inconsistent session data
            responses.append("I'm sorry, something went wrong with your order. Let's start over.")
            session["is_fixing"] = False
            session["missing_field_context"] = {}
        else:
            # Extract the value using the updated function
            value = extract_field_value(field, cleaned_sentence)

            if DEBUG:
                logger.debug(f"Extracted value: {value}")

            if value:
                update_msg = update_order(session_id, order_id_fix, field, value)
                responses.append(update_msg)
                # After updating, check if there are more missing fields
                missing_fields_response = check_missing_fields(session)
                if missing_fields_response:
                    responses.append(missing_fields_response)
                else:
                    responses.append("Anything else I can help with?")
            else:
                responses.append(f"Sorry, I didn't understand what you're saying. {field_prompts[field]}")

    else:
        # Handle other intents
        if predicted_tag in intent_handlers and confidence != "low":
            handler_function = intent_handlers[predicted_tag]
            # Call the handler function
            response = handler_function(session_id, session, sentence)
            responses.append(response)

            # After handling, check if there are missing fields
            missing_fields_response = check_missing_fields(session)
            if missing_fields_response:
                responses.append(missing_fields_response)
            else:
                responses.append("Anything else I can help with?")
        elif confidence == "medium":
            # Moderate confidence: ask for confirmation
            responses.append(f"I think you want to {predicted_tag.replace('_', ' ')}. Is that correct?")
        elif confidence == "low":
            # Low confidence: use fallback intent
            responses.append("I'm sorry, I didn't understand that. Could you please rephrase or specify your request?")
        else:
            responses.append("I do not understand...")

    # Update session data
    update_session(sessions_collection, session, session_id)

    return jsonify({"response": "\n".join(responses), "session_id": session_id})

def _build_cors_preflight_response():
    response = jsonify({'status': 'OK'})
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:5001")
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:5001")
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

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

## Uncomment the following lines if you want to run the Flask app locally
if __name__ == '__main__':
   port = int(os.environ.get("PORT", 5000))
   app.run(host='0.0.0.0', port=port, debug=True)