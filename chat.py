import difflib
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

# Import the configuration
import config

# Import the database connection
from connect import database as db  # Ensure connect.py is correctly set up with get_db()

# Import the MenuFuzzer
from fuzzer import MenuFuzzer  # Ensure fuzzer.py is in the same directory or in Python path

app = Flask(__name__, static_folder='frontend/build')  # Set static_folder to frontend/build

# Updated CORS setup using config.py
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}}, supports_credentials=True)
#CORS(app, resources={r"/*": {"origins": config.CORS_ORIGINS}}, supports_credentials=True)

# Updated DEBUG flag using config.py
DEBUG = config.DEBUG

## Logging Configuration ##

# Create the root logger
root_logger = logging.getLogger()
root_logger.setLevel(config.ROOT_LOG_LEVEL)  # Default to INFO to reduce verbosity

# Create formatter
formatter = logging.Formatter(config.LOG_FORMAT)

# Create and configure console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(config.CONSOLE_LOG_LEVEL)  # Console shows INFO and above
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# Create and configure rotating file handler
file_handler = RotatingFileHandler(
    config.LOG_FILE,
    maxBytes=config.LOG_MAX_BYTES,
    backupCount=config.LOG_BACKUP_COUNT
)
file_handler.setLevel(config.FILE_LOG_LEVEL)  # File logs DEBUG if DEBUG=True
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Suppress verbose logs from external libraries
for logger_name in config.EXTERNAL_LOGGERS:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Create a logger for your application
logger = logging.getLogger(__name__)
logger.setLevel(config.FILE_LOG_LEVEL)  # Application logger

## End of Logging Configuration ##

# Load SpaCy model and Sentence-BERT model using config.py
nlp = spacy.load(config.SPACY_MODEL)
sentence_model = SentenceTransformer(config.SENTENCE_MODEL)

# Define similarity thresholds and weights using config.py
SIMILARITY_THRESHOLD_HIGH = config.SIMILARITY_THRESHOLD_HIGH
SIMILARITY_THRESHOLD_MEDIUM = config.SIMILARITY_THRESHOLD_MEDIUM

WEIGHT_COSINE = config.WEIGHT_COSINE
WEIGHT_EUCLIDEAN = config.WEIGHT_EUCLIDEAN
WEIGHT_JACCARD = config.WEIGHT_JACCARD

# Define field prompts using config.py
field_prompts = config.FIELD_PROMPTS

bot_name = config.BOT_NAME

# Function to clean sentences
def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence

# Load intents using config.py
with open(config.INTENTS_FILE, 'r') as f:
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
    
    # Handle the case where value is None or empty
    if value is None:
        if DEBUG:
            logger.debug(f"No value provided for field '{field}'. Skipping update.")
        return (f"No modifications provided for {field}.", False)
    
    # Special handling for "none" selection
    if isinstance(value, list) and len(value) == 1 and value[0].lower() == "none":
        if DEBUG:
            logger.debug(f"Setting {field} to empty array due to 'none' selection")
        
        # Use $set to explicitly set an empty array and mark field as completed
        update_data = {
            "$set": {
                field: [],
                f"{field}_completed": True  # Mark this field as intentionally empty
            }
        }
        
        result = db.get_db().Orders.update_one(
            {"session_id": session_id, "order_id": order_id},
            update_data
        )
        
        if DEBUG:
            logger.debug(f"Update result for 'none' selection: matched {result.matched_count}, modified {result.modified_count}")
        
        # Verify the update
        order = db.get_db().Orders.find_one({"session_id": session_id, "order_id": order_id})
        if DEBUG:
            logger.debug(f"Updated order after 'none' selection: {order}")
        
        # Check if all required fields are now complete
        required_fields = ["meats", "rice", "beans", "toppings"]
        fields_status = {}
        for req_field in required_fields:
            is_completed = (
                order.get(f"{req_field}_completed", False) or  # Check completion flag
                (isinstance(order.get(req_field), list) and len(order.get(req_field)) > 0)  # Or has values
            )
            fields_status[req_field] = is_completed
            
        if DEBUG:
            logger.debug(f"Fields completion status: {fields_status}")
        
        if all(fields_status.values()):
            db.get_db().Orders.update_one(
                {"session_id": session_id, "order_id": order_id},
                {"$set": {"completed": True}}
            )
            if DEBUG:
                logger.debug("Order marked as completed after 'none' selection")
        
        return (f"Set {field} to none for order {order_id}", True)
    
    # Normal case - update with provided values
    update_data = {
        "$set": {
            field: value,
            f"{field}_completed": True  # Mark field as completed with actual values
        }
    }
    
    result = db.get_db().Orders.update_one(
        {"session_id": session_id, "order_id": order_id},
        update_data
    )
    
    if DEBUG:
        logger.debug(f"Update result: matched {result.matched_count}, modified {result.modified_count}")
    
    # Verify the update
    order = db.get_db().Orders.find_one({"session_id": session_id, "order_id": order_id})
    if DEBUG:
        logger.debug(f"Updated order: {order}")
    
    # Check if all required fields are now complete
    required_fields = ["meats", "rice", "beans", "toppings"]
    fields_status = {}
    for req_field in required_fields:
        is_completed = (
            order.get(f"{req_field}_completed", False) or  # Check completion flag
            (isinstance(order.get(req_field), list) and len(order.get(req_field)) > 0)  # Or has values
        )
        fields_status[req_field] = is_completed
        
    if DEBUG:
        logger.debug(f"Fields completion status: {fields_status}")
    
    if all(fields_status.values()):
        db.get_db().Orders.update_one(
            {"session_id": session_id, "order_id": order_id},
            {"$set": {"completed": True}}
        )
        if DEBUG:
            logger.debug("Order marked as completed")
    
    return (f"Updated order {order_id} with {field}: {', '.join(value) if value else 'none'}", True)

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

    # Build the menu dictionary with item details and colloquialisms
    menu = {}
    name_to_colloquial = {}  # Mapping of colloquialisms to official names
    
    for item in all_items:
        name = item['name'].lower()
        price = item['size_details'][0].get('price', 0) if 'size_details' in item and item['size_details'] else 0
        category = item.get('category', 'other')
        colloquial = item.get('colloquialism', '').lower() if item.get('colloquialism') else None
        
        if isinstance(category, list):
            category = [c.lower() for c in category]
        elif isinstance(category, str):
            category = category.lower()
        else:
            category = 'other'

        # Add the main item name
        menu[name] = {
            'price': price,
            'category': category
        }

        # Add colloquialism mapping if it exists
        if colloquial:
            name_to_colloquial[colloquial] = name
            menu[colloquial] = menu[name]  # Share the same menu entry

    logger.debug(f"Loaded {len(menu)} menu items.")

    # Fetch addons with colloquialisms
    meats = []
    rice = []
    beans = []
    toppings = []

    # Helper function to add items and their colloquialisms to appropriate lists
    def add_to_category(item, category_list):
        category_list.append(item['name'].lower())
        if item.get('colloquialism'):
            category_list.append(item['colloquialism'].lower())

    # Fetch and process each category
    for item in menu_item_collection.find({'category': {'$regex': '^protein$', '$options': 'i'}}):
        add_to_category(item, meats)
    
    for item in menu_item_collection.find({'category': {'$regex': '^rice$', '$options': 'i'}}):
        add_to_category(item, rice)
    
    for item in menu_item_collection.find({'category': {'$regex': '^beans$', '$options': 'i'}}):
        add_to_category(item, beans)
    
    for item in menu_item_collection.find({'category': {'$regex': '^toppings$', '$options': 'i'}}):
        add_to_category(item, toppings)

    logger.debug(f"Meats list: {meats}")
    logger.debug(f"Rice list: {rice}")
    logger.debug(f"Beans list: {beans}")
    logger.debug(f"Toppings list: {toppings}")

    # Define add-ons
    addons = set(meats + rice + beans + toppings)

    # Define main items: all menu items not in add-ons
    main_items = [name for name in menu.keys() if name not in addons]

    logger.debug(f"Main items: {main_items}")

    return menu, meats, rice, beans, toppings, main_items, name_to_colloquial

# Initial fetch of menu data
menu, meats, rice, beans, toppings, main_items, name_to_colloquial = fetch_menu_data()

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

# Define weights as per similarity metrics
# Already defined earlier using config.py

# Define field prompts as per config.py
field_prompts = config.FIELD_PROMPTS

bot_name = config.BOT_NAME

# Initialize the logger
logger.info(f"Hi, I am an automated {config.BOT_NAME} AI menu. What would you like to order! (type 'quit' to exit)")

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
    return num_words.get(textnum.lower())

# Function to extract field values using PhraseMatcher
def extract_field_value(field, user_input):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    
    # Handle explicit "none" or "no" inputs
    user_input_lower = user_input.lower()
    if user_input_lower in ["none", "no", "nothing", "skip"]:
        return ["none"], True
    
    # Add field-specific patterns
    patterns = []
    if field == "meats":
        patterns = [nlp.make_doc(text) for text in meats]
    elif field == "rice":
        patterns = [nlp.make_doc(text) for text in rice]
    elif field == "beans":
        patterns = [nlp.make_doc(text) for text in beans]
    elif field == "toppings":
        patterns = [nlp.make_doc(text) for text in toppings]
    else:
        return None, False
        
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
            value = span.text.lower()
            
            # If the matched text is a colloquialism, get the official name
            official_name = name_to_colloquial.get(value)
            if official_name and official_name != value:
                value = official_name
                
            # If "none" is selected, it should be the only value
            if value == "none":
                return ["none"], True
                
            values.add(value)
            
        if DEBUG:
            logger.debug(f"Matches found: {values}")
        return list(values), True
    else:
        if DEBUG:
            logger.debug("No matches found.")
        return None, False

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
            
            # If main_item is a colloquialism, get the official name
            official_name = name_to_colloquial.get(main_item)
            if official_name:
                main_item = official_name
                
            items.append(main_item)

            # Use the single addon_matcher to find all add-ons
            matches = addon_matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                category = nlp.vocab.strings[match_id]  # Retrieve the category label
                addon = span.text.lower()
                
                # Convert colloquialism to official name if necessary
                official_addon = name_to_colloquial.get(addon, addon)
                
                # Map the category label back to lowercase for consistency
                category_key = category.lower()
                if official_addon not in item_addons[main_item][category_key]:
                    item_addons[main_item][category_key].append(official_addon)

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
                    "category": category_type,
                    "completed": False
                }
                if category_type == "entree":
                    order_document["meats"] = item_addons[item]["meats"] if item_addons[item]["meats"] else []
                    order_document["rice"] = item_addons[item]["rice"] if item_addons[item]["rice"] else []
                    order_document["beans"] = item_addons[item]["beans"] if item_addons[item]["beans"] else []
                    order_document["toppings"] = item_addons[item]["toppings"] if item_addons[item]["toppings"] else []

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
def handle_remove_order(session_id, session, sentence):
    # Check if the user wants to cancel the remove_order action
    if clean_sentence(sentence) == "cancel":
        # Clear the pending_action and reset session flags
        session.pop("pending_action", None)
        session['is_fixing'] = False
        session['missing_field_context'] = {}
        # Update the session in the database
        db.get_db()['Sessions'].replace_one({'session_id': session_id}, session, upsert=True)
        logger.info(f"Session {session_id}: User canceled the remove_order action.")
        return "Okay, I've canceled the removal request. How else can I assist you?"

    # Existing remove_order logic continues here...

    # Retrieve data from pending_action or directly from parameters
    pending_action = session.get('pending_action') or {}
    data = pending_action.get('data', {})
    order_ids = data.get('order_ids', [])
    features = data.get('features', {})

    # If not using pending_action, extract directly from the user's input
    if not order_ids:
        order_ids = extract_order_ids(sentence)
    if not features or all(not v for v in features.values()):
        features = extract_features(sentence)

    if DEBUG:
        logger.debug(f"Order IDs extracted: {order_ids}")
        logger.debug(f"Features extracted: {features}")

    missing_fields = []
    if not order_ids:
        missing_fields.append('order_id')
        if not features or all(not v for v in features.values()):
            # Both order_id and features are missing
            missing_fields.append('features')

    if not missing_fields:
        # Order IDs are present
        removed_items = []

        for oid in order_ids:
            order = db.get_db().Orders.find_one({"session_id": session_id, "order_id": oid})
            if not order:
                logger.debug(f"Order ID {oid} not found.")
                continue  # Skip if order not found

            if features and any(v for v in features.values()):
                # Features are specified, remove them from the order
                for field in ['meats', 'rice', 'beans', 'toppings']:
                    if features.get(field):
                        existing_addons = order.get(field, [])
                        addons_to_remove = [addon for addon in features[field] if addon in existing_addons]
                        if addons_to_remove:
                            # Remove specified addons
                            updated_addons = [addon for addon in existing_addons if addon not in addons_to_remove]
                            # Update the order in the database
                            db.get_db().Orders.update_one(
                                {"session_id": session_id, "order_id": oid},
                                {"$set": {field: updated_addons}}
                            )
                            removed_items.append(f"Removed {', '.join(addons_to_remove)} from Order ID {oid}.")
            else:
                # No features specified, remove the entire order
                db.get_db().Orders.delete_one({"session_id": session_id, "order_id": oid})
                removed_items.append(f"Removed Order ID {oid}.")

        if removed_items:
            # **Clear the pending_action and reset session flags after successful removal**
            session.pop("pending_action", None)
            session['is_fixing'] = False
            session['missing_field_context'] = {}
            # Update the session in the database
            db.get_db()['Sessions'].replace_one({'session_id': session_id}, session, upsert=True)

            return " ".join(removed_items) + " Anything else I can help with?"
        else:
            return "No valid items were removed from your order."
    else:
        # Set pending_action with the missing fields
        session['pending_action'] = {
            'action': 'remove_order',
            'missing_fields': missing_fields,
            'data': {
                'order_ids': order_ids,
                'features': features
            }
        }
        # Create appropriate prompt based on missing fields
        if 'order_id' in missing_fields and 'features' in missing_fields:
            response = "Sure, which Order ID would you like to remove, and what items or add-ons would you like to remove? ('cancel' to stop this request)"
        elif 'order_id' in missing_fields:
            response = "Sure, which Order ID would you like to remove items from? Please provide the Order ID. ('cancel' to stop this request)"
        elif 'features' in missing_fields:
            response = "What items or add-ons would you like to remove from your order? ('cancel' to stop this request)"
        else:
            response = "Please provide the necessary information to proceed with removing your order."

        # **Ensure that the session is updated in the database**
        db.get_db()['Sessions'].replace_one({'session_id': session_id}, session, upsert=True)

        return response

# Function to extract order IDs
def extract_order_ids(input_sentence):
    doc = nlp(input_sentence)
    order_ids = []

    # Extract numerical entities
    for ent in doc.ents:
        if ent.label_ == 'CARDINAL':
            try:
                order_ids.append(int(ent.text))
            except ValueError:
                num = text2int(ent.text)
                if num is not None:
                    order_ids.append(num)

    # Additionally, use regex to find standalone numbers possibly prefixed by keywords
    regex_ids = re.findall(r'\b(?:order\s*id\s*|order\s*number\s*|item\s*id\s*)?(\d+)\b', input_sentence, re.IGNORECASE)
    for rid in regex_ids:
        try:
            order_ids.append(int(rid))
        except ValueError:
            continue

    # Remove duplicates and ensure all IDs are positive integers
    order_ids = list(set([oid for oid in order_ids if oid > 0]))
    order_ids.sort()

    return order_ids

# Function to remove items by IDs
def remove_items_by_ids_and_features(session_id, order_ids, features):
    removed_items = []
    for oid in order_ids:
        order = db.get_db().Orders.find_one({"session_id": session_id, "order_id": oid})
        if not order:
            continue  # Skip if order not found
        
        # Iterate over each feature field
        for field in ['item', 'meats', 'rice', 'beans', 'toppings']:
            if features.get(field):
                if field == 'item':
                    # Remove the entire order if the item matches
                    if order.get('item') == features['item']:
                        removed_items.append(f"{order['item'].capitalize()} (Order ID {oid})")
                        db.get_db().Orders.delete_one({"session_id": session_id, "order_id": oid})
                        break  # Move to next order
                else:
                    existing_addons = order.get(field, [])
                    addons_to_remove = [addon for addon in features[field] if addon in existing_addons]
                    if addons_to_remove:
                        # Remove specified addons
                        existing_addons = [addon for addon in existing_addons if addon not in addons_to_remove]
                        # Update the order in the database
                        db.get_db().Orders.update_one(
                            {"session_id": session_id, "order_id": oid},
                            {"$set": {field: existing_addons}}
                        )
                        removed_items.append(f"Removed {', '.join(addons_to_remove)} from Order ID {oid}.")
    
    if removed_items:
        return " ".join(removed_items)
    else:
        return "No valid items were removed from your order."

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

# Function to predict intent
def predict_intent(user_input):
    cleaned_input = clean_sentence(user_input)
    input_embedding = sentence_model.encode([cleaned_input])[0]

    # Cosine Similarity
    cosine_similarities = np.dot(pattern_embeddings, input_embedding) / (np.linalg.norm(pattern_embeddings, axis=1) * np.linalg.norm(input_embedding))
    
    # Euclidean Distance
    euclidean_distances = np.linalg.norm(pattern_embeddings - input_embedding, axis=1)
    
    # Jaccard Similarity
    input_tokens = set(cleaned_input.split())
    jaccard_similarities = np.array([
        len(input_tokens.intersection(set(pattern.split()))) / len(input_tokens.union(set(pattern.split()))) 
        if len(input_tokens.union(set(pattern.split()))) > 0 else 0 
        for pattern in all_patterns
    ])
    
    # Word Mover's Distance (using SpaCy)
    wmd_distances = np.array([
        nlp(cleaned_input).similarity(nlp(pattern))
        for pattern in all_patterns
    ])
    
    # Normalize all metrics to [0,1] range
    scaler = MinMaxScaler()
    normalized_euclidean = scaler.fit_transform(euclidean_distances.reshape(-1, 1)).flatten()
    normalized_wmd = scaler.fit_transform(wmd_distances.reshape(-1, 1)).flatten()
    
    # Dynamic weight adjustment based on input length
    input_length = len(cleaned_input.split())
    if input_length <= 3:
        weights = config.SHORT_INPUT_WEIGHTS
    else:
        weights = config.LONG_INPUT_WEIGHTS

    # Combine scores with weights
    combined_scores = (
        weights['cosine'] * cosine_similarities +
        weights['euclidean'] * (1 - normalized_euclidean) +
        weights['jaccard'] * jaccard_similarities +
        weights['wmd'] * (1 - normalized_wmd)
    )

    # Get top 3 matches
    top_indices = np.argsort(combined_scores)[-3:][::-1]
    top_scores = combined_scores[top_indices]
    top_tags = [pattern_tags[i] for i in top_indices]

    if DEBUG:
        logger.debug(f"Top 3 intents: {list(zip(top_tags, top_scores))}")
        logger.debug(f"Input length: {input_length}, Using weights: {weights}")

    # Use a voting system for confidence levels
    if top_scores[0] >= SIMILARITY_THRESHOLD_HIGH:
        if len(top_scores) > 1 and (top_scores[0] - top_scores[1]) < config.CONFIDENCE_MARGIN:
            confidence = "medium"  # Close second match reduces confidence
        else:
            confidence = "high"
    elif top_scores[0] >= SIMILARITY_THRESHOLD_MEDIUM:
        confidence = "medium"
    else:
        confidence = "low"

    return top_tags[0], confidence

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
                # Check if field is incomplete
                is_field_complete = (
                    order.get(f"{field}_completed", False) or  # Check completion flag
                    (isinstance(order.get(field), list) and len(order.get(field)) > 0)  # Or has values
                )
                
                if not is_field_complete:
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

def checkout_order(session_id, session, sentence):
    # Get all orders for the session
    orders = list(db.get_db().Orders.find({"session_id": session_id}))
    
    if not orders:
        return "Your order is empty. Would you like to place an order?"
    
    # Calculate total cost and prepare order summary
    total_cost = 0
    order_summary = []
    
    for order in orders:
        item_name = order['item'].capitalize()
        price = order.get('price', 0)
        total_cost += price
        
        # Build order details string
        details = [item_name]
        if order.get('category') == 'entree':
            if order.get('meats'):
                details.append(f"with {', '.join(order['meats'])}")
            if order.get('rice'):
                details.append(f"{', '.join(order['rice'])}")
            if order.get('beans'):
                details.append(f"{', '.join(order['beans'])}")
            if order.get('toppings'):
                details.append(f"topped with {', '.join(order['toppings'])}")
        
        order_summary.append(f"• {' '.join(details)} (${price:.2f})")
    
    # Build the completion message
    completion_message = [
        "Thank you for your order! Here's your order summary:",
        "",  # Empty line for spacing
        *order_summary,  # Unpack the order details
        "",  # Empty line for spacing
        f"Total Cost: ${total_cost:.2f}",
        "",
        "Your order has been processed and will be ready for pickup soon!"
    ]
    
    # Delete the orders after successful checkout
    db.get_db().Orders.delete_many({"session_id": session_id})
    
    # Reset session data
    session['is_fixing'] = False
    session['missing_field_context'] = {}
    session['chat_length'] = 0
    session['last_activity'] = datetime.now(timezone.utc)
    
    return "\n".join(completion_message)

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
    # Get items from MenuItem collection
    menu_items = db.get_db().MenuItem.find({})
    
    # Organize items by category
    entree_items = set()
    side_items = set()
    
    for item in menu_items:
        # Handle both single category and list of categories
        categories = item['category'] if isinstance(item['category'], list) else [item['category']]
        name = item['name'].capitalize()
        
        # Check each category
        if 'Main' in categories:
            entree_items.add(name)
        if 'OptionalSide' in categories:
            side_items.add(name)
    
    # Sort items alphabetically
    entree_items = sorted(list(entree_items))
    side_items = sorted(list(side_items))
    
    # Build response message
    response = []
    
    # Add Entree section
    response.append("Entree Items:")
    response.append("• " + "\n• ".join(entree_items) + ".")
    
    # Add spacing between sections
    response.append("")  # Empty line for spacing
    
    # Add Sides section
    response.append("Side Items:")
    response.append("• " + "\n• ".join(side_items) + ".")
    
    return "\n".join(response)

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

def determine_field(addon):
    addon = addon.lower()
    if addon in meats:
        return 'meats'
    elif addon in rice:
        return 'rice'
    elif addon in beans:
        return 'beans'
    elif addon in toppings:
        return 'toppings'
    else:
        # Try to fuzzy match the addon to the closest category
        closest_match = None
        highest_similarity = 0
        
        all_items = {
            'meats': meats,
            'rice': rice,
            'beans': beans,
            'toppings': toppings
        }
        
        for category, items in all_items.items():
            for item in items:
                # Compute similarity between addon and item
                similarity = difflib.SequenceMatcher(None, addon, item.lower()).ratio()
                if similarity > highest_similarity and similarity > 0.8:  # 80% similarity threshold
                    highest_similarity = similarity
                    closest_match = category
        
        return closest_match
    
def get_nutrition_info(session_id, session, sentence):
    if DEBUG:
        logger.debug(f"Processing nutrition info request for sentence: '{sentence}'")
        logger.debug(f"Current menu_matcher patterns: {[p.text for p in menu_patterns]}")
        logger.debug(f"Available items in menu: {main_items}")
        logger.debug(f"Available addon items: {addons_list}")

    # Extract mentioned menu items
    doc = nlp(sentence)
    if DEBUG:
        logger.debug(f"SpaCy doc tokens: {[token.text for token in doc]}")
    
    # Try to match both menu items and addons
    menu_matches = menu_matcher(doc)
    addon_matches = addon_matcher(doc)
    
    if DEBUG:
        logger.debug(f"Menu matcher matches: {[(doc[start:end].text, start, end) for _, start, end in menu_matches]}")
        logger.debug(f"Addon matcher matches: {[(doc[start:end].text, start, end) for _, start, end in addon_matches]}")

    mentioned_items = set()
    
    # Add main menu items
    for match_id, start, end in menu_matches:
        span = doc[start:end]
        item_name = span.text.lower()
        if DEBUG:
            logger.debug(f"Found main menu item: {item_name}")
        
        # If the item is a colloquialism, get the official name
        official_name = name_to_colloquial.get(item_name, item_name)
        if DEBUG:
            logger.debug(f"Official name for {item_name}: {official_name}")
        mentioned_items.add(official_name)
    
    # Add addon items (like meats, rice, etc.)
    for match_id, start, end in addon_matches:
        span = doc[start:end]
        item_name = span.text.lower()
        category = nlp.vocab.strings[match_id].lower()
        if DEBUG:
            logger.debug(f"Found addon item: {item_name} from category: {category}")
        
        # If the item is a colloquialism, get the official name
        official_name = name_to_colloquial.get(item_name, item_name)
        if DEBUG:
            logger.debug(f"Official name for addon {item_name}: {official_name}")
        mentioned_items.add(official_name)

    if DEBUG:
        logger.debug(f"Final mentioned items after processing: {mentioned_items}")
    
    if not mentioned_items:
        return "I couldn't identify which menu item you'd like nutritional information for. Could you please specify the item?"
    
    responses = []
    
    # Query each mentioned item
    for item_name in mentioned_items:
        if DEBUG:
            logger.debug(f"Querying MenuItem collection for: {item_name}")
            
        # First get the MenuItem document to get its ID
        # Try exact match first
        menu_item = db.get_db().MenuItem.find_one({"name": {"$regex": f"^{item_name}$", "$options": "i"}})
        
        # If no exact match, try matching as an ingredient/addon
        if not menu_item:
            menu_item = db.get_db().MenuItem.find_one({
                "$or": [
                    {"category": "protein", "name": {"$regex": f"^{item_name}$", "$options": "i"}},
                    {"category": "rice", "name": {"$regex": f"^{item_name}$", "$options": "i"}},
                    {"category": "beans", "name": {"$regex": f"^{item_name}$", "$options": "i"}},
                    {"category": "toppings", "name": {"$regex": f"^{item_name}$", "$options": "i"}}
                ]
            })
        
        if DEBUG:
            logger.debug(f"MenuItem query result for {item_name}: {menu_item}")
        
        if not menu_item:
            responses.append(f"Sorry, I couldn't find nutritional information for {item_name}.")
            continue
        
        # Get nutritional information using the menuitem ID
        nutrition_info = db.get_db().NutritionalInformation.find_one({"menuitemID": menu_item['_id']})
        
        if DEBUG:
            logger.debug(f"NutritionalInformation query result for {item_name}: {nutrition_info}")
        
        if not nutrition_info:
            responses.append(f"Sorry, nutritional information for {item_name} is not available at the moment.")
            continue
        
        # Format the nutritional information
        nutrition_text = f"\nNutritional Information for {item_name.capitalize()}:\n"
        nutrition_text += f"• Calories: {nutrition_info['total_calories']}\n"
        nutrition_text += f"• Fat: {nutrition_info['total_fat']}g (Calories from Fat: {nutrition_info['total_fat_calories']})\n"
        nutrition_text += f"  - Saturated Fat: {nutrition_info['saturated_fat']}g\n"
        nutrition_text += f"  - Trans Fat: {nutrition_info['trans_fat']}g\n"
        nutrition_text += f"• Cholesterol: {nutrition_info['cholesterol']}mg\n"
        nutrition_text += f"• Sodium: {nutrition_info['sodium']}mg\n"
        nutrition_text += f"• Total Carbohydrates: {nutrition_info['carbohydrates']}g\n"
        nutrition_text += f"  - Dietary Fiber: {nutrition_info['dietary_fiber']}g\n"
        nutrition_text += f"  - Sugars: {nutrition_info['sugars']}g\n"
        nutrition_text += f"• Protein: {nutrition_info['protein']}g"
        
        responses.append(nutrition_text)
    
    if not responses:
        return "I couldn't find any nutritional information. Could you please specify which menu item you're interested in?"
    
    return "\n".join(responses)

def check_price(session_id, session, sentence):
    # Check if the user wants to cancel the price check action
    if clean_sentence(sentence) == "cancel":
        # Clear the pending_action and reset session flags
        session.pop("pending_action", None)
        session['is_fixing'] = False
        session['missing_field_context'] = {}
        # Update the session in the database
        db.get_db()['Sessions'].replace_one({'session_id': session_id}, session, upsert=True)
        logger.info(f"Session {session_id}: User canceled the price check action.")
        return "Okay, I've canceled the price check request. How else can I assist you?"

    # Use SpaCy to identify menu items in the sentence
    doc = nlp(sentence)
    items = set()
    
    # Check for main menu items
    menu_matches = menu_matcher(doc)
    for match_id, start, end in menu_matches:
        span = doc[start:end]
        item_name = span.text.lower()
        # If the item is a colloquialism, get the official name
        official_name = name_to_colloquial.get(item_name, item_name)
        items.add(official_name)
    
    # Also check for add-on items
    addon_matches = addon_matcher(doc)
    for match_id, start, end in addon_matches:
        span = doc[start:end]
        item_name = span.text.lower()
        # If the item is a colloquialism, get the official name
        official_name = name_to_colloquial.get(item_name, item_name)
        items.add(official_name)

    if DEBUG:
        logger.debug(f"Items found for price check: {items}")

    if not items:
        # No items found, set pending action
        session['pending_action'] = {
            'action': 'check_price',
            'missing_fields': ['item'],
            'data': {}
        }
        # Update the session in the database
        db.get_db()['Sessions'].replace_one({'session_id': session_id}, session, upsert=True)
        return "Which item would you like to know the price for? ('cancel' to stop this request)"

    # If we found items, look up their prices
    price_responses = []
    for item_name in items:
        # Query the MenuItem collection
        menu_item = db.get_db().MenuItem.find_one(
            {"name": {"$regex": f"^{item_name}$", "$options": "i"}}
        )
        
        if not menu_item:
            # Try matching as an ingredient/addon
            menu_item = db.get_db().MenuItem.find_one({
                "$or": [
                    {"category": "protein", "name": {"$regex": f"^{item_name}$", "$options": "i"}},
                    {"category": "rice", "name": {"$regex": f"^{item_name}$", "$options": "i"}},
                    {"category": "beans", "name": {"$regex": f"^{item_name}$", "$options": "i"}},
                    {"category": "toppings", "name": {"$regex": f"^{item_name}$", "$options": "i"}}
                ]
            })

        if menu_item:
            # Get the base price from size_details
            if 'size_details' in menu_item and menu_item['size_details']:
                base_price = menu_item['size_details'][0].get('price', 0)
                if base_price > 0:  # Only show price if it's greater than 0
                    price_responses.append(f"The price of {item_name.capitalize()} is ${base_price:.2f}")
                else:
                    price_responses.append(f"{item_name.capitalize()} is included with your order")
            else:
                price_responses.append(f"Sorry, I couldn't find pricing information for {item_name}")
        else:
            price_responses.append(f"Sorry, I couldn't find {item_name} in our menu")

    # Clear the pending action
    session.pop("pending_action", None)
    # Update the session in the database
    db.get_db()['Sessions'].replace_one({'session_id': session_id}, session, upsert=True)

    return "\n".join(price_responses)

def extract_modifications(user_input):
    modifications = defaultdict(list)
    
    # Use SpaCy's dependency parsing to identify replace actions
    doc = nlp(user_input.lower())
    
    for token in doc:
        if token.text in ['replace', 'change', 'swap', 'modify']:
            # Look for the object to replace (direct object)
            obj = None
            for child in token.children:
                if child.dep_ == 'dobj':
                    obj = child.text
                    break
            # Look for the new value (prep object)
            prep = None
            for child in token.children:
                if child.dep_ == 'prep' and child.text in ['with', 'for', 'to']:
                    for prep_child in child.children:
                        if prep_child.dep_ == 'pobj':
                            prep = prep_child.text
                            break
            if obj and prep:
                # Determine which field the object belongs to
                field = determine_field(obj)
                if field:
                    # Remove duplicates while keeping order
                    if prep not in modifications[field]:
                        modifications[field].append(prep)
                    if obj not in modifications[field]:
                        modifications[field].append(obj)  # Including obj for removal
    
    # If no replace actions detected, check for direct add-ons
    if not modifications:
        for field in ['meats', 'rice', 'beans', 'toppings']:
            addons = extract_field_value(field, user_input)[0]  # Use the first element of the tuple
            if addons:
                modifications[field].extend(addon for addon in addons if addon not in modifications[field])
    
    # Convert defaultdict to regular dict and ensure unique values while preserving order
    final_modifications = {}
    for field, addons in modifications.items():
        # Use dict.fromkeys to remove duplicates while preserving order
        final_modifications[field] = list(dict.fromkeys(addons))
    
    if DEBUG:
        logger.debug(f"Extracted modifications: {final_modifications}")
    
    return final_modifications if final_modifications else None

def modify_order_handler(session_id, session, sentence):
    # Retrieve data from pending_action or directly from parameters
    pending_action = session.get('pending_action') or {}
    data = pending_action.get('data', {})
    order_ids = data.get('order_ids', [])
    modifications = data.get('modifications', {})
    
    # If not using pending_action, extract directly
    if not order_ids:
        order_ids = extract_order_ids(sentence)
    if not modifications:
        modifications = extract_modifications(sentence) or {}  # Ensure modifications is a dict
    
    if DEBUG:
        logger.debug(f"Modifying Order IDs: {order_ids}")
        logger.debug(f"With Modifications: {modifications}")
    
    missing_fields = []
    if not order_ids:
        missing_fields.append('order_id')
    if not modifications:
        missing_fields.append('modifications')
    
    if not missing_fields:
        # Both Order ID and modifications are specified
        # Proceed to modify the order
        valid_order_ids = []
        for oid in order_ids:
            order = db.get_db().Orders.find_one({"session_id": session_id, "order_id": oid})
            if order:
                valid_order_ids.append(oid)
            else:
                logger.debug(f"Order ID {oid} not found.")
        
        if not valid_order_ids:
            return "I couldn't find any valid Order IDs in your request."
        
        # Proceed to update each valid order
        confirmation_messages = []
        for oid in valid_order_ids:
            order = db.get_db().Orders.find_one({"session_id": session_id, "order_id": oid})
            if not order:
                continue  # Skip if order not found
            
            # Iterate over each modification field
            for field, addons in modifications.items():
                existing_addons = order.get(field, [])
                # Remove addons specified in modifications
                addons_to_remove = [addon for addon in addons if addon in existing_addons]
                # Add addons specified in modifications that aren't already present
                addons_to_add = [addon for addon in addons if addon not in existing_addons]
                
                if addons_to_remove:
                    existing_addons = [addon for addon in existing_addons if addon not in addons_to_remove]
                if addons_to_add:
                    existing_addons.extend(addons_to_add)
                
                # Update the order in the database
                db.get_db().Orders.update_one(
                    {"session_id": session_id, "order_id": oid},
                    {"$set": {field: existing_addons}}
                )
                
                if addons_to_remove and addons_to_add:
                    confirmation_messages.append(f"Replaced {', '.join(addons_to_remove)} with {', '.join(addons_to_add)} in Order ID {oid}.")
                elif addons_to_remove:
                    confirmation_messages.append(f"Removed {', '.join(addons_to_remove)} from Order ID {oid}.")
                elif addons_to_add:
                    confirmation_messages.append(f"Added {', '.join(addons_to_add)} to Order ID {oid}.")
        
        if confirmation_messages:
            return " ".join(confirmation_messages)
        else:
            return "No valid modifications were made to your order."
    else:
        # Set pending_action with the missing fields
        session['pending_action'] = {
            'action': 'modify_order',
            'missing_fields': missing_fields,
            'data': {
                'order_ids': order_ids,
                'modifications': modifications or {}
            }
        }
        # Create appropriate prompt based on missing fields
        if 'order_id' in missing_fields and 'modifications' in missing_fields:
            return "Sure, which Order ID would you like to modify, and what changes would you like to make? ('cancel' to stop this request)"
        elif 'order_id' in missing_fields:
            return "Sure, which Order ID would you like to modify? Please provide the Order ID. ('cancel' to stop this request)"
        elif 'modifications' in missing_fields:
            return "What modifications would you like to make to your order? ('cancel' to stop this request)"

def handle_modify_order(session_id, session, sentence):
    # Check if the user wants to cancel the modify_order action
    if clean_sentence(sentence) == "cancel":
        # Clear the pending_action and reset session flags
        session.pop("pending_action", None)
        session['is_fixing'] = False
        session['missing_field_context'] = {}
        # Update the session in the database
        db.get_db()['Sessions'].replace_one({'session_id': session_id}, session, upsert=True)
        logger.info(f"Session {session_id}: User canceled the modify_order action.")
        return "Okay, I've canceled the modification request. How else can I assist you?"

    try:
        # Retrieve data from pending_action or directly from parameters
        pending_action = session.get('pending_action') or {}
        data = pending_action.get('data', {})
        order_ids = data.get('order_ids', [])
        modifications = data.get('modifications', {})

        # If not using pending_action, extract directly from the user's input
        if not order_ids:
            order_ids = extract_order_ids(sentence)
        if not modifications:
            modifications = extract_modifications(sentence)
        
        if DEBUG:
            logger.debug(f"Order IDs extracted: {order_ids}")
            logger.debug(f"Modifications extracted: {modifications}")

        missing_fields = []
        if not order_ids:
            missing_fields.append('order_id')
        if not modifications:
            missing_fields.append('modifications')

        if not missing_fields:
            # Both Order ID and modifications are specified
            valid_order_ids = []
            for oid in order_ids:
                order = db.get_db().Orders.find_one({"session_id": session_id, "order_id": oid})
                if order:
                    valid_order_ids.append(oid)
                else:
                    logger.debug(f"Order ID {oid} not found.")

            if not valid_order_ids:
                return "I couldn't find any valid Order IDs in your request."

            # Proceed to update each valid order
            confirmation_messages = []
            for oid in valid_order_ids:
                order = db.get_db().Orders.find_one({"session_id": session_id, "order_id": oid})
                if not order:
                    continue

                # Iterate over each modification field
                for field, addons in modifications.items():
                    existing_addons = order.get(field, [])
                    # Remove addons specified in modifications that exist in current order
                    addons_to_remove = [addon for addon in addons if addon in existing_addons]
                    # Add new addons specified in modifications
                    addons_to_add = [addon for addon in addons if addon not in existing_addons and addon not in addons_to_remove]

                    if addons_to_remove or addons_to_add:
                        # Update existing_addons list
                        updated_addons = [addon for addon in existing_addons if addon not in addons_to_remove]
                        updated_addons.extend(addons_to_add)

                        # Update the order in the database
                        db.get_db().Orders.update_one(
                            {"session_id": session_id, "order_id": oid},
                            {"$set": {field: updated_addons}}
                        )

                        if addons_to_remove and addons_to_add:
                            confirmation_messages.append(f"Replaced {', '.join(addons_to_remove)} with {', '.join(addons_to_add)} in Order ID {oid}.")
                        elif addons_to_remove:
                            confirmation_messages.append(f"Removed {', '.join(addons_to_remove)} from Order ID {oid}.")
                        elif addons_to_add:
                            confirmation_messages.append(f"Added {', '.join(addons_to_add)} to Order ID {oid}.")

            if confirmation_messages:
                # Clear the pending_action and reset session flags after successful modification
                session.pop("pending_action", None)
                session['is_fixing'] = False
                session['missing_field_context'] = {}
                # Update the session in the database
                db.get_db()['Sessions'].replace_one({'session_id': session_id}, session, upsert=True)

                return " ".join(confirmation_messages) + " Anything else I can help with?"
            else:
                return "No valid modifications were made to your order."
        else:
            # Set pending_action with the missing fields
            session['pending_action'] = {
                'action': 'modify_order',
                'missing_fields': missing_fields,
                'data': {
                    'order_ids': order_ids,
                    'modifications': modifications or {}
                }
            }
            
            # Create appropriate prompt based on missing fields
            if 'order_id' in missing_fields and 'modifications' in missing_fields:
                response = "Sure, which Order ID would you like to modify, and what changes would you like to make? ('cancel' to stop this request)"
            elif 'order_id' in missing_fields:
                response = "Sure, which Order ID would you like to modify? Please provide the Order ID. ('cancel' to stop this request)"
            elif 'modifications' in missing_fields:
                response = "What modifications would you like to make to your order? ('cancel' to stop this request)"
            else:
                response = "Please provide the necessary information to proceed with modifying your order."

            # Ensure that the session is updated in the database
            db.get_db()['Sessions'].replace_one({'session_id': session_id}, session, upsert=True)

            return response
            
    except Exception as e:
        logger.exception(f"Error in handle_modify_order: {str(e)}")
        return "Sorry, there was an error processing your modification request. Please try again."

# Mapping of intent tags to handler functions
intent_handlers = {
    "order": process_order,
    "remove_order": handle_remove_order,
    "modify_order": handle_modify_order,
    "reset_order": handle_remove_order,  # If reset_order uses the same handler
    "checkout": checkout_order,
    "check_order": check_order,
    "restart_order": handle_remove_order,  # If restart_order uses the same handler
    "show_menu": check_menu,
    "ask_options": provide_options,
    "nutrition_info": get_nutrition_info,
    "check_price": check_price,
    "vegan_options": lambda sid, s, sen: random.choice(intents_dict['vegan_options']['responses']),
    "fallback": lambda sid, s, sen: random.choice(intents_dict['fallback']['responses'])
}

# Define the inactivity timeout using config.py
INACTIVITY_TIMEOUT = timedelta(minutes=config.INACTIVITY_TIMEOUT_MINUTES)

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
            if last_activity.tzinfo is None:
                last_activity = last_activity.replace(tzinfo=timezone.utc)

            if datetime.now(timezone.utc) - last_activity > INACTIVITY_TIMEOUT:
                # Session has been inactive for more than the timeout
                db_instance.Orders.delete_many({'session_id': session_id})
                if DEBUG:
                    logger.debug(f"Session {session_id} has been inactive for over {INACTIVITY_TIMEOUT}. Orders deleted.")
                # Reset session data
                session = {
                    'session_id': session_id,
                    'is_fixing': False,
                    'missing_field_context': {},
                    'chat_length': 0,
                    'last_activity': datetime.now(timezone.utc)
                }
                # Remove pending_action from session
                session.pop('pending_action', None)
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
            # Remove pending_action from session
            session.pop('pending_action', None)
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
        # Remove pending_action from session
        session.pop('pending_action', None)
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

    # Handle 'cancel' command first
    if cleaned_sentence == "cancel":
        if session.get("pending_action"):
            # Clear the pending action and reset session flags
            session.pop("pending_action", None)
            session['is_fixing'] = False
            session['missing_field_context'] = {}
            # Update the session in the database
            db.get_db()['Sessions'].replace_one({'session_id': session_id}, session, upsert=True)
            logger.info(f"Session {session_id}: User issued a global cancel command.")
            responses.append("Okay, I've canceled your current request. How else can I assist you?")
            # Check for any missing fields and prompt accordingly
            missing_fields_response = check_missing_fields(session)
            if missing_fields_response:
                responses.append(missing_fields_response)
            update_session(sessions_collection, session, session_id)
            return jsonify({"response": "\n".join(responses), "session_id": session_id})
        else:
            # No pending action to cancel
            responses.append("There is no active action to cancel. How can I help you?")
            update_session(sessions_collection, session, session_id)
            return jsonify({"response": "\n".join(responses), "session_id": session_id})

    # Predict intent
    predicted_tag, confidence = predict_intent(sentence)

    if DEBUG:
        logger.debug(f"Predicted intent: {predicted_tag}, Confidence: {confidence}")

    # Handle pending actions first
    if session.get("pending_action"):
        # Existing pending action
        pending_action = session.get("pending_action")
        action_type = pending_action.get("action")

        # Depending on the action_type, call the respective handler
        if action_type == "modify_order":
            response = handle_modify_order(session_id, session, sentence)
            responses.append(response)
        elif action_type == "remove_order":
            response = handle_remove_order(session_id, session, sentence)
            responses.append(response)
        elif action_type == "check_price":
            response = check_price(session_id, session, sentence)
            responses.append(response)

        missing_fields_response = check_missing_fields(session)
        if missing_fields_response:
            responses.append(missing_fields_response)

        update_session(sessions_collection, session, session_id)
        return jsonify({"response": "\n".join(responses), "session_id": session_id})

    # Check if we're in fixing mode (filling addons)
    if is_fixing and missing_field_context:
        field = missing_field_context.get("field")
        order_id = missing_field_context.get("order_id")
        
        # Try to extract addon value first
        values, success = extract_field_value(field, sentence)
        
        if success:
            # Successfully found addon(s), update the order
            update_message, was_updated = update_order(session_id, order_id, field, values)
            responses.append(update_message)
            
            if was_updated:
                # Only check for missing fields if the update was successful
                missing_fields_response = check_missing_fields(session)
                if missing_fields_response:
                    responses.append(missing_fields_response)
                else:
                    # Order is complete
                    order = db.get_db().Orders.find_one({"session_id": session_id, "order_id": order_id})
                    if order and order.get("completed"):
                        responses.append("Your order is complete! Would you like anything else?")
            
            update_session(sessions_collection, session, session_id)
            return jsonify({"response": "\n".join(responses), "session_id": session_id})

    # If we're here, either we're not fixing addons or addon extraction failed
    # Now handle interrupt intents
    # NEW: Add check_price to interrupt_intents
    interrupt_intents = ["remove_order", "modify_order", "restart_order", "show_menu", 
                        "check_order", "ask_options", "reset_order", "nutrition_info", 
                        "check_price"]

    if predicted_tag in interrupt_intents and confidence != "low":
        handler_function = intent_handlers.get(predicted_tag, intent_handlers["fallback"])
        response = handler_function(session_id, session, sentence)
        responses.append(response)
        
        missing_fields_response = check_missing_fields(session)
        if missing_fields_response:
            responses.append(missing_fields_response)
        else:
            responses.append("Anything else I can help with?")
            
        update_session(sessions_collection, session, session_id)
        return jsonify({"response": "\n".join(responses), "session_id": session_id})

    # Handle other intents
    if predicted_tag in intent_handlers and confidence != "low":
        handler_function = intent_handlers.get(predicted_tag)
        # Call the handler function
        response = handler_function(session_id, session, sentence)
        responses.append(response)

        # After handling, check if there are missing fields
        missing_fields_response = check_missing_fields(session)
        if missing_fields_response:
            responses.append(missing_fields_response)
        else:
            order = db.get_db().Orders.find_one(
                {"session_id": session_id},
                sort=[("_id", -1)]
            )
            if order and order.get("completed"):
                responses.append("Your order is complete! Would you like anything else?")
            else:
                responses.append("Anything else I can help with?")

    elif confidence == "medium":
        # Moderate confidence: ask for confirmation
        responses.append(f"I think you want to {predicted_tag.replace('_', ' ')}. Is that correct?")
    else:
        responses.append("I'm sorry, I didn't understand that. Could you please rephrase or specify your request?")

    # Update session data
    update_session(sessions_collection, session, session_id)
    return jsonify({"response": "\n".join(responses), "session_id": session_id})

def _build_cors_preflight_response():
    response = jsonify({'status': 'OK'})
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
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
    
@app.route('/reset_session', methods=['POST'])
def reset_session():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            logger.warning("reset_session called without session_id.")
            return jsonify({"error": "session_id not provided."}), 400

        # Access collections via db.get_db()
        db_instance = db.get_db()
        result_orders = db_instance.Orders.delete_many({"session_id": session_id})
        result_sessions = db_instance.Sessions.delete_one({"session_id": session_id})

        logger.info(f"Deleted {result_orders.deleted_count} orders and {result_sessions.deleted_count} sessions for session_id {session_id}.")

        return jsonify({"status": "Session reset successfully."}), 200

    except Exception as e:
        logger.error(f"Error resetting session {session_id}: {e}")
        return jsonify({"error": "Internal server error."}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify that the server is running.
    """
    try:
        # You can add more comprehensive health checks here if needed
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "error"}), 500


## Uncomment the following lines if you want to run the Flask app locally
#if __name__ == '__main__':
#    port = int(os.environ.get("PORT", 5000))
#    app.run(host='0.0.0.0', port=port, debug=DEBUG)