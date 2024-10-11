import random
import json
import torch
import re
import spacy
import numpy as np
from sklearn.cluster import KMeans
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer, util
from collections import Counter

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

order_id = 1
is_fixing = False

menu = {
    "bowl": 4.99,
    "burrito": 6.49,
}

meats = ["smoked brisket", "steak", "carnitas", "chicken", "beef barbacoa", "sofritas", "fajita veggies"]
rice = ["white rice", "brown rice", "none"]
beans = ["black beans", "pinto beans", "none"]
toppings = ["guacamole", "fresh tomato salsa", "roasted chili-corn salsa", "tomatillo-green chili salsa",
            "tomatillo-red chili salsa", "sour cream", "fajita veggies", "cheese", "romaine lettuce",
            "queso blanco", "none"]

addons_list = meats + rice + beans + toppings

orders = []

missing_field_context = {}

field_prompts = {
    "meats": "What kind of meat would you like?",
    "rice": "What type of rice would you like?",
    "beans": "Would you like black beans, pinto beans, or none?",
    "toppings": "What toppings would you like?"
}

bot_name = "Chipotle"
print("Let's chat! (type 'quit' to exit)")

def remove_item_from_order(order_id):
    global orders
    for i, order in enumerate(orders):
        if order["id"] == order_id:
            del orders[i]
            # Update order IDs
            for index, order in enumerate(orders):
                order["id"] = index + 1
            return True
    return False

def check_missing_fields():
    global is_fixing
    for order in orders:
        for field in ["meats", "rice", "beans", "toppings"]:
            if not order[field]:  # Checks if the list is empty
                missing_field_context["order_id"] = order["id"]
                missing_field_context["field"] = field
                prompt_user_for_missing_field(order["id"], field)
                is_fixing = True
                return  # Stop after finding the first missing field
    is_fixing = False  # No missing fields left
    print(f"{bot_name}: Anything else I can help with?")

def prompt_user_for_missing_field(order_id, field):
    print(f"{bot_name}: For order {order_id}, {field_prompts[field]}")

def update_order(order_id, field, value):
    for order in orders:
        if order["id"] == order_id:
            order[field] = value
            print(f"{bot_name}: Updated order {order_id} with {field}: {value}")
            break
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
        if meat in doc_text:
            meats_op.append(meat)
    for rice_type in rice:
        if rice_type in doc_text:
            rice_op.append(rice_type)
    for bean in beans:
        if bean in doc_text:
            beans_op.append(bean)
    for topping in toppings:
        if topping in doc_text:
            toppings_op.append(topping)

    if items:
        for item in items:
            price = menu[item]
            for _ in range(quantity):
                orders.append({
                    "id": order_id,
                    "item": item,
                    "price": price,
                    "meats": meats_op if meats_op else [],
                    "rice": rice_op if rice_op else [],
                    "beans": beans_op if beans_op else [],
                    "toppings": toppings_op if toppings_op else []
                })
                order_id += 1
        return f"Added {quantity} {', '.join(items)} to your order."
    else:
        return "Sorry, I didn't understand the items you want to order."

def extract_item_spacy(input_sentence):
    doc = nlp(input_sentence)
    for token in doc:
        if token.lemma_ in menu:
            return token.lemma_
    return None

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
    matches = matcher(doc)
    if matches:
        # Collect all matching phrases
        values = set()
        for match_id, start, end in matches:
            span = doc[start:end]
            values.add(span.text.lower())
        return list(values)
    else:
        return None

def display_current_order():
    if orders:
        print(f"{bot_name}: Here is your current order:")
        for order in orders:
            meats = ', '.join(order['meats']) if order['meats'] else 'None'
            rice = ', '.join(order['rice']) if order['rice'] else 'None'
            beans = ', '.join(order['beans']) if order['beans'] else 'None'
            toppings = ', '.join(order['toppings']) if order['toppings'] else 'None'
            print(f"Order ID: {order['id']}, Item: {order['item']}, Meats: {meats}, Rice: {rice}, Beans: {beans}, Toppings: {toppings}")
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
    global orders
    removed_items = []
    for oid in order_ids:
        for i, order in enumerate(orders):
            if order["id"] == oid:
                removed_items.append(f"{order['item']} (Order ID {oid})")
                del orders[i]
                break
    # Update the IDs of the remaining orders
    for index, order in enumerate(orders):
        order["id"] = index + 1
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
        if meat in doc_text:
            features["meats"].append(meat)
    for rice_type in rice:
        if rice_type in doc_text:
            features["rice"].append(rice_type)
    for bean in beans:
        if bean in doc_text:
            features["beans"].append(bean)
    for topping in toppings:
        if topping in doc_text:
            features["toppings"].append(topping)

    return features

def remove_items_by_features(features):
    global orders
    removed_items = []
    new_orders = []

    for order in orders:
        match = True
        if features["item"] and order["item"] != features["item"]:
            match = False
        if features["meats"] and not any(meat in order["meats"] for meat in features["meats"]):
            match = False
        if features["rice"] and not any(rice in order["rice"] for rice in features["rice"]):
            match = False
        if features["beans"] and not any(bean in order["beans"] for bean in features["beans"]):
            match = False
        if features["toppings"] and not any(topping in order["toppings"] for topping in features["toppings"]):
            match = False

        if match:
            removed_items.append(f"{order['item']} (Order ID {order['id']})") # More verbose update added
        else:
            new_orders.append(order)

    orders = new_orders

    # Update the IDs of the remaining orders
    for index, order in enumerate(orders):
        order["id"] = index + 1

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
                #print(f"{bot_name}: hehe {field_prompts[field]}")
                
            continue
        else: # not certain this is necesary, may cause a bug or prevent one lol? replace with exception block eventually
            # If value not found, proceed to process intents
            pass

    # Compute embedding with clustering
    predicted_tag = predict_intent_with_clustering(sentence)

    if DEBUG:
        print(f"[DEBUG] Input embedding shape: {predicted_tag}")

    if predicted_tag:
        for intent in intents['intents']:
            if predicted_tag == intent["tag"]:
                if predicted_tag == "order" and not is_fixing:
                    response = process_order_spacy(sentence)
                    chat_length += 1
                    print(f"{bot_name}: {response}")
                    if DEBUG:
                        print(f"[DEBUG] Current orders: {orders}")
                    check_missing_fields()

                elif predicted_tag == "remove_id" and not is_fixing:
                    order_ids = extract_order_ids(sentence)
                    if order_ids:
                        removed_items = remove_items_by_ids(order_ids)
                        if removed_items:
                            print(f"{bot_name}: I've removed the following items from your order: {', '.join(removed_items)}.")
                            if DEBUG:
                                print(f"[DEBUG] Current orders: {orders}")
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
                            print(f"[DEBUG] Current orders: {orders}")
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
                        print(f"{bot_name}: Sorry, I dont understand what you are saying. {field_prompts[field]}")
                    else:
                        print(f"{bot_name}: {random.choice(intent['responses'])}")

                
    else:
        print(f"{bot_name}: I do not understand...")

    if chat_length > 0 and not is_fixing and predicted_tag != "checkout":
        print(f"{bot_name}: Anything else I can help with?")