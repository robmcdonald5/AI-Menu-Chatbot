import random
import json
import torch
import re
import spacy
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer, util

# Load SpaCy model and Sentence-BERT model
nlp = spacy.load('en_core_web_sm')
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Toggleable debug mode
DEBUG = True  # Set to True to enable debug output

with open('intents.json', 'r') as f:
    intents = json.load(f)

# Precompute intent embeddings
intent_sentences = []
intent_tags = []

for intent in intents['intents']:
    intent_tags.append(intent['tag'])
    sentence = intent['representative_sentence']
    intent_sentences.append(sentence)

intent_embeddings = sentence_model.encode(intent_sentences, convert_to_tensor=True)

order_id = 1
is_fixing = False

menu = {
    "bowl": 4.99,
    "burrito": 6.49,
}

meats = ["smoked brisket", "steak", "carnitas", "chicken", "beef barbacoa", "sofritas", "fajita veggies"]
rice = ["white rice", "brown rice", "none"]
beans = ["black beans", "pinto beans", "none"]
toppings = ["guacamole", "fresh tomato salsa", "roasted chili-corn salsa", "tomatillo-green chili salsa", "tomatillo-red chili salsa", "sour cream", "fajita veggies", "cheese", "romaine lettuce", "queso blanco", "none"]

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

def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence

def remove_item_from_order(item):
    global orders
    for i, order in enumerate(orders):
        if order["item"] == item:
            del orders[i]
            return True
    return False

def check_missing_fields():
    for order in orders:
        for field in ["meats", "rice", "beans", "toppings"]:
            if order[field] == "":
                missing_field_context["order_id"] = order["id"]
                missing_field_context["field"] = field
                prompt_user_for_missing_field(order["id"], field)
                return  # Stop after finding the first missing field

def prompt_user_for_missing_field(order_id, field):
    global is_fixing
    is_fixing = True
    print(f"{bot_name}: For order {order_id}, {field_prompts[field]}")

def update_order(order_id, field, value):
    global is_fixing
    for order in orders:
        if order["id"] == order_id:
            order[field] = value
            print(f"{bot_name}: Updated order {order_id} with {field}: {value}")
            break
    missing_field_context.clear()
    is_fixing = False
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
                    "meats": meats_op if meats_op else "",
                    "rice": rice_op if rice_op else "",
                    "beans": beans_op if beans_op else "",
                    "toppings": toppings_op if toppings_op else ""
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
        # Return the first matching phrase
        match_id, start, end = matches[0]
        span = doc[start:end]
        return span.text.lower()
    else:
        return None

def display_current_order():
    if orders:
        print(f"{bot_name}: Here is your current order:")
        for order in orders:
            print(f"Order ID: {order['id']}, Item: {order['item']}, Meats: {order['meats']}, Rice: {order['rice']}, Beans: {order['beans']}, Toppings: {order['toppings']}")
    else:
        print(f"{bot_name}: Your order is currently empty.")

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    cleaned_sentence = clean_sentence(sentence)

    if DEBUG:
        print(f"[DEBUG] Cleaned sentence: {cleaned_sentence}")

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
        else:
            print(f"{bot_name}: Sorry, I didn't understand. {field_prompts[field]}")
        continue

    # Compute embedding
    input_embedding = sentence_model.encode(cleaned_sentence, convert_to_tensor=True)

    if DEBUG:
        print(f"[DEBUG] Input embedding shape: {input_embedding.shape}")

    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(input_embedding, intent_embeddings)[0]

    if DEBUG:
        print(f"[DEBUG] Similarities: {similarities}")

    # Find the intent with the highest similarity
    max_sim_index = torch.argmax(similarities).item()
    max_sim_score = similarities[max_sim_index].item()
    predicted_tag = intent_tags[max_sim_index]

    if DEBUG:
        print(f"[DEBUG] Predicted tag: {predicted_tag}, Score: {max_sim_score}")

    if max_sim_score > 0.5:  # Threshold can be adjusted
        # Proceed with predicted_tag
        for intent in intents['intents']:
            if predicted_tag == intent["tag"]:
                if predicted_tag == "order":
                    # Process the order using SpaCy for slot filling
                    response = process_order_spacy(cleaned_sentence)
                    print(f"{bot_name}: {response}")
                    if DEBUG:
                        print(f"[DEBUG] Current orders: {orders}")
                    check_missing_fields()

                elif predicted_tag == "remove":
                    # Similar processing for remove intent
                    item = extract_item_spacy(cleaned_sentence)
                    if item:
                        if remove_item_from_order(item):
                            print(f"{bot_name}: I've removed {item} from your order.")
                            if DEBUG:
                                print(f"[DEBUG] Current orders: {orders}")
                        else:
                            print(f"{bot_name}: I'm sorry, I don't think {item} is on the list.")
                    else:
                        print(f"{bot_name}: I'm sorry, I don't think the item you entered is on the menu.")

                elif predicted_tag == "check_order":
                    # Display the current order
                    display_current_order()

                else:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
