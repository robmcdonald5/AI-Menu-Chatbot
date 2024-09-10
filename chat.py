import random 
import json
import torch
import re
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

menu = {
    "8 pc chicken nuggets": 4.99,
    "vegan cheeseburger": 6.49,
    "large fries": 2.99,
    "medium soda": 1.99,
    "pepperoni pizza": 9.99,
}

orders = []

bot_name = "McDonald's"
print("Let's chat! (type 'quit' to exit)")

def clean_sentence(sentence):
    # Lowercase and remove punctuation
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence

def match_order(sentence, menu):
    # Check if any menu item is present in the cleaned sentence
    for item in menu.keys():
        # Use regex to find full item match in the sentence
        pattern = r"\b" + re.escape(item.lower()) + r"\b"
        if re.search(pattern, sentence):
            return item
    return None

def remove_item_from_order(item):
    # Find and remove the first occurrence of the item in the orders list
    for i, order in enumerate(orders):
        if order["item"] == item:
            del orders[i]
            return True
    return False

def extract_quantity(sentence):
    # Find the first number (quantity) in the user's sentence
    match = re.search(r'\b\d+\b', sentence)
    if match:
        return int(match.group())
    return 1  # Default to 1 if no number is found

def process_order_part(order_part, menu):
    """Processes a part of the order (e.g., '3 vegan cheeseburgers') and adds it to orders."""
    quantity = extract_quantity(order_part)  # Extract quantity from the part
    item = match_order(order_part, menu)  # Match item
    if item:
        price = menu[item]
        for _ in range(quantity):
            orders.append({"item": item, "price": price})  # Add item multiple times
        return f"Added {quantity} {item}(s)"
    else:
        return f"Sorry, we don't have that item."

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    cleaned_sentence = clean_sentence(sentence)
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "order":
                    item = match_order(cleaned_sentence, menu)
                    order_parts = re.split(r',\s(?:and)\s|\s(?:and)\s|,', cleaned_sentence)
                    responses = []

                    # Process each part of the order
                    for part in order_parts:
                        response = process_order_part(part, menu)
                        responses.append(response)

                    # Combine responses and show the updated order list
                    print(f"{bot_name}: {' and '.join(responses)}")
                    print(f"Current orders: {orders}")
                elif tag == "remove":
                    item = match_order(cleaned_sentence, menu)
                    if item:
                        if remove_item_from_order(item):
                            print(f"{bot_name}: I've removed {item} from your order.")
                            print(f"Current orders: {orders}")
                        else:
                            print(f"{bot_name}: I'm sorry, I don't think {item} is on the list.")
                    else:
                        print(f"{bot_name}: I'm sorry, I don't think the item you entered is on the menu.")
                else:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")