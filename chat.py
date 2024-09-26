import random 
import json
import torch
import re
from model import NeuralNet
from fuzzywuzzy import process
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

order_id = 1

menu = {
    "bowl": 4.99,
    "burrito": 6.49,
}

meats = ["smoked brisket", "steak", "carnitas", "chicken", "beef barbacoa", "sofritas"]
veg_meat = ["fajita veggies"]

rice = ["white rice", "brown rice", "none"]

beans = ["black beans", "pinto beans"]

toppings = ["guacamole", "fresh tomato salsa", "roasted chili-corn salsa", "tomatillo-green chili salsa", "tomatillo-red chili salsa", "sour cream", "fajita veggies", "cheese", "romaine lettuce", "queso blanco"]

orders = []

bot_name = "McDonald's"
print("Let's chat! (type 'quit' to exit)")

def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence

def match_order(sentence, menu):
    
    sentence = sentence.lower()
    
   
    menu_items = list(menu.keys())
    best_match, score = process.extractOne(sentence, menu_items)
    
    if score >= 80:
        return best_match
    return None

def remove_item_from_order(item):
    
    for i, order in enumerate(orders):
        if order["item"] == item:
            del orders[i]
            return True
    return False

def extract_quantity_part(part):
    match = re.search(r'\b\d+\b', part)
    if match:
        return int(match.group())
    return 1  

def process_order_part(order_part, menu):
    quantity = extract_quantity_part(order_part)
    item = match_order(order_part, menu)
    global order_id
    
    if item:
        price = menu[item]
        for _ in range(quantity):
            orders.append({"id": order_id, "item": item, "price": price})
            order_id += 1
        return f"Added {quantity} {item}(s)"
    else:
        return f"Sorry, we don't have that item."

def process_order(sentence):
    order_parts = re.split(r',\s*|\s+and\s+|\s*,\s*', sentence)
    responses = []

    for part in order_parts:
        response = process_order_part(part, menu)
        responses.append(response)

    return ' and '.join(responses)

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
                    
                    response = process_order(cleaned_sentence)

                    
                    print(f"{bot_name}: {response}")
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