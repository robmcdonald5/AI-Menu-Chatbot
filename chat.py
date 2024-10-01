import random 
import json
import torch
import re
from model import NeuralNet
from fuzzywuzzy import process
from nltk_utils import bag_of_words, tokenize
from nltk.tokenize import word_tokenize

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

is_fixing = False

menu = {
    "bowl": 4.99,
    "burrito": 6.49,
}

num_words = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "2x":2
}

meats = ["smoked brisket", "steak", "carnitas", "chicken", "beef barbacoa", "sofritas", "fajita veggies"]


rice = ["white rice", "brown rice", "none"]

beans = ["black beans", "pinto beans", "none"]

toppings = ["guacamole", "fresh tomato salsa", "roasted chili-corn salsa", "tomatillo-green chili salsa", "tomatillo-red chili salsa", "sour cream", "fajita veggies", "cheese", "romaine lettuce", "queso blanco", "none"]

addons_list = meats +  rice + beans + toppings

orders = []

missing_field_context = {} 

bot_name = "Chipotle"
print("Let's chat! (type 'quit' to exit)")

def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence

def match_order(sentence, menu):
    
    sentence = sentence.lower()
    
   
    menu_items = list(menu.keys())
    best_match, score = process.extractOne(sentence, menu_items)
    
    if score >= 60:
        return best_match
    return None

def remove_item_from_order(item):
    
    for i, order in enumerate(orders):
        if order["item"] == item:
            del orders[i]
            return True
    return False

def tokenize_quant(input_sentence):
    tokens = word_tokenize(input_sentence.lower())
    # Replace word numbers with digits
    tokens = [str(num_words[word]) if word in num_words else word for word in tokens]
    return tokens

def extract_quantity_part(part):
    match = re.search(r'\b\d+\b', part)
    if match:
        return int(match.group())
    return 1  

def extract_addons(sentence, addons_list):
    addons_found = []
    for addon in addons_list:
        if addon in sentence:
            addons_found.append(addon)
    return addons_found

def process_order_part(order_part, menu):
    quantity = extract_quantity_part(order_part)
    item = match_order(order_part, menu)
    global order_id
    
    if item:
        price = menu[item]
        meat_op = extract_addons(order_part, meats)
        rice_op = extract_addons(order_part, rice)
        beans_op = extract_addons(order_part, beans)
        toppings_op = extract_addons(order_part, toppings)
        for _ in range(quantity):
            orders.append({
                "id": order_id, 
                "item": item, 
                "price": price, 
                "meats": meat_op if meat_op else "",
                "rice": rice_op if rice_op else "",
                "beans": beans_op if beans_op else "",
                "toppings": toppings_op if toppings_op else ""
            })

            order_id += 1
        return f"Added {quantity} {item}(s)"
    else:
        return f"Sorry, we don't have that item."



def process_order(input_sentence):
    tokens = tokenize_quant(input_sentence) 
    sentence = ' '.join(tokens)
    responses = []
    
    
    item_pattern = re.compile(r'(\d+\s+)?(' + '|'.join(map(re.escape, menu)) + r')', re.IGNORECASE)

    
    divided_sentence = item_pattern.sub(r'##\g<0>', sentence)
    
    fragments = [frag.strip() for frag in divided_sentence.split('##') if frag.strip()]
    
    print(fragments)
    for frag in fragments:
        response = process_order_part(frag, menu)
        responses.append(response)
    
    

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
    field_prompts = {
        "meats": "What kind of meat would you like?",
        "rice": "What type of rice would you like?",
        "beans": "Would you like black beans, pinto beans, or none?",
        "toppings": "What toppings would you like?"
    }
    is_fixing = True
    print(f"{bot_name}: For order {order_id}, {field_prompts[field]}")
    print(missing_field_context)

def update_order(order_id, field, value):
    for order in orders:
        if order["id"] == order_id:
            order[field] = value
            print(f"{bot_name}: Updated order {order_id} with {field}: {value}")
            break
    missing_field_context.clear()

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
                if tag == "order" and not is_fixing:
                    
                    response = process_order(cleaned_sentence)
                    
                    print(f"{bot_name}: {response}")
                    print(f"Current orders: {orders}")

                    check_missing_fields()

                elif tag == "remove" and not is_fixing:
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