import nltk
import re
from nltk.tokenize import word_tokenize


# Sample menu
menu_items = ["vegan cheeseburgers", "medium sodas"]



num_words = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "2x":2
}

# Preprocessing: Tokenize and handle number words
def preprocess_input(input_sentence):
    tokens = word_tokenize(input_sentence.lower())
    # Replace word numbers with digits
    tokens = [str(num_words[word]) if word in num_words else word for word in tokens]
    return tokens

# Extract quantities and items
def extract_items(input_sentence):
    tokens = preprocess_input(input_sentence)
    
    # Join tokens back for easier regex matching
    sentence = ' '.join(tokens)
    
    # Regular expression to capture items and quantities
    item_pattern = re.compile(r'(\d*)\s*(\b(?:' + '|'.join(menu_items) + r')\b)')

    print(item_pattern)
    
    # Find all matches for quantities and items
    matches = item_pattern.findall(sentence)
    
    order = {}
    
    # A variable to keep track of the last matched item
    last_item = None
    
    for match in matches:
        print(match)
        quantity = int(match[0]) if match[0] else 1  # Default to 1 if no quantity is specified
        item = match[1]
        
        if item in order:
            order[item] += quantity  # Sum quantities if item is repeated
        else:
            order[item] = quantity
        
        last_item = item
    
    # Handle "X of them" pattern
    of_them_pattern = re.compile(r'(\d+)\s*of them')
    of_them_matches = of_them_pattern.findall(sentence)
    
    if of_them_matches and last_item:
        # Add the quantity from "X of them" to the last matched item
        quantity = int(of_them_matches[0])
        order[last_item] = quantity
    
    return order

def extract_items2(input_sentence):
    tokens = preprocess_input(input_sentence) 
    sentence = ' '.join(tokens)
    
    
    item_pattern = re.compile(r'(\d+\s+)?(' + '|'.join(map(re.escape, menu_items)) + r')', re.IGNORECASE)

    
    # Insert a special divider marker wherever there is a match for an item
    divided_sentence = item_pattern.sub(r'##\g<0>', sentence)
    
    # Split the sentence using the divider to separate the elements
    fragments = [frag.strip() for frag in divided_sentence.split('##') if frag.strip()]
    
    return fragments

# Test cases
test_cases = [
    "Can I order 2 vegan cheeseburgers, and 2 medium sodas",
    "Can I order vegan cheeseburgers, 2 of them, and 3 medium sodas",
    "Can I get 2x vegan cheeseburgers and with extra cheese, and a medium sodas",
    "Can I get two vegan cheeseburgers two medium sodas"
]

# Running test cases
for test in test_cases:
    print(f"Input: {test}")
    print("Extracted Order:", extract_items2(test))
    print("-" * 40)