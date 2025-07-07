
import json
import random

with open('intents.json', 'r') as f:
    intents = json.load(f)

def get_response(message):
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if pattern.lower() in message.lower():
                return random.choice(intent['responses'])
    return random.choice([intent['responses'] for intent in intents['intents'] if intent['tag'] == 'fallback'][0])

if __name__ == "__main__":
    print("Chatbot is running! Type 'quit' to exit.")
    while True:
        user_message = input("You: ")
        if user_message.lower() == 'quit':
            break
        response = get_response(user_message)
        print(f"Bot: {response}")
