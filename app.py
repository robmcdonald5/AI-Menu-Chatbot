from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Sample menu and orders
menu = {
    "bowl": 4.99,
    "burrito": 6.49,
}

orders = []

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = process_order(user_input)
    return jsonify({"response": response})

def process_order(input_sentence):
    global orders
    items = []
    for item in menu:
        if item in input_sentence.lower():
            items.append(item)
            orders.append({"item": item, "price": menu[item]})
    if items:
        return f"Added {', '.join(items)} to your order."
    else:
        return "Sorry, I didn't understand the items you want to order."

if __name__ == '__main__':
    app.run(debug=True)