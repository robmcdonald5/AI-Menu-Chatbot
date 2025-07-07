import requests
import json

url = 'http://localhost:5000/chat'
data = {
    'message': 'I want to order a burrito',
    'session_id': '12345'
}

response = requests.post(url, json=data)

print(response.json())