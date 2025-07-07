import requests
import json
import uuid
import time
import pytest

# Configuration
CHATBOT_URL = "http://localhost:5000/chat"
RESET_URL = "http://localhost:5000/reset_session"
HEALTH_URL = "http://localhost:5000/health"
RESET_DELAY = 2
MAX_RETRIES = 5
RETRY_DELAY = 2

# Test cases from tester.py
tests = [
    {
        "name": "Test1",
        "inputs": [
            "Let me get a burrito and a diet coke",
            "Steak please",
            "White rice",
            "Pinto beans",
            "Cheese",
            "That will be all"
        ]
    },
    {
        "name": "Test2",
        "inputs": [
            "I would like a burrito with white rice pinto beans steak and sour cream",
            "That will be all"
        ]
    },
    {
        "name": "Test3",
        "inputs": [
            "What are the vegan options?",
            "What is the menu?"
        ]
    },
    {
        "name": "Test4",
        "inputs": [
            "I would like a burrito"
        ]
    },
    {
        "name": "Test5",
        "inputs": [
            "I would like a bowl"
        ]
    },
    {
        "name": "Test6",
        "inputs": [
            "Let me get a burrito"
        ]
    },
    {
        "name": "Test7",
        "inputs": [
            "Hey, what’s on the menu?",
            "Are there any vegan options?"
        ]
    },
    {
        "name": "Test8",
        "inputs": [
            "Yah let me get a burrito and a diet coke please.",
            "Steak please.",
            "White rice will do.",
            "What is my current order?",
            "Pinto beans sounds good.",
            "Just sour cream and cheese.",
            "That will be all thanks."
        ]
    },
    {
        "name": "Test9",
        "inputs": [
            "I will have a burrito with steak, a bowl with chicken, a diet coke, and a side of chips.",
            "Brown rice.",
            "Pinto beans sounds good.",
            "Guac and cheese.",
            "White rice for order 2.",
            "Black beans also.",
            "Just fajita veggies as a side.",
            "Replace the pinto beans in order 1 with black beans.",
            "Swap the brown rice with white rice.",
            "For order 2.",
            "Remove the chicken in order 2.",
            "None.",
            "Remove order 1.",
            "Remove the pinto beans.",
            "Order 1."
        ]
    },
    {
        "name": "Test10",
        "inputs": [
            "How much does a burrito cost?",
            "What are the health facts for steak?"
        ]
    },
    {
        "name": "Test11",
        "inputs": [
            "I would like a burrito with chicken white rice pinto beans and sour cream.",
            "That will be all"
        ]
    }
]

@pytest.fixture(scope="module", autouse=True)
def wait_for_server_fixture():
    wait_for_server(HEALTH_URL)

def wait_for_server(url, max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(delay)
    pytest.fail("Server is not available after multiple attempts.")

@pytest.mark.parametrize("test_case", tests, ids=[t['name'] for t in tests])
def test_chat_scenarios(test_case):
    session_id = str(uuid.uuid4())
    for user_input in test_case['inputs']:
        payload = {
            "message": user_input,
            "session_id": session_id
        }
        try:
            response = requests.post(CHATBOT_URL, json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert "session_id" in data
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Exception occurred during test: {e}")
    reset_session(session_id)
    time.sleep(RESET_DELAY)

def reset_session(session_id):
    payload = {"session_id": session_id}
    try:
        response = requests.post(RESET_URL, json=payload)
        assert response.status_code == 200
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Failed to reset session: {e}")
