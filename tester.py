import requests
import json
import uuid
import time

# Configuration
CHATBOT_URL = "http://localhost:5000/chat"         # URL where chat.py is running
RESET_URL = "http://localhost:5000/reset_session"  # URL for session reset
HEALTH_URL = "http://localhost:5000/health"        # Health check endpoint
RESET_DELAY = 2                                    # Seconds to wait before starting the next test
MAX_RETRIES = 5                                    # Maximum number of retries to connect to the server
RETRY_DELAY = 2                                    # Seconds to wait between retries

# Define your test cases here
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

def reset_session(session_id):
    """
    Sends a POST request to reset the session for the given session ID.
    
    Args:
        session_id (str): The session ID to reset.
    """
    payload = {"session_id": session_id}
    try:
        response = requests.post(RESET_URL, json=payload)
        if response.status_code == 200:
            print(f"[INFO] Session {session_id} reset successfully.\n")
        else:
            print(f"[WARNING] Failed to reset session {session_id}. Status: {response.status_code}")
            print(f"Response: {response.text}\n")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Exception occurred while resetting session {session_id}: {e}\n")

def run_test(test):
    """
    Runs a single test case by sending each input to the chatbot and printing the responses.
    
    Args:
        test (dict): A dictionary containing the test name and list of inputs.
    """
    test_name = test.get("name", "Unnamed Test")
    inputs = test.get("inputs", [])
    session_id = str(uuid.uuid4())  # Generate a unique session_id for the test

    print(f"=== Running {test_name} ===")
    print(f"Session ID: {session_id}\n")

    for idx, user_input in enumerate(inputs, 1):
        print(f"Input {idx}: {user_input}")
        
        payload = {
            "message": user_input,
            "session_id": session_id
        }
        
        try:
            response = requests.post(CHATBOT_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                chatbot_response = data.get("response", "")
                returned_session_id = data.get("session_id", session_id)
                print(f"Output {idx}: {chatbot_response}\n")
                
                # Optional: Verify that the session_id remains consistent
                if returned_session_id != session_id:
                    print(f"[WARNING] Session ID mismatch. Expected {session_id}, got {returned_session_id}\n")
                    session_id = returned_session_id  # Update session_id if changed by chatbot
            else:
                print(f"Output {idx}: Error {response.status_code} - {response.text}\n")
        
        except requests.exceptions.RequestException as e:
            print(f"Output {idx}: Exception occurred - {e}\n")
        
        # Optional: Brief pause between requests to mimic real user interaction
        time.sleep(0.5)
    
    print(f"=== {test_name} Completed ===\n{'-'*50}\n")
    # Reset the session after the test
    reset_session(session_id)
    # Optional: Wait before starting the next test
    time.sleep(RESET_DELAY)

def wait_for_server(url, max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    """
    Waits for the server to be available by repeatedly sending a GET request to the health endpoint.
    
    Args:
        url (str): The health check URL.
        max_retries (int): Maximum number of retries.
        delay (int): Seconds to wait between retries.
    
    Returns:
        bool: True if the server is available, False otherwise.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("[INFO] Server is up and running.\n")
                return True
        except requests.exceptions.RequestException:
            pass
        print(f"[WARNING] Server not available. Retrying in {delay} seconds... (Attempt {attempt}/{max_retries})")
        time.sleep(delay)
    print("[ERROR] Server is not available after multiple attempts.")
    return False

def main():
    """
    Main function to run all defined test cases.
    """
    if not tests:
        print("No tests defined. Please add test cases to the 'tests' list.")
        return

    print("Starting automated tests for chat.py chatbot...\n")

    # Wait for the chat.py server to be ready
    if not wait_for_server(HEALTH_URL):
        return

    for test in tests:
        run_test(test)

    print("All tests completed.")

if __name__ == "__main__":
    main()