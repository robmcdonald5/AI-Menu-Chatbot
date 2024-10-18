import unittest
import subprocess

class TestChatbot(unittest.TestCase):
    def run_chatbot(self, inputs):
        """
        Helper method to run chat.py with the given inputs.
        Returns the stdout and stderr outputs.
        """
        input_str = '\n'.join(inputs) + '\n'  # Ensure the last input is processed
        try:
            process = subprocess.Popen(
                ['python', 'chat.py'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True  # For Python 3.7+, use 'universal_newlines=True' for earlier versions
            )
            stdout, stderr = process.communicate(input=input_str, timeout=10)  # Timeout to prevent hanging
            return stdout, stderr
        except subprocess.TimeoutExpired:
            process.kill()
            return "", "Process timed out."

    def test_dialogue_1_simple_burrito_order(self):
        """
        Test Dialogue 1: Simple burrito order
        """
        inputs = [
            "Hi, I'd like a burrito.",
            "I'll go with chicken.",
            "White rice, please.",
            "Black beans.",
            "I'll have salsa, cheese, and a little bit of guacamole.",
            "No, that should be it.",
            "quit"
        ]
        stdout, stderr = self.run_chatbot(inputs)
        
        # Here you can add assertions based on expected outputs
        # For example:
        # self.assertIn("Your burrito order:", stdout)
        print("Test Dialogue 1 Output:")
        print(stdout)
        if stderr:
            print("Error Output:")
            print(stderr)

    def test_dialogue_2_simple_bowl_order(self):
        """
        Test Dialogue 2: Simple bowl order
        """
        inputs = [
            "I’d like to get a burrito, please.",
            "Steak.",
            "None.",
            "Pinto beans, please.",
            "Just sour cream.",
            "Yes, that’s all good.",
            "quit"
        ]
        stdout, stderr = self.run_chatbot(inputs)
        
        print("Test Dialogue 2 Output:")
        print(stdout)
        if stderr:
            print("Error Output:")
            print(stderr)

    def test_dialogue_3_vegetarian_option_default(self):
        """
        Test Dialogue 3: Vegetarian option default
        """
        inputs = [
            "Can I get a bowl?",
            "Fajita veggies please.",
            "Brown rice.",
            "I’ll have black beans.",
            "romaine lettuce and chili salsa.",
            "Yes, that's perfect.",
            "quit"
        ]
        stdout, stderr = self.run_chatbot(inputs)
        
        print("Test Dialogue 3 Output:")
        print(stdout)
        if stderr:
            print("Error Output:")
            print(stderr)

    def test_dialogue_4_multiple_items(self):
        """
        Test Dialogue 4: Multiple items
        """
        inputs = [
            "Hi, let me get a burrito and a bowl.",
            "I’ll have a chicken for the burrito.",
            "White rice please.",
            "I’ll have black beans.",
            "None for sides.",
            "Steak for the bowl thanks.",
            "None rice.",
            "Black beans please.",
            "Umm.. I’ll have guacamole and cheese on top of that",
            "Yes, that's everything.",
            "quit"
        ]
        stdout, stderr = self.run_chatbot(inputs)
        
        print("Test Dialogue 4 Output:")
        print(stdout)
        if stderr:
            print("Error Output:")
            print(stderr)

    def test_dialogue_5_vegetarian_order_with_question(self):
        """
        Test Dialogue 5: Vegetarian order with question
        """
        inputs = [
            "What vegetarian options do you have?",
            "Okay, give me a bowl.",
            "Fajita veggies.",
            "None for rice.",
            "Black beans please.",
            "I’ll have cheese and lettuce.",
            "No, that’s all.",
            "quit"
        ]
        stdout, stderr = self.run_chatbot(inputs)
        
        print("Test Dialogue 5 Output:")
        print(stdout)
        if stderr:
            print("Error Output:")
            print(stderr)

    def test_dialogue_6_multi_order_with_general_removal(self):
        """
        Test Dialogue 6: Multi order with a general removal
        """
        inputs = [
            "Hello, can I get a burrito and a bowl with steak.",
            "White rice for the first.",
            "None for beans.",
            "Cheese and sour cream please.",
            "Brown rice.",
            "Pinto beans please.",
            "None for toppings on the bowl.",
            "Yah, I actually want to change my order.",
            "Remove order 2.",
            "No, that’s all.",
            "quit"
        ]
        stdout, stderr = self.run_chatbot(inputs)
        
        print("Test Dialogue 6 Output:")
        print(stdout)
        if stderr:
            print("Error Output:")
            print(stderr)

if __name__ == '__main__':
    unittest.main()