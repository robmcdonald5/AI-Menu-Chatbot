# AI Menu Chatbot

The **AI Menu Chatbot** is an AI-powered application designed to replace conventional menu interaction at restaurants. Instead of communicating with a human server or manually browsing a menu, customers can interact directly with the chatbot to place orders, ask about menu items, and provide information about menu items. The chatbot leverages deep learning models and natural language processing (NLP) to understand user inputs, process them intelligently (discerning intent), and provide meaningful responses.

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack) 
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Interactive Menu:** Users can browse the menu, ask about ingredients, and receive personalized responses based on their queries.
- **Order Placement:** Customers can place orders directly through the chatbot using text or voice input.
- **Voice Recognition:** The chatbot supports voice input using browser-based APIs, converting speech to text for seamless interaction.
- **Dynamic Responses:** The AI model uses NLP to generate appropriate and consistent responses for a variety of menu-related queries.
- **Multi-Platform Support:** Works seamlessly on desktop and mobile devices with a responsive design.
- **Scalable Deployment:** Deployed using Heroku with containerization support to ensure scalability and ease of maintenance.

## Tech Stack
  ### Frontend
  - **React** + **Tailwind CSS**
  - Browser-based voice recognition APIs (Google Text-to-Speech)
  
  ### Backend
  - **Python**, **BERT Sentence-Transformers**, **SpaCy**, **Flask**, **SKLearn**, **Levenshtein Fuzzer**
  - Deep learning model for intent recognition and menu interaction logic

### Database
- **MongoDB** hosted on **Atlas**

### Deployment
- **Heroku** with Docker containerization

### Contributors
- Joshua: Project Manager
- Nikhil: Architect
- Alex: Dev-Ops
- Kevin: Dev-Ops
- Azmair: QA
- Ishan: QA

### License
- MIT permissive software license
