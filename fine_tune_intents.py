import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os

# Configuration
INTENTS_FILE = 'intents.json'
BASE_MODEL_NAME = 'all-MiniLM-L6-v2'
FINE_TUNED_MODEL_PATH = 'fine_tuned_intent_model'

# Training parameters
NUM_EPOCHS = 4
TRAIN_BATCH_SIZE = 16
LEARNING_RATE = 2e-5

def fine_tune_model():
    print(f"Loading intents from {INTENTS_FILE}...")
    with open(INTENTS_FILE, 'r') as f:
        intents_data = json.load(f)

    train_examples = []
    for intent in intents_data['intents']:
        tag = intent['tag']
        patterns = intent['patterns']
        for pattern in patterns:
            # For classification, we can use a dummy label (e.g., 0) and rely on the intent tag
            # The SoftmaxLoss will learn to separate embeddings based on their class (intent tag)
            train_examples.append(InputExample(texts=[pattern], label=tag))

    # Map labels to integers for SoftmaxLoss
    label_map = {label: i for i, label in
                 enumerate(sorted(list(set([example.label
                                            for example in
                                            train_examples]))))}
    for example in train_examples:
        example.label = label_map[example.label]

    print(f"Loaded {len(train_examples)} training examples.")

    print(f"Loading base model: {BASE_MODEL_NAME}...")
    model = SentenceTransformer(BASE_MODEL_NAME)

    # Define the DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True,
                                  batch_size=TRAIN_BATCH_SIZE)

    # Define the loss function. SoftmaxLoss is suitable for classification tasks.
    train_loss = losses.SoftmaxLoss(model=model,
                                    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                    num_labels=len(label_map))

    print("Starting fine-tuning...")
    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=NUM_EPOCHS,
              warmup_steps=100,
              output_path=FINE_TUNED_MODEL_PATH,
              show_progress_bar=True,
              optimizer_params={'lr': LEARNING_RATE})

    print(f"Fine-tuning complete. Model saved to {FINE_TUNED_MODEL_PATH}")

if __name__ == '__main__':
    fine_tune_model()