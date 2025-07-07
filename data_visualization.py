
import re
import matplotlib
matplotlib.use('Agg')
import json
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

def plot_intent_distribution(intents_file='intents.json'):
    """
    Analyzes the distribution of patterns per intent from a JSON file and creates a bar plot.

    Args:
        intents_file (str): The path to the intents JSON file.
    """
    with open(intents_file, 'r') as f:
        data = json.load(f)

    intent_counts = {intent['tag']: len(intent['patterns']) for intent in data['intents']}
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(intent_counts.keys()), y=list(intent_counts.values()), palette='viridis')
    plt.title('Number of Patterns per Intent', fontsize=16)
    plt.xlabel('Intent', fontsize=12)
    plt.ylabel('Number of Patterns', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('intent_distribution.png')
    plt.show()

def plot_pattern_length_distribution(intents_file='intents.json'):
    """
    Analyzes and plots the distribution of pattern lengths (number of words).

    Args:
        intents_file (str): The path to the intents JSON file.
    """
    with open(intents_file, 'r') as f:
        data = json.load(f)

    pattern_lengths = [len(pattern.split()) for intent in data['intents'] for pattern in intent['patterns']]

    plt.figure(figsize=(12, 8))
    sns.histplot(pattern_lengths, bins=np.arange(min(pattern_lengths), max(pattern_lengths) + 2) - 0.5, kde=True, palette='mako')
    plt.title('Distribution of Pattern Lengths', fontsize=16)
    plt.xlabel('Pattern Length (Number of Words)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(range(min(pattern_lengths), max(pattern_lengths) + 1))
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('pattern_length_distribution.png')
    plt.show()

def generate_word_cloud(intents_file='intents.json'):
    """
    Generates a word cloud from all patterns in the intents file.

    Args:
        intents_file (str): The path to the intents JSON file.
    """
    with open(intents_file, 'r') as f:
        data = json.load(f)

    all_patterns = ' '.join([pattern for intent in data['intents'] for pattern in intent['patterns']])

    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='inferno').generate(all_patterns)

    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of All Patterns', fontsize=20)
    plt.tight_layout()
    plt.savefig('patterns_word_cloud.png')
    plt.show()

import os

def generate_dependency_graph(directory='.'):
    """
    Generates a dependency graph of Python modules in the specified directory.

    Args:
        directory (str): The directory to analyze.
    """
    try:
        from graphviz import Digraph
    except ImportError:
        print("Graphviz is not installed. Please install it using: pip install graphviz")
        return

    dot = Digraph(comment='Module Dependency Graph', format='png')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    dot.attr('edge', color='gray')
    dot.attr(rankdir='LR')

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                module_path = os.path.join(root, file)
                module_name = os.path.splitext(os.path.basename(module_path))[0]
                dot.node(module_name)

                try:
                    with open(module_path, 'r', encoding='latin-1') as f:
                        for line in f:
                            if line.startswith('import ') or line.startswith('from '):
                                parts = line.split()
                                if len(parts) > 1:
                                    dependency = parts[1].split('.')[0]
                                    if os.path.exists(os.path.join(directory, dependency + '.py')):
                                        dot.edge(module_name, dependency)
                except Exception as e:
                    print(f"Could not read file {module_path}: {e}")

    dot.render('module_dependency_graph', view=False)
    print("Generated module_dependency_graph.png")

import re

def create_test_intent_matrix(intents_file='intents.json', tester_file='tester.py'):
    """
    Creates a heatmap showing which test cases likely cover which intents.

    Args:
        intents_file (str): Path to the intents JSON file.
        tester_file (str): Path to the tester Python file.
    """
    with open(intents_file, 'r') as f:
        intents_data = json.load(f)
    
    intents = {intent['tag']: intent['patterns'] for intent in intents_data['intents']}
    intent_names = list(intents.keys())

    # A more robust way to extract the tests from the tester.py file content
    with open(tester_file, 'r', encoding='utf-8') as f:
        tester_content = f.read()
    
    # Isolate the list definition
    try:
        from tester import tests as tests_list
        test_names = [test['name'] for test in tests_list]
        test_inputs = [[i.lower() for i in test['inputs']] for test in tests_list]
    except ImportError:
        print("Could not import 'tests' from tester.py")
        return
    except Exception as e:
        print(f"Could not process the 'tests' list from tester.py: {e}")
        return

    # Create a matrix to hold the connections
    matrix = np.zeros((len(intent_names), len(test_names)))

    # Heuristic: check if any word from an intent pattern appears in a test's inputs
    for i, intent_tag in enumerate(intent_names):
        # Flatten all patterns for the intent into a set of unique words
        intent_words = set()
        for p in intents[intent_tag]:
            intent_words.update(p.lower().replace('?', '').replace('.', '').split())
            
        for j, inputs in enumerate(test_inputs):
            test_words = set(' '.join(inputs).split())
            # If there's any overlap in words, we assume a connection
            if not intent_words.isdisjoint(test_words):
                matrix[i, j] = 1

    plt.figure(figsize=(15, 10))
    sns.heatmap(matrix, xticklabels=test_names, yticklabels=intent_names, annot=True, cmap="YlGnBu", cbar=False, linewidths=.5)
    plt.title('Test Case vs. Intent Matrix', fontsize=20)
    plt.xlabel('Test Cases', fontsize=15)
    plt.ylabel('Intents', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('test_intent_matrix.png')
    plt.show()
    print("Generated test_intent_matrix.png")

def plot_intent_similarity_heatmap(intents_file='intents.json', model_name='all-MiniLM-L6-v2'):
    """
    Calculates and plots the cosine similarity between the average embeddings of each intent.

    Args:
        intents_file (str): Path to the intents JSON file.
        model_name (str): The name of the SentenceTransformer model to use.
    """
    print("Loading sentence transformer model...")
    model = SentenceTransformer(model_name)
    
    with open(intents_file, 'r') as f:
        data = json.load(f)

    intent_embeddings = {}
    intent_names = [intent['tag'] for intent in data['intents']]

    print("Generating intent embeddings...")
    for intent in data['intents']:
        patterns = intent['patterns']
        if patterns:
            # Average the embeddings of all patterns for the intent
            embeddings = model.encode(patterns)
            intent_embeddings[intent['tag']] = np.mean(embeddings, axis=0)
        else:
            # Handle intents with no patterns
            intent_embeddings[intent['tag']] = np.zeros(model.get_sentence_embedding_dimension())

    # Create a matrix of the average embeddings
    embedding_matrix = np.array([intent_embeddings[name] for name in intent_names])

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(embedding_matrix)

    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, xticklabels=intent_names, yticklabels=intent_names, annot=True, cmap='Reds', fmt='.2f')
    plt.title('Intent Similarity Heatmap', fontsize=20)
    plt.xlabel('Intents', fontsize=15)
    plt.ylabel('Intents', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('intent_similarity_heatmap.png')
    plt.show()
    print("Generated intent_similarity_heatmap.png")

def plot_tsne_pattern_embeddings(intents_file='intents.json', model_name='all-MiniLM-L6-v2'):
    """
    Creates a t-SNE plot of all pattern embeddings, colored by intent.

    Args:
        intents_file (str): Path to the intents JSON file.
        model_name (str): The name of the SentenceTransformer model to use.
    """
    print("Loading sentence transformer model...")
    model = SentenceTransformer(model_name)
    
    with open(intents_file, 'r') as f:
        data = json.load(f)

    all_patterns = []
    all_labels = []
    intent_names = [intent['tag'] for intent in data['intents']]

    print("Generating pattern embeddings for t-SNE...")
    for intent in data['intents']:
        for pattern in intent['patterns']:
            all_patterns.append(pattern)
            all_labels.append(intent['tag'])

    embeddings = model.encode(all_patterns)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings)-1), random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(16, 12))
    palette = sns.color_palette("hsv", len(intent_names))
    sns.scatterplot(x=embeddings_2d[:,0], y=embeddings_2d[:,1], hue=all_labels, palette=palette, s=100, alpha=0.7, legend='full')
    plt.title('t-SNE Visualization of Pattern Embeddings', fontsize=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=15)
    plt.ylabel('t-SNE Dimension 2', fontsize=15)
    plt.legend(title='Intents', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('tsne_pattern_embeddings.png')
    plt.show()
    print("Generated tsne_pattern_embeddings.png")

if __name__ == '__main__':
    plot_intent_distribution()
    plot_pattern_length_distribution()
    generate_word_cloud()
    # The dependency graph requires external software (Graphviz) to be installed by the user.
    # generate_dependency_graph() 
    create_test_intent_matrix()
    plot_intent_similarity_heatmap()
    plot_tsne_pattern_embeddings()
