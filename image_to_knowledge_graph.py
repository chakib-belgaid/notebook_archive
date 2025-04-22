import os
from google.cloud import vision
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gemeni

# Function to process images and extract text using Google GenAI for OCR
def process_images(image_folder):
    client = vision.ImageAnnotatorClient()
    text_data = []
    for image_file in os.listdir(image_folder):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_file)
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = client.text_detection(image=image)
            texts = response.text_annotations
            if texts:
                text_data.append(texts[0].description)
    return text_data

# Function to create a knowledge graph from text data
def create_knowledge_graph(text_data):
    nlp = spacy.load("en_core_web_sm")
    graph = nx.DiGraph()

    for text in text_data:
        doc = nlp(text)
        for ent in doc.ents:
            graph.add_node(ent.text, label=ent.label_)
            for token in doc:
                if token.dep_ in ("nsubj", "dobj"):
                    graph.add_edge(ent.text, token.text)

    return graph

# Function to index and analyze text data
def index_text_data(text_data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return tfidf_matrix

# Function to create a knowledge graph for the vector database
def create_vector_knowledge_graph(tfidf_matrix):
    graph = nx.DiGraph()
    similarity_matrix = cosine_similarity(tfidf_matrix)

    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix[i])):
            if i != j and similarity_matrix[i][j] > 0.5:
                graph.add_edge(i, j, weight=similarity_matrix[i][j])

    return graph

# Function to store text data in markdown format
def store_text_data_as_markdown(text_data, output_file):
    with open(output_file, 'w') as f:
        for text in text_data:
            f.write(f"## Text Chunk\n{text}\n\n")

# Main function to handle the transformation of images into a knowledge graph
def main(image_folder):
    text_data = process_images(image_folder)
    knowledge_graph = create_knowledge_graph(text_data)
    tfidf_matrix = index_text_data(text_data)
    vector_knowledge_graph = create_vector_knowledge_graph(tfidf_matrix)

    # Save and visualize the knowledge graph
    nx.write_gml(knowledge_graph, "knowledge_graph.gml")
    nx.draw(knowledge_graph, with_labels=True)
    plt.show()

    # Save and visualize the vector knowledge graph
    nx.write_gml(vector_knowledge_graph, "vector_knowledge_graph.gml")
    nx.draw(vector_knowledge_graph, with_labels=True)
    plt.show()

    # Store text data in markdown format
    store_text_data_as_markdown(text_data, "text_data.md")

if __name__ == "__main__":
    image_folder = "path_to_image_folder"
    main(image_folder)
