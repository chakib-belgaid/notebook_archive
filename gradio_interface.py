import gradio as gr
import networkx as nx
import matplotlib.pyplot as plt
from image_to_knowledge_graph import process_images, create_knowledge_graph, index_text_data, create_vector_knowledge_graph

def transform_images_to_knowledge_graph(image_files):
    image_folder = "uploaded_images"
    for image_file in image_files:
        image_file.save(f"{image_folder}/{image_file.name}")
    
    text_data = process_images(image_folder)
    knowledge_graph = create_knowledge_graph(text_data)
    tfidf_matrix = index_text_data(text_data)
    vector_knowledge_graph = create_vector_knowledge_graph(tfidf_matrix)
    
    return knowledge_graph, vector_knowledge_graph

def draw_graph(graph):
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
    plt.show()

def display_knowledge_graph(image_files):
    knowledge_graph, vector_knowledge_graph = transform_images_to_knowledge_graph(image_files)
    draw_graph(knowledge_graph)
    draw_graph(vector_knowledge_graph)
    return "Knowledge graph generated and displayed."

iface = gr.Interface(
    fn=display_knowledge_graph,
    inputs=gr.inputs.File(file_count="multiple", label="Upload Images"),
    outputs="text",
    title="Image to Knowledge Graph",
    description="Upload multiple images to transform them into a knowledge graph."
)

if __name__ == "__main__":
    iface.launch()
