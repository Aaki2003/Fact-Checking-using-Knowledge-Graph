from transformers import pipeline
import pandas as pd
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the triplet extractor pipeline
triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')

# Function to parse the generated text and extract triplets
def extract_triplets(text):
    triplets = []
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'relation': relation.strip(), 'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'relation': relation.strip(), 'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'relation': relation.strip(), 'tail': object_.strip()})
    return triplets

# Load the claim dataset
file_path = "/kaggle/input/book001/Book1.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Ensure the dataset has a column named 'claim'
claim_column_name = "Claim"  # Update this if your column name is different
if claim_column_name not in df.columns:
    raise ValueError(f"The dataset must contain a '{claim_column_name}' column.")

# Extract triplets for each claim
triplets_data = []
for claim in df[claim_column_name].head(50):
    # Generate triplet extraction output
    output = triplet_extractor(claim, return_tensors=True, return_text=False)
    extracted_text = triplet_extractor.tokenizer.batch_decode([output[0]["generated_token_ids"]])[0]
    # Parse and extract triplets
    triplets = extract_triplets(extracted_text)
    triplets_data.append({"claim": claim, "triplets": triplets})

# Create a new DataFrame with extracted triplets
triplets_df = pd.DataFrame(triplets_data)

# Save the extracted triplets to a new CSV file
output_file = "/kaggle/working/extracted_triplets.csv"
triplets_df.to_csv(output_file, index=False)

print(f"Triplets extracted and saved to {output_file}")




# Load the extracted triplets file
input_file = "/kaggle/working/extracted_triplets.csv"  # Replace with the path to your file
df = pd.read_csv(input_file)

# Initialize a directed graph
kg = nx.DiGraph()

# Add nodes and edges to the graph
for _, row in df.iterrows():
    claim = row['claim']
    try:
        triplets = eval(row['triplets'])  # Convert stringified list back to a list of dictionaries
        for triplet in triplets:
            head = triplet['head']
            tail = triplet['tail']
            relation = triplet['relation']
            # Add nodes and edges
            kg.add_node(head, type='entity')
            kg.add_node(tail, type='entity')
            kg.add_edge(head, tail, relation=relation, claim=claim)
    except Exception as e:
        print(f"Error processing row: {row}\nError: {e}")

# Function to extract edge labels for visualization
def get_edge_labels(graph):
    return {(u, v): d['relation'] for u, v, d in graph.edges(data=True)}

# Draw the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(kg, k=0.5, seed=42)  # Layout for visualization

# Draw nodes and edges
nx.draw(kg, pos, with_labels=True, node_color="lightblue", node_size=3000, font_size=10, font_weight='bold', edge_color="gray", arrowsize=10)
edge_labels = get_edge_labels(kg)
nx.draw_networkx_edge_labels(kg, pos, edge_labels=edge_labels, font_color="red")

# Show the graph
plt.title("Knowledge Graph from Extracted Triplets")
plt.show()

# Optionally save the graph
output_graph_file = "knowledge_graph.gml"
nx.write_gml(kg, output_graph_file)
print(f"Knowledge graph saved to {output_graph_file}")

user_claim = input("Enter a claim: ")

print(user_claim)
triplet_extractor_inp = pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')
# We need to use the tokenizer manually since we need special tokens.
extracted_text_inp = triplet_extractor_inp.tokenizer.batch_decode([triplet_extractor_inp(user_claim, return_tensors=True, return_text=False)[0]["generated_token_ids"]])
print(extracted_text_inp[0])
extracted_triplets_inp = extract_triplets(extracted_text_inp[0])
print(extracted_triplets_inp)

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load a pre-trained sentence transformer model for similarity calculation
model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(triplet, edge_data):
    """
    Compute the similarity between the input triplet and an edge from the knowledge graph.
    """
    input_text = f"{triplet['head']} {triplet['relation']} {triplet['tail']}"
    edge_text = f"{edge_data['head']} {edge_data['relation']} {edge_data['tail']}"
    embeddings = model.encode([input_text, edge_text])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def retrieve_top_k_contexts(kg, triplet_inp, k=3):
    """
    Retrieve the top-k relevant contexts for the input triplet from the knowledge graph.
    """
    contexts = []
    for u, v, data in kg.edges(data=True):
        # Retrieve edge data
        edge_triplet = {
            'head': u,
            'relation': data['relation'],
            'tail': v
        }
        # Compute similarity
        similarity = compute_similarity(triplet_inp, edge_triplet)
        contexts.append((edge_triplet, similarity, data.get('claim', 'Unknown claim')))
    
    # Sort contexts by similarity and retrieve the top-k
    top_k = sorted(contexts, key=lambda x: x[1], reverse=True)[:k]
    return top_k

# Example usage
# Assume triplet_inp is the triplet extracted from user input
triplet_inp = extracted_triplets_inp[0]

# Retrieve top-k relevant contexts from the Knowledge Graph
top_k_contexts = retrieve_top_k_contexts(kg, triplet_inp, k=2)

required_contexts = []

# Print the results
print("Top-k Relevant Contexts:")
for idx, (context, similarity, claim) in enumerate(top_k_contexts, 1):
    required_contexts.append(context)
    print(f"{idx}. Context: {context}, Similarity: {similarity:.4f}, Source Claim: {claim}")


print(required_contexts)


# Set up the device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "meta-llama/Llama-2-7b-chat-hf"
access_token = "your acc_token"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token)
llama = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=access_token,
    torch_dtype=torch.float16
).to(device)

# Function to format the input for triplet-based claim verification
def format_input_from_triplets(claim_triplet, context_triplets):
    # Serialize claim and context into text format
    claim_text = f"Claim: Head: {claim_triplet['head']}, Relation: {claim_triplet['relation']}, Tail: {claim_triplet['tail']}.\n"
    context_text = "Context Triplets:\n" + "\n".join(
        [f"- Head: {t['head']}, Relation: {t['relation']}, Tail: {t['tail']}." for t in context_triplets]
    )
    prompt = (
        f"The task is to verify the claim based on the provided context triplets. "
        f"Label the claim as 'True', 'False', or 'Not Enough Information'.\n\n"
        f"{claim_text}\n{context_text}\nAnswer:"
    )
    return prompt

# Function to predict claim verification
def predict_triplet_claim_verification(claim_triplet, context_triplets):
    # Prepare the input
    input_text = format_input_from_triplets(claim_triplet, context_triplets)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate prediction
    output = llama.generate(**inputs, max_length=256, temperature=0.7, top_p=0.9)
    prediction = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the response after "Answer:"
    answer_start = prediction.find("Answer:") + len("Answer:")
    return prediction[answer_start:].strip()

# Example triplets
claim_triplet = extracted_triplets_inp[0]

context_triplets = required_contexts
# Predict
result = predict_triplet_claim_verification(claim_triplet, context_triplets)

print(f"Claim Triplet: {claim_triplet}")
print(f"Context Triplets: {context_triplets}")
print(f"Prediction: {result}")

