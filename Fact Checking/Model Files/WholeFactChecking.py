from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from langchain.chains import TransformChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
import torch
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from tqdm import tqdm
from collections import OrderedDict

# Define entity extraction logic

# Initialize the Named Entity Recognition pipeline
ner_model = pipeline("ner", grouped_entities=True)

def get_entities(text):
    """
    Extract entities using both NER and specific format markers like <subj> and <obj>.
    """
    # Extract entities using the NER model
    ner_results = ner_model(text)
    ner_entities = {entity['word'] for entity in ner_results}  # General entities from NER

    # Handle specific format markers
    tokens = text.split()
    format_entities = set()
    subject, object_, relation = '', '', ''
    current = None  # Keeps track of whether we're inside <subj>, <obj>, or <relation>
    
    for token in tokens:
        if token == "<subj>":
            current = "subj"
            if object_ and relation:  # Store the previous triplet if any
                format_entities.add(subject.strip())
                format_entities.add(object_.strip())
            subject = ''
        elif token == "<obj>":
            current = "obj"
            relation = ''
        elif token == "<triplet>":
            current = "triplet"
        elif token in ["<s>", "<pad>", "</s>"]:
            continue  # Ignore padding or special tokens
        else:
            # Assign tokens based on the current marker
            if current == "subj":
                subject += ' ' + token
            elif current == "obj":
                object_ += ' ' + token
            elif current == "triplet":
                relation += ' ' + token

    # Add the last triplet if it exists
    if subject and object_:
        format_entities.add(subject.strip())
        format_entities.add(object_.strip())

    # Combine NER-based and format-based entities
    combined_entities = ner_entities.union(format_entities)
    return combined_entities
 

# Set device and initialize model/tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "meta-llama/Llama-2-7b-chat-hf"
access_token = "Your access token"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token)

llama = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=access_token,
    torch_dtype=torch.float16
).to(device)

# Initialize entity extractor
entity_extractor = pipeline(
    'text2text-generation',
    model='Babelscape/rebel-large',
    tokenizer='Babelscape/rebel-large',
    device=-1
)

# Load embeddings and entity mapping
model = torch.load('your trained model', map_location=torch.device('cpu'))
entity_embeddings = model.entity_representations[0]._embeddings.weight.detach().numpy()
entity_to_id_df = pd.read_csv('/kaggle/input/entity-model/entity_to_id.tsv', sep='\t')

# Initialize NearestNeighbors for similarity search
knn = NearestNeighbors(n_neighbors=10, metric='cosine')
knn.fit(entity_embeddings)

# Load triplets and claims data
triplets_df = pd.read_json('/kaggle/input/triplets-copy/Copy of test_triplets.json')
triplets_df.columns = range(len(triplets_df.columns))
claims_df = pd.read_csv('/kaggle/input/test-data/transE_Predictions.csv')

# Function to retrieve triplets
def retrieve_triplets(claim):
    extracted_text = entity_extractor.tokenizer.batch_decode(
        [entity_extractor(claim, return_tensors=True, return_text=False)[0]["generated_token_ids"]]
    )
    entities = list(get_entities(extracted_text[0]))

    topK = 10
    claim_entities_ids = entity_to_id_df[entity_to_id_df['label'].isin(entities)]['id'].tolist()
    context_entities_map = {i: set() for i in range(topK)}

    for id in claim_entities_ids:
        query_embedding = entity_embeddings[id].reshape(1, -1)
        distances, indices = knn.kneighbors(query_embedding, n_neighbors=topK)
        indices = indices[0]
        for i, index in enumerate(indices):
            context_entities_map[i].add(entity_to_id_df['label'].iloc[index])

    order = [(i, j) for i in range(topK) for j in range(topK)]
    context = []

    for i, j in order:
        for entity1 in context_entities_map.get(i, []):
            for entity2 in context_entities_map.get(j, []):
                if entity1 == entity2:
                    continue
                condition = ((triplets_df[0] == entity1) & (triplets_df[2] == entity2)) | \
                            ((triplets_df[0] == entity2) & (triplets_df[2] == entity1))
                matching_triplets = triplets_df[condition]
                for _, row in matching_triplets.iterrows():
                    other_row = (row[2], row[1], row[0])
                    if tuple(row) not in context and other_row not in context:
                        context.extend([tuple(row)])

    context = context[:min(len(context), 12)]
    return {'query': claim, 'context': context}

# Wrap the retrieval function in a TransformChain
retrieval_tool = TransformChain(
    input_variables=["input_claim"],
    output_variables=["query", "context"],
    transform=lambda inputs: retrieve_triplets(inputs["input_claim"])
)

# Function to create a prompt
def create_prompt(claim, triplet_context):
    system_prompt = """You are a well-informed and expert fact-checker."""
    user_prompt = (
        f"You are provided with the evidence in the form of triplets retrieved from Knowledge Graph for the following claim:'{claim}'\n"
        f"Evidence: {triplet_context}.\n"
        f"Based on the evidence provided, rate the claim as strictly one of 'True', 'False' or 'NEI'(Not Enough Information). Please provide explanation for the rating."
    )
    prompt_template = PromptTemplate(template="{system_prompt}\n{user_prompt}", input_variables=["system_prompt", "user_prompt"])
    formatted_prompt = prompt_template.format(system_prompt=system_prompt, user_prompt=user_prompt)
    return {'prompt': formatted_prompt}

# Wrap the prompt creation function in a TransformChain
prompt_creation_tool = TransformChain(
    input_variables=["query", "context"],
    output_variables=["prompt"],
    transform=lambda inputs: create_prompt(inputs["query"], inputs["context"])
)

# Function to generate a response
def llama_generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output = llama.generate(
            input_ids,
            max_new_tokens=200,
            repetition_penalty=1.1,
            temperature=0.7,
            num_beams=3,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"final_output": response}

# Wrap response generation in a TransformChain
generation_tool = TransformChain(
    input_variables=["prompt"],
    output_variables=["final_output"],
    transform=lambda inputs: llama_generate_response(inputs["prompt"])
)

# Create a sequential chain for the pipeline
from langchain.chains import SequentialChain
sequential_chain = SequentialChain(
    chains=[retrieval_tool, prompt_creation_tool, generation_tool],
    input_variables=["input_claim"],
    output_variables=["final_output"],
    verbose=True
)

# Process claims and save results
start_index = 250
total_rows = len(claims_df) - start_index

for index, row in tqdm(claims_df.iloc[start_index:].iterrows(), total=total_rows, desc="Processing claims"):
    claim = row['Claim']
    result = sequential_chain.invoke({"input_claim": claim})
    claims_df.at[index, 'Result'] = result['final_output']
    if index % 50 == 0:
        print(index)


claims_df.to_csv('/kaggle/working/transE_Predictions.csv', index=False)