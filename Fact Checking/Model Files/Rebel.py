from transformers import pipeline
import pandas as pd

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
file_path = "/kaggle/input/claims/Corpus2.csv"  # Replace with your file path
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
output_file = "/kaggle/working/extracted_triplets2.csv"
triplets_df.to_csv(output_file, index=False)

print(f"Triplets extracted and saved to {output_file}")
