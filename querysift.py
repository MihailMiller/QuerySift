import argparse
import os
import json
import pandas as pd
import re
import string
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
from docx.enum.text import WD_COLOR_INDEX

# Setup command line arguments
parser = argparse.ArgumentParser(description="Process and analyze text data using NLP models.")
parser.add_argument("--chunk_size", type=int, default=1000, help="Size of text chunks for processing.")
parser.add_argument("--chunk_overlap", type=int, default=500, help="Overlap size between consecutive text chunks.")
parser.add_argument("--highlight_limit", type=int, default=5, help="Maximum number of key phrases to highlight in the text.")
parser.add_argument("--top_results_limit", type=int, default=50, help="Number of top entries to retain per query for analysis.")
parser.add_argument("--minimum_merge_overlap", type=int, default=10, help="Minimum number of overlapping words required to merge text snippets.")
parser.add_argument("--embedding_model", type=str, default="jinaai/jina-embeddings-v3", help="Transformer model name for generating embeddings.")
parser.add_argument("--qa_model", type=str, default="deepset/gelectra-large-germanquad", help="Transformer model name for question answering.")
args = parser.parse_args()

# Use command line arguments
WINDOW_SIZE = args.chunk_size
OVERLAP = args.chunk_overlap
HIGHLIGHT_LIMIT = args.highlight_limit
TOP_EMB_LIMIT = args.top_results_limit
MIN_OVERLAP = args.minimum_merge_overlap
model_name = args.embedding_model
qa_model_name = args.qa_model

# Load models and tokenizers
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
nlp = pipeline('question-answering', model=qa_model_name, tokenizer=qa_model_name, topk=HIGHLIGHT_LIMIT)

# Enable CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load questions from JSON
def load_questions_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

category_questions = load_questions_from_json('category_questions.json')

def extract_questions_with_categories(category_dict, category=None, subcategory=None):
    queries = []
    for key, value in category_dict.items():
        if isinstance(value, dict):
            new_category = category if category else key  # Preserve category level
            new_subcategory = key if category else None  # Only assign subcategory if category is already set
            queries.extend(extract_questions_with_categories(value, new_category, new_subcategory))
        elif isinstance(value, list):
            for query in value:
                queries.append((query, category, subcategory, key))
    return queries

query_data = extract_questions_with_categories(category_questions)
queries, categories, subcategories, subsubcategories = zip(*query_data)

# Encode texts in batches
def encode_texts(texts, batch_size=16):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_outputs = model.encode(batch, task="text-matching")
            embeddings.extend(batch_outputs)
    return torch.tensor(embeddings).to(device)

# Calculate cosine similarity
def calculate_similarity(query_embeddings, passage_embeddings):
    return cosine_similarity(query_embeddings.cpu().numpy(), passage_embeddings.cpu().numpy())

# Normalize text for duplicate removal
def normalize_text(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

# Highlight relevant key phrases without duplication
def highlight_relevant_text(query, text, doc):
    QA_input = {'question': query, 'context': text}
    res = nlp(QA_input)
    para = doc.add_paragraph()
    
    matches = []  # Stores (start_idx, end_idx) for all valid matches
    covered_idx = set()  # To keep track of indices already covered

    for answer in res:
        for match in re.finditer(re.escape(answer['answer']), text):
            start_idx, end_idx = match.start(), match.end()
            # Check if the text span is already covered
            if not any(idx in covered_idx for idx in range(start_idx, end_idx)):
                matches.append((start_idx, end_idx))
                covered_idx.update(range(start_idx, end_idx))
    
    # Sort matches to maintain correct text order
    matches = sorted(matches)  # Remove duplicates while keeping order
    last_index = 0
    
    for start_idx, end_idx in matches:
        if last_index < start_idx:
            para.add_run(text[last_index:start_idx])  # Normal text
        run = para.add_run(text[start_idx:end_idx])  # Highlighted text
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        last_index = end_idx
    
    if last_index < len(text):
        para.add_run(text[last_index:])

# Function to merge snippets based on content overlap
def merge_snippets_based_on_content(snippets, MIN_OVERLAP):
    if not snippets:
        return []
    
    merged_snippets = [snippets[0]]  # Start with the first snippet

    for current_snippet in snippets[1:]:
        previous_snippet = merged_snippets[-1]
        prev_words = previous_snippet.split()
        current_words = current_snippet.split()
        
        overlap_length = 0
        for i in range(1, min(len(prev_words), len(current_words)) + 1):
            if prev_words[-i:] == current_words[:i]:
                overlap_length = i
        
        if overlap_length >= MIN_OVERLAP:
            merged_snippets[-1] = ' '.join(prev_words + current_words[overlap_length:])
        else:
            merged_snippets.append(current_snippet)

    return merged_snippets

# Process text files in directory
input_dir = "texts"
text_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

batch_results = []
query_embeddings = encode_texts(queries)

for filename in tqdm(text_files, desc="Processing text files"):
    file_path = os.path.join(input_dir, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        full_text = file.read()

    # Split the text into chunks based on the window size and overlap
    snippets = [full_text[i:i + WINDOW_SIZE] for i in range(0, max(1, len(full_text) - WINDOW_SIZE + 1), WINDOW_SIZE - OVERLAP)]
    merged_snippets = merge_snippets_based_on_content(snippets, MIN_OVERLAP)

    seen_passages = set()
    file_results = []
    for snippet in merged_snippets:
        norm_snippet = normalize_text(snippet)
        if norm_snippet in seen_passages:
            continue  # Skip duplicate-like passages
        seen_passages.add(norm_snippet)

        snippet_embedding = encode_texts([snippet])
        similarities = calculate_similarity(query_embeddings, snippet_embedding)

        for k in range(len(queries)):
            similarity = similarities[k, 0]
            file_results.append([queries[k], categories[k], subcategories[k], subsubcategories[k], similarity, filename, snippet])

    batch_results.extend(file_results)

# Convert results to DataFrame
df = pd.DataFrame(batch_results, columns=["Query", "Category", "Subcategory", "Subsubcategory", "Similarity", "Filename", "Extract"])
df_sorted = df.sort_values(by=["Query", "Similarity"], ascending=[True, False])
df_top_emb = df_sorted.groupby("Query").head(TOP_EMB_LIMIT)  # Use TOP_EMB_LIMIT global variable

# Remove duplicates based on normalized text
df_top_emb["Normalized_Extract"] = df_top_emb["Extract"].apply(normalize_text)
df_top_emb = df_top_emb.loc[df_top_emb.groupby(["Filename", "Normalized_Extract"])["Similarity"].idxmax()].drop(columns=["Normalized_Extract"])

# Save to Excel
output_file = "output_results.xlsx"
with pd.ExcelWriter(output_file) as writer:
    df_top_emb.to_excel(writer, sheet_name="Top Matches", index=False)

# Generate Word reports and save them in category-based subfolders
def process_and_generate_word(input_file, output_dir):
    df = pd.read_excel(input_file)
    grouped = df.groupby("Query")
    
    for query, group in grouped:
        category = group.iloc[0]["Category"] or "Uncategorized"
        subcategory = group.iloc[0]["Subcategory"] or "General"
        subsubcategory = group.iloc[0]["Subsubcategory"] or "Misc"
        
        query_dir = os.path.join(output_dir, subcategory, subsubcategory)
        os.makedirs(query_dir, exist_ok=True)
        
        safe_query = query.translate(str.maketrans('', '', string.punctuation)).replace(' ', '_')
        doc = Document()
        doc.add_heading(query, level=1)
        
        for filename, file_group in group.groupby("Filename"):
            doc.add_heading(filename, level=2)
            for _, row in file_group.iterrows():
                highlight_relevant_text(query, row["Extract"], doc)
                doc.add_paragraph("\n---\n")
        
        file_path = os.path.join(query_dir, f"{safe_query}.docx")
        doc.save(file_path)
        print(f"Saved: {file_path}")

process_and_generate_word(output_file, "analysis_word")
