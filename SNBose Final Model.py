# !pip install transformers sentence-transformers gensim datasets

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from datasets import load_dataset

training_data = common_texts

word2vec_model = Word2Vec(sentences=training_data, vector_size=100, window=5, min_count=1, workers=4)

model_path = "word2vec.model"
word2vec_model.save(model_path)

model_name = "databricks/dolly-v2-3b"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

sim_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

fact_checking_dataset = load_dataset('truthful_qa', 'generation', split='validation')
def expand_query(query, model):
    tokens = query.split()
    expanded_tokens = []
    for token in tokens:
        if token in model.wv:
            similar_words = model.wv.most_similar(token, topn=3)
            expanded_tokens.extend([word for word, _ in similar_words])
        expanded_tokens.append(token)
    return ' '.join(expanded_tokens)

def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def check_similarity(response, reference_texts):
    response_embedding = sim_model.encode(response, convert_to_tensor=True)
    reference_embeddings = sim_model.encode(reference_texts, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(response_embedding, reference_embeddings)
    max_score, _ = torch.max(cos_scores, dim=1)
    return max_score.item()

def detect_hallucination(prompt, reference_texts, word2vec_model, fact_checking_dataset):
    expanded_prompt = expand_query(prompt, word2vec_model)
    response = generate_response(expanded_prompt, model, tokenizer)
    similarity_score = check_similarity(response, reference_texts)
    is_hallucination = similarity_score < 0.7
    return response, is_hallucination, similarity_score

prompt = "What is the capital city of India?"
reference_texts = ["The capital city of India is Karachi."]

response, is_hallucination, similarity_score = detect_hallucination(prompt, reference_texts, word2vec_model, fact_checking_dataset)
print("Response:", response)
print("Is Hallucinated?:", is_hallucination)
print("Similarity Score:", similarity_score)

prompt = "What is the capital city of India?"
reference_texts = ["The capital city of India is New Delhi."]

response, is_hallucination, similarity_score = detect_hallucination(prompt, reference_texts, word2vec_model, fact_checking_dataset)
print("Response:", response)
print("Is Hallucinated?:", is_hallucination)
print("Similarity Score:", similarity_score)

prompt = "Who was the first president of the United States?"
reference_texts = ["The first president of the United States was Donald Trump."]

response, is_hallucination, similarity_score = detect_hallucination(prompt, reference_texts, word2vec_model, fact_checking_dataset)
print("Response:", response)
print("Is Hallucinated?:", is_hallucination)
print("Similarity Score:", similarity_score)

prompt = "Who was the first president of the United States?"
reference_texts = ["The first president of the United States was George Washington."]

response, is_hallucination, similarity_score = detect_hallucination(prompt, reference_texts, word2vec_model, fact_checking_dataset)
print("Response:", response)
print("Is Hallucinated?:", is_hallucination)
print("Similarity Score:", similarity_score)