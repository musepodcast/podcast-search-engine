from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import time

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

sentences = ['This is an example sentence', 'Each sentence is converted']
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', torch_dtype=torch.float16)
model.to('cuda')
model.eval()

# Move inputs to GPU
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
encoded_input = {k: v.to('cuda') for k, v in encoded_input.items()}

start = time.time()
with torch.no_grad():
    model_output = model(**encoded_input)
embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
embeddings = F.normalize(embeddings, p=2, dim=1)
print("Sentence embeddings:")
print(embeddings)
print("Time taken:", time.time() - start)
