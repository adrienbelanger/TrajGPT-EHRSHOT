import pandas as pd
import numpy as np
import json

print("Starting...")
with open('data/vocab.json', 'r') as f:
    vocab = json.load(f)
vocab_size = len(vocab)

d_model = 200  

new_phecodes = list(range(vocab_size + 1))
random_embeddings = np.random.normal(scale=0.1, size=(len(new_phecodes), d_model))


emb_cols = [f'dim_{i}' for i in range(d_model)]
new_emb = pd.DataFrame(random_embeddings, columns=emb_cols)
new_emb.insert(0, 'phecode', new_phecodes)


new_emb.to_csv('data/phecode_embeddings.csv', index=False)
print("Success!")