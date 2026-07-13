import json
with open(r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\config.json') as f:
    c = json.load(f)
for k in ['hidden_size','num_hidden_layers','intermediate_size','num_attention_heads','num_key_value_heads','vocab_size','hidden_act']:
    print(f'{k}: {c.get(k)}')
