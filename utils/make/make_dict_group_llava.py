import json
from collections import defaultdict
from tqdm import tqdm
from nltk.stem import PorterStemmer
from transformers import LlamaTokenizer
import nltk
import re

nltk.download('punkt')

tokenizer = LlamaTokenizer.from_pretrained("vip_llava/vip-llava-7b")
stemmer = PorterStemmer()

vocab_size = tokenizer.vocab_size
id2str = {}
stem2ids = defaultdict(list)

for tid in tqdm(range(vocab_size)):
    word = tokenizer.decode([tid]).strip("▁Ġ ").lower()
    
    if not word or word.strip() == "":
        continue
    
    if not re.match(r"^[a-z0-9\-]+$", word):
        continue

    id2str[tid] = word
    stem = stemmer.stem(word)
    stem2ids[stem].append(tid)

token_groups = list(stem2ids.values())
with open("token_groups.json", "w") as f:
    json.dump(token_groups, f, indent=2)

with open("id2str.json", "w") as f:
    json.dump(id2str, f, indent=2, ensure_ascii=False)

print(f"[✓] Saved {len(token_groups)} token groups and {len(id2str)} id-to-string mappings.")
