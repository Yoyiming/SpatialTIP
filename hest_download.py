from huggingface_hub import login
import datasets
import os
my_token = "Your HuggingFace token"
login(token=my_token, add_to_git_credential=True)

local_dir='hest_data'
ids_to_query = ['MISC1', 'MISC5', 'TENX13', 'NCBI572'] # Datasets ID to download

list_patterns = [f"*{id}[_.]**" for id in ids_to_query]

dataset = datasets.load_dataset(
    'MahmoodLab/hest', 
    cache_dir=local_dir,
    patterns=list_patterns
)
