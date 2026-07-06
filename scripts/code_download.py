import pandas as pd
import requests
import time

def fetch_code(repo, hash, path):
    # Convert github URL metadata to raw GitHub URL
    # Format: https://raw.githubusercontent.com/{org}/{repo}/{hash}/{path}
    # Clean the repo string (remove git@github.com: and .git)
    repo_clean = repo.replace('git@github.com:', '').replace('.git', '')
    raw_url = f"https://raw.githubusercontent.com/{repo_clean}/{hash}/{path.lstrip('/')}"
    
    try:
        response = requests.get(raw_url, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            return None
    except Exception as e:
        print(f"Failed to fetch {path}: {e}")
        return None

# Load your current CSV
df = pd.read_csv("data/mlcq/mlcq.csv")

# Fetch code for each row
print("Fetching source code from GitHub...")
# Updated list comprehension to handle missing values
df['source_code'] = [
    fetch_code(str(r['repository']), str(r['commit_hash']), str(r['path'])) 
    if pd.notna(r['repository']) else None 
    for _, r in df.iterrows()
]

# Save the augmented CSV
df.to_csv("data/mlcq/mlcq_with_code.csv", index=False)
print("Saved augmented CSV to data/mlcq/mlcq_with_code.csv")