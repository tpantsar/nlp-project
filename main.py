import os

import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

# Load environment variables from .env file
load_dotenv()

# Access Google API key and Search Engine ID
API_KEY = os.getenv("API_KEY")
CX = os.getenv("CX")


# Define a function to get the result count for a given query
def get_search_count(query):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": API_KEY, "cx": CX, "q": query}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error: {response.json().get('error', {}).get('message')}")
        return 0

    data = response.json()
    # Check if search information is present in the response
    try:
        count = int(data["searchInformation"]["totalResults"])
    except KeyError:
        count = 0  # Handle cases with no results
    return count


def web_jaccard_similarity(P, Q):
    """WebJaccard similarity function for a word pair P and Q."""
    N11 = get_search_count(f"{P} AND {Q}")
    N10 = get_search_count(f"{P} -{Q}")
    N01 = get_search_count(f"{Q} -{P}")

    # Calculate the WebJaccard similarity
    similarity = N11 / (N11 + N10 + N01) if (N11 + N10 + N01) > 0 else 0
    return similarity


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a dataset from the specified path.
    The score is not necessarily normalized among these datasets.
    """
    try:
        df = pd.read_csv(path, delimiter=";")
        scaler = MinMaxScaler()

        # Normalize the human scores between 0 and 1
        df["human_score"] = scaler.fit_transform(
            df["human_score"].values.reshape(-1, 1)
        )

        # Round the normalized scores to 2 decimal places
        df["human_score"] = df["human_score"].round(2)

        base, ext = os.path.splitext(path)
        normalized_path = f"{base}_normalized{ext}"

        # Save the updated DataFrame to a new CSV file
        df.to_csv(normalized_path, index=False, sep=";")
    except FileNotFoundError:
        print(f"Error: File not found at '{path}'")
        return None
    return df


def compute_correlation(dataset_path):
    """Compute the correlation between the WebJaccard similarity and human ratings."""
    # Load the dataset
    df = load_dataset(dataset_path)
    if df is None:
        return

    # Initialize lists to store WebJaccard scores and human-annotated scores
    web_jaccard_scores = []
    human_scores = df["human_score"].tolist()

    for _, row in df.iterrows():
        P, Q = row["word1"], row["word2"]
        score = web_jaccard_similarity(P, Q)
        web_jaccard_scores.append(score)

    # Compute Pearson correlation
    correlation, _ = pearsonr(web_jaccard_scores, human_scores)
    return correlation, web_jaccard_scores, human_scores


# Example usage
P = "laptop"
Q = "computer"
similarity_score = web_jaccard_similarity(P, Q)
print(f"WebJaccard Similarity between '{P}' and '{Q}': {similarity_score}")

# Paths to datasets
datasets = {
    "MC": "datasets/mc.csv",
    "RG": "datasets/rg.csv",
    "WordSim353": "datasets/wordsim.csv",
}

# Compute and print correlations for each dataset
for name, path in datasets.items():
    correlation, web_jaccard_scores, human_scores = compute_correlation(path)
    if correlation is not None:
        print(f"Pearson correlation for {name}: {correlation:.2f}")
    else:
        print(f"Error computing Pearson correlation for {name}")
