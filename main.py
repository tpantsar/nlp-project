import os

import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

from logger_config import logger

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
        logger.error(f"Error: {response.json().get('error', {}).get('message')}")
        return 0

    data = response.json()
    logger.debug(f"Response for query '{query}': {data}")
    # Check if search information is present in the response
    try:
        count = int(data["searchInformation"]["totalResults"])
        logger.info(f"Result count for query '{query}': {count}")
    except KeyError:  # Handle cases with no results
        count = 0
        logger.warning(f"No results found for query '{query}'")
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
        logger.info(f"Loaded dataset from '{path}'")
    except FileNotFoundError:
        logger.error(f"Error: File not found at '{path}'")
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

    logger.info(f"WebJaccard scores: {web_jaccard_scores}")
    logger.info(f"Human scores: {human_scores}")

    # Compute Pearson correlation
    correlation, _ = pearsonr(web_jaccard_scores, human_scores)
    return correlation, web_jaccard_scores, human_scores


def test():
    # Example usage
    P = "tiger"
    Q = "tiger"
    similarity_score = web_jaccard_similarity(P, Q)
    result = f"WebJaccard Similarity between '{P}' and '{Q}': {similarity_score}"
    logger.info(result)

    # Save the similarity score to a file
    with open("similarity_score.txt", "w") as file:
        file.write(result)


# Paths to datasets
datasets = {
    "MC": "datasets/mc.csv",
    "RG": "datasets/rg.csv",
    "WordSim353": "datasets/wordsim.csv",
}


def normalize_human_scores():
    """Normalize human scores in the DataFrame."""
    for name, path in datasets.items():
        df = load_dataset(path)

        # Normalize the human scores between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit the scaler to the range [0, 10] and transform the values
        df["human_score"] = scaler.fit([[0], [10]]).transform(
            df["human_score"].values.reshape(-1, 1)
        )

        # Round the normalized scores to 3 decimal places
        df["human_score"] = df["human_score"].round(3)

        base, ext = os.path.splitext(path)
        normalized_path = f"{base}_normalized{ext}"

        # Save the updated DataFrame to a new CSV file
        df.to_csv(normalized_path, index=False, sep=";")

        if df is not None:
            logger.info(f"Normalized '{name}' dataset to '{normalized_path}'")
        else:
            logger.error("Error normalizing human similarity scores")


def calculate_correlations():
    # Compute and print correlations for each dataset
    for name, path in datasets.items():
        correlation, web_jaccard_scores, human_scores = compute_correlation(path)
        if correlation is not None:
            logger.info(f"Pearson correlation for {name}: {correlation:.2f}")
        else:
            logger.error(f"Error computing Pearson correlation for {name}")


def main():
    # normalize_human_scores()
    test()
    # calculate_correlations()


if __name__ == "__main__":
    main()
