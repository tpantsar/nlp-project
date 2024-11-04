import os

import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

from logger_config import logger

# Load environment variables from .env file
load_dotenv()

# Access Google API key and Search Engine ID
API_KEY = os.getenv("API_KEY")
CX = os.getenv("CX")


def wu_palmer(S1, S2):
    return S1.wup_similarity(S2)


def path_length(S1, S2):
    return S1.path_similarity(S2)


def lch(S1, S2):
    return S1.lch_similarity(S2)


def web_jaccard(P, Q):
    """WebJaccard similarity function for a word pair P and Q."""
    N11 = get_search_count(f"{P} AND {Q}")
    N10 = get_search_count(f"{P} -{Q}")
    N01 = get_search_count(f"{Q} -{P}")

    if any(count == -1 for count in (N11, N10, N01)):
        return -1

    similarity = N11 / (N11 + N10 + N01) if (N11 + N10 + N01) > 0 else 0
    return similarity


def get_search_count(query):
    """Get the number of search results for a query using the Google Custom Search API."""
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


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a dataset from the specified path.
    The score is not necessarily normalized among these datasets.
    """
    try:
        df = pd.read_csv(path, delimiter=";")
        logger.debug(f"Loaded dataset from '{path}'")
    except FileNotFoundError:
        logger.error(f"Error: File not found at '{path}'")
        return None
    return df


def normalize_dataset_scores(datasets: dict, column: str = "human_score"):
    """Normalize dataset scores in the DataFrame for [0, 1] range."""
    for name, path in datasets.items():
        df = load_dataset(path)

        # Normalize the human scores between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit the scaler to the range [0, 10] and transform the values
        df[column] = scaler.fit([[0], [10]]).transform(df[column].values.reshape(-1, 1))

        # Round the normalized scores to 3 decimal places
        df[column] = df[column].round(3)

        base, ext = os.path.splitext(path)
        normalized_path = f"{base}_normalized{ext}"

        # Save the updated DataFrame to a new CSV file
        df.to_csv(normalized_path, index=False, sep=";")

        if df is not None:
            logger.debug(f"Normalized '{name}' dataset to '{normalized_path}'")
        else:
            logger.error("Error normalizing similarity scores in the dataset")


def round_dataset_scores(
    datasets: dict, column: str = "human_score", decimals: int = 3
):
    """Round dataset scores in the DataFrame."""
    for name, path in datasets.items():
        df = load_dataset(path)

        # Round the human scores to the specified number of decimal places
        df[column] = df[column].round(decimals)

        base, ext = os.path.splitext(path)
        rounded_path = f"{base}_rounded{ext}"

        # Save the updated DataFrame to a new CSV file
        df.to_csv(rounded_path, index=False, sep=";")

        if df is not None:
            logger.debug(f"Rounded '{name}' dataset to '{rounded_path}'")
        else:
            logger.error("Error rounding similarity scores in the dataset")
