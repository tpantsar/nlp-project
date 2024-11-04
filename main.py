import os

import nltk
import pandas as pd
import requests
from dotenv import load_dotenv
from nltk.corpus import wordnet as wn
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

from logger_config import logger

nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

# Load environment variables from .env file
load_dotenv()

# Access Google API key and Search Engine ID
API_KEY = os.getenv("API_KEY")
CX = os.getenv("CX")

# Paths to datasets
datasets = {
    "MC": "datasets/mc.csv",
    "RG": "datasets/rg.csv",
    "WordSim353": "datasets/wordsim.csv",
}


def web_jaccard(P, Q):
    """WebJaccard similarity function for a word pair P and Q."""
    N11 = get_search_count(f"{P} AND {Q}")
    N10 = get_search_count(f"{P} -{Q}")
    N01 = get_search_count(f"{Q} -{P}")

    # Calculate the WebJaccard similarity
    similarity = N11 / (N11 + N10 + N01) if (N11 + N10 + N01) > 0 else 0
    return similarity


def wu_palmer(S1, S2):
    return S1.wup_similarity(S2)


def path_length(S1, S2):
    return S1.path_similarity(S2)


def lch(S1, S2):
    return S1.lch_similarity(S2)


# Wordnet similarity models
models = {
    0: wu_palmer,
    1: path_length,
    2: lch,
    3: web_jaccard,
}


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


def get_wordnet_pos(word):
    """Map POS tag to first character for lemmatization with WordNet."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wn.ADJ, "N": wn.NOUN, "V": wn.VERB, "R": wn.ADV}
    return tag_dict.get(tag, wn.NOUN)


def wordnet_similarity(word1, word2, model_index):
    """Calculate similarity between two words only if they share the same POS."""
    pos1 = get_wordnet_pos(word1)
    pos2 = get_wordnet_pos(word2)

    synsets1 = wn.synsets(word1, pos=pos1)
    synsets2 = wn.synsets(word2, pos=pos2)

    if synsets1 and synsets2:
        S1 = synsets1[0]
        S2 = synsets2[0]
        try:
            similarity = models[model_index](S1, S2)
            if similarity:
                return round(similarity, 2)
        except nltk.corpus.reader.wordnet.WordNetError:
            return 0
    return 0


def calculate_wordnet_correlations():
    # Compute and print correlations for each dataset
    results = []
    for name, path in datasets.items():
        df = pd.read_csv(path, delimiter=";")
        # web_jaccard_scores = []
        wu_palmer_scores = []
        path_length_scores = []
        lch_scores = []
        human_scores = df["human_score"].values

        for _, row in df.iterrows():
            P, Q = row["word1"], row["word2"]
            # web_jaccard = web_jaccard_similarity(P, Q)
            wu_palmer = wordnet_similarity(P, Q, 0)
            path_length = wordnet_similarity(P, Q, 1)
            lch = wordnet_similarity(P, Q, 2)

            # web_jaccard_scores.append(web_jaccard)
            wu_palmer_scores.append(wu_palmer)
            path_length_scores.append(path_length)
            lch_scores.append(lch)

        # df["web_jaccard_similarity"] = web_jaccard_scores
        df["wu_palmer_similarity"] = wu_palmer_scores
        df["path_length_similarity"] = path_length_scores
        df["lch_similarity"] = lch_scores

        # Calculate Pearson correlations
        # web_jaccard_corr, _ = pearsonr(df["web_jaccard_similarity"], human_scores)
        wu_palmer_corr, _ = pearsonr(df["wu_palmer_similarity"], human_scores)
        path_length_corr, _ = pearsonr(df["path_length_similarity"], human_scores)
        lch_corr, _ = pearsonr(df["lch_similarity"], human_scores)

        results.append(
            {
                "Dataset": name,
                # "WebJaccard": web_jaccard_corr,
                "WuPalmer": wu_palmer_corr,
                "PathLength": path_length_corr,
                "LCH": lch_corr,
            }
        )

        # logger.info(
        #     f"Pearson correlation for {name} - WebJaccard: {web_jaccard_corr:.2f}, WuPalmer: {wu_palmer_corr:.2f}, PathLength: {path_length_corr:.2f}, LCH: {lch_corr:.2f}"
        # )
        logger.info(
            f"Pearson correlation for {name} - WuPalmer: {wu_palmer_corr:.2f}, PathLength: {path_length_corr:.2f}, LCH: {lch_corr:.2f}"
        )

    # Summarize results in a table
    results_df = pd.DataFrame(results).round(2)
    logger.info(results_df)


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
        score = web_jaccard(P, Q)
        web_jaccard_scores.append(score)

    logger.info(f"WebJaccard scores: {web_jaccard_scores}")
    logger.info(f"Human scores: {human_scores}")

    # Compute Pearson correlation
    correlation, _ = pearsonr(web_jaccard_scores, human_scores)
    return correlation, web_jaccard_scores, human_scores


def test():
    # Example usage
    P = "book"
    Q = "library"
    similarity_score = web_jaccard(P, Q)
    result = f"WebJaccard Similarity between '{P}' and '{Q}': {similarity_score}"
    logger.info(result)


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
            logger.debug(f"Normalized '{name}' dataset to '{normalized_path}'")
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
    normalize_human_scores()
    calculate_wordnet_correlations()
    # test()
    # calculate_correlations()


if __name__ == "__main__":
    main()
