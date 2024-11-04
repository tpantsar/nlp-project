from scipy.stats import pearsonr

from logger_config import logger
from utils import (
    calculate_wordnet_correlations,
    datasets,
    load_dataset,
    web_jaccard,
)


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


def calculate_correlations():
    # Compute and print correlations for each dataset
    for name, path in datasets.items():
        correlation, web_jaccard_scores, human_scores = compute_correlation(path)
        if correlation is not None:
            logger.info(f"Pearson correlation for {name}: {correlation:.2f}")
        else:
            logger.error(f"Error computing Pearson correlation for {name}")


def main():
    # normalize_dataset_scores(datasets)
    calculate_wordnet_correlations(datasets)
    # test()
    # calculate_correlations()


if __name__ == "__main__":
    main()
