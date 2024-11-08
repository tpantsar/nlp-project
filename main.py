from logger_config import logger
from utils import (
    calculate_wordnet_correlations,
    datasets,
    fuzzywuzzy,
    normalize_dataset_scores,
    web_jaccard,
)


def test():
    # Example usage
    P = "book"
    Q = "library"
    similarity_score = web_jaccard(P, Q)
    result = f"WebJaccard Similarity between '{P}' and '{Q}': {similarity_score}"
    logger.info(result)


def calculate_overlap_percentage():
    """Calculate the overlap percentage between two snippets."""
    documents = [
        ("documents/love.txt", "documents/hate.txt"),
        ("documents/pollution.txt", "documents/ecofriendly.txt"),
        ("documents/sustainable.txt", "documents/unsustainable.txt"),
    ]

    # Summarize the result of the three double queries in a table.
    for doc1, doc2 in documents:
        snippet1 = open(doc1).read()
        snippet2 = open(doc2).read()
        overlap_percentage = fuzzywuzzy(snippet1, snippet2)
        logger.info(
            f"Snippet overlap between '{doc1}' and '{doc2}': {overlap_percentage:.2f}"
        )


def main():
    # normalize_dataset_scores(datasets)
    calculate_wordnet_correlations(datasets)
    calculate_overlap_percentage()


if __name__ == "__main__":
    main()
