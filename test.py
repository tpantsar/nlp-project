import os

import pandas as pd
from dotenv import load_dotenv

from logger_config import logger
from utils import (
    web_jaccard_similarity,
    get_snippets,
    pre_process,
    snippet_similarity,
    fuzzywuzzy,
)

# Load environment variables from .env file
load_dotenv()

# Access Google API key and Search Engine ID
API_KEY = os.getenv("API_KEY")
CX = os.getenv("CX")


def calculate_webjaccard_scores(dataset, output_file):
    # Read the existing webjaccard_scores.csv file
    if os.path.exists(output_file):
        existing_scores = pd.read_csv(output_file, delimiter=";")
    else:
        existing_scores = pd.DataFrame(columns=["word1", "word2", "web_jaccard"])

    # Create a set of existing word pairs to avoid recalculating
    existing_pairs = set(zip(existing_scores["word1"], existing_scores["word2"]))

    # Read the mc_normalized.csv file
    mc_data = pd.read_csv(dataset, delimiter=";")

    new_scores = []

    for _, row in mc_data.iterrows():
        word1, word2 = row["word1"], row["word2"]
        if (word1, word2) not in existing_pairs and (
            word2,
            word1,
        ) not in existing_pairs:
            similarity = web_jaccard_similarity(word1, word2)
            if similarity == -1:
                logger.error(
                    f"Error occurred while calculating WebJaccard similarity for '{word1}' and '{word2}'"
                )
                continue

            # Round the similarity to 3 decimal places
            similarity = round(similarity, 3)

            new_scores.append(
                {"word1": word1, "word2": word2, "web_jaccard": similarity}
            )
            logger.info(
                f"Calculated WebJaccard similarity for '{word1}' and '{word2}': {similarity}"
            )

    # Append new scores to the existing scores
    if new_scores:
        new_scores_df = pd.DataFrame(new_scores)
        updated_scores = pd.concat([existing_scores, new_scores_df], ignore_index=True)
    else:
        updated_scores = existing_scores

    # Save the updated scores to the output file
    updated_scores.to_csv(output_file, index=False, sep=";")
    logger.info(f"WebJaccard scores saved to '{output_file}'")


def test_web_jaccard():
    P = "automobile"
    Q = "car"
    similarity_score = web_jaccard_similarity(P, Q)
    result = f"WebJaccard Similarity between '{P}' and '{Q}': {similarity_score}"
    logger.info(result)


def test_snippet_similarity(word1, word2):
    snippets = get_snippets(word1)
    # Get first five snippets
    snippets = snippets[:5]
    # Concatenate the snippets
    snippets = " ".join(snippets)

    # Pre-process the snippets
    snippet_1 = pre_process(snippets)
    logger.info(f"{word1} snippets: {snippet_1}")

    snippets = get_snippets(word2)
    # Get first five snippets
    snippets = snippets[:5]
    # Concatenate the snippets
    snippets = " ".join(snippets)

    # Pre-process the snippets
    snippet_2 = pre_process(snippets)
    logger.info(f"{word2} snippets: {snippet_2}")

    similarity = snippet_similarity(snippet_1, snippet_2)
    logger.info(f"Snippet similarity for {word1} and {word2}: {similarity}")

    # Append results to a file
    with open(
        "results/similarities_snippet_similarity_mc.txt", "a", encoding="utf-8"
    ) as file:
        file.write(f"{word1},{word2},{similarity}\n")


def test_fuzzywuzzy(word1, word2):
    snippets = get_snippets(word1)
    snippets = " ".join(snippets)
    snippet1_preprocessed = pre_process(snippets)
    logger.info(f"{word1} snippets: {snippet1_preprocessed}")

    snippets = get_snippets(word2)
    snippets = " ".join(snippets)
    snippet2_preprocessed = pre_process(snippets)
    logger.info(f"{word2} snippets: {snippet2_preprocessed}")

    similarity = fuzzywuzzy(snippet1_preprocessed, snippet2_preprocessed)
    result = f"FuzzyWuzzy Similarity between '{word1}' and '{word2}': {similarity}"
    logger.info(result)

    # Append results to a file
    with open("results/fuzzywuzzy_scores_wordsim.txt", "a", encoding="utf-8") as file:
        file.write(f"{word1},{word2},{similarity}\n")


def calculate_fuzzywuzzy_scores():
    for name, path in datasets.items():
        df = pd.read_csv(path, delimiter=";")
        for _, row in df.iterrows():
            word1, word2 = row["word1"], row["word2"]
            test_fuzzywuzzy(word1, word2)


def calculate_snippet_scores():
    for name, path in datasets.items():
        df = pd.read_csv(path, delimiter=";")
        for _, row in df.iterrows():
            word1, word2 = row["word1"], row["word2"]
            test_snippet_similarity(word1, word2)


if __name__ == "__main__":
    # output_file = "results/webjaccard_scores_rg.csv"
    # dataset = "datasets/rg_normalized.csv"
    # calculate_webjaccard_scores(dataset, output_file)

    # snippets = get_snippets("love")
    # snippets = get_snippets("hate")
    # snippets = get_snippets("pollution")
    # snippets = get_snippets("eco-friendly")
    # snippets = get_snippets("sustainable")
    # snippets = get_snippets("unsustainable")
    # logger.info(snippets)

    datasets = {
        "mc": "datasets/mc_normalized_test.csv",
        # "rg": "datasets/rg_normalized_test.csv",
        # "ws": "datasets/wordsim_normalized_test.csv",
    }

    # calculate_snippet_scores()

    # TODO: calculate pearson correlation for the following datasets
    calculate_fuzzywuzzy_scores()

    # calculate_wordnet_correlations(
    #     {
    #         "webjaccard": output_file,
    #     },
    #     column="web_jaccard",
    # )

    # round_dataset_scores(
    #    {
    #        "webjaccard": output_file,
    #    },
    #    column="web_jaccard",
    # )

    # test_web_jaccard()
