import os

import pandas as pd
from dotenv import load_dotenv

from logger_config import logger
from utils import calculate_wordnet_correlations, web_jaccard

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
            similarity = web_jaccard(word1, word2)
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


if __name__ == "__main__":
    output_file = "results/webjaccard_scores_mc.csv"
    # dataset = "datasets/mc_normalized.csv"
    # calculate_webjaccard_scores(dataset, output_file)

    calculate_wordnet_correlations(
        {
            "webjaccard": output_file,
        },
        column="web_jaccard",
    )

    # round_dataset_scores(
    #    {
    #        "webjaccard": output_file,
    #    },
    #    column="web_jaccard",
    # )

    word1 = "automobile"
    word2 = "car"

    similarity_score = web_jaccard(word1, word2)
    logger.info(
        f"WebJaccard similarity between '{word1}' and '{word2}': {similarity_score}"
    )
