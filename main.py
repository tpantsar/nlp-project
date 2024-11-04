import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from scipy.stats import pearsonr

from logger_config import logger
from utils import load_dataset, normalize_dataset_scores, web_jaccard

nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

# Paths to datasets
datasets = {
    "MC": "datasets/mc.csv",
    "RG": "datasets/rg.csv",
    "WordSim353": "datasets/wordsim.csv",
}


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


def calculate_correlations():
    # Compute and print correlations for each dataset
    for name, path in datasets.items():
        correlation, web_jaccard_scores, human_scores = compute_correlation(path)
        if correlation is not None:
            logger.info(f"Pearson correlation for {name}: {correlation:.2f}")
        else:
            logger.error(f"Error computing Pearson correlation for {name}")


def main():
    normalize_dataset_scores(datasets=datasets)
    calculate_wordnet_correlations()
    # test()
    # calculate_correlations()


if __name__ == "__main__":
    main()
