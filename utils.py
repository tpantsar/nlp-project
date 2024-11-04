import json
import math
import os
from transformers import DistilBertModel, DistilBertTokenizer
import gensim
import torch
import nltk
import gensim.downloader as api
import numpy as np
import pandas as pd
import requests
from fuzzywuzzy import fuzz
from gensim.models import FastText, Word2Vec
from dotenv import load_dotenv
from nltk import WordNetLemmatizer
from nltk.corpus import abc, brown
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from logger_config import logger

# nltk.download("abc")
# nltk.download("punkt")
# nltk.download('punkt_tab')
# nltk.download("brown")
# nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger_eng")
nltk.download("stopwords")

# TODO: Uncomment
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = DistilBertModel.from_pretrained("distilbert-base-uncased", torch_dtype=torch.float16, attn_implementation="eager")

# Load environment variables from .env file
load_dotenv()

# Access Google API key and Search Engine ID
API_KEY = os.getenv("API_KEY")
CX = os.getenv("CX")

# Paths to datasets
datasets = {
    "MC": "datasets/mc_normalized_test.csv",
    "RG": "datasets/rg_normalized_test.csv",
    "WordSim353": "datasets/wordsim_normalized_test.csv",
}

abc_sentences = abc.sents()
brown_sentences = brown.sents()
glove_model = api.load("glove-twitter-25")

# Train a small Word2Vec model on the "abc" corpus
word2vec_model = Word2Vec(
    abc_sentences, vector_size=100, window=5, min_count=1, epochs=10
)

# Train a small FastText model on the "brown" corpus
fasttext_model = FastText(
    brown_sentences, vector_size=100, window=5, min_count=1, epochs=10
)

def wu_palmer(S1, S2):
    return S1.wup_similarity(S2)


def path_length(S1, S2):
    return S1.path_similarity(S2)


def lch(S1, S2):
    lch_score = S1.lch_similarity(S2)
    if lch_score is not None:
        normalized_lch_score = lch_score / MAX_LCH_SCORE
        return normalized_lch_score
    return None


def web_jaccard(P, Q):
    """WebJaccard similarity function for a word pair P and Q."""
    N11 = get_search_count(f"{P} AND {Q}")
    N10 = get_search_count(f"{P} -{Q}")
    N01 = get_search_count(f"{Q} -{P}")

    if any(count == -1 for count in (N11, N10, N01)):
        return -1

    similarity = N11 / (N11 + N10 + N01) if (N11 + N10 + N01) > 0 else 0
    return similarity


def snippet_similarity(snippet1, snippet2):
    """WebJaccard similarity function for a word pair P and Q."""
    # Calculate the ratio of the common words between the five snippets of P and the five snippets of Q
    # over the total number of distinct words of all the ten snippets

    # Get common words between the two snippets
    common_words = set(snippet1).intersection(snippet2)
    logger.info(f"Common words: {common_words}")
    # Get all distinct words from both snippets
    all_words = set(snippet1).union(snippet2)
    logger.info(f"All words: {all_words}")

    similarity = len(common_words) / len(all_words) if len(all_words) > 0 else 0

    return similarity
    

def fuzzywuzzy(snippet1, snippet2):
    """
    FuzzyWuzzy similarity, accounts only for string matching.
    Calculate the number of shared snippets and the percentage of overlapping between two sets of snippets.
    """
    # Concatenate the documents into strings
    D1 = " ".join(snippet1)
    D2 = " ".join(snippet2)

    # Preprocess the documents
    D1 = pre_process(D1)
    D2 = pre_process(D2)

    # Calculate the percentage of overlapping using fuzzy-string matching
    overlap_percentage = fuzz.ratio(D1, D2) / 100

    return overlap_percentage


def word2vec(S1, S2):
    """Word2Vec similarity."""
    return similarity_model(word2vec_model, S1.definition(), S2.definition())


def fasttext(S1, S2):
    """FastText similarity."""
    return similarity_model(fasttext_model, S1.definition(), S2.definition())


def glove(S1, S2):
    """Glove similarity."""
    return similarity_model(glove_model, S1.definition(), S2.definition())


def distilbert(S1, S2):
    """DistilBERT similarity."""
    return distilbert_similarity(S1.definition(), S2.definition())


def pre_process(sentence):
    """Tokenize, remove stopwords, and clean the sentence."""
    stop_words = list(set(nltk.corpus.stopwords.words("english")))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(sentence)

    filtered_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words
    ]

    tokens = [
        word.lower() for word in filtered_tokens if word.isalpha() and word not in stop_words
    ]  # Get rid of numbers and stopwords

    # Remove duplicate tokens
    tokens = list(set(tokens))

    return tokens


def similarity_model(model, str1, str2):
    def get_vector(sentence):
        words = word_tokenize(sentence.lower())
        if isinstance(model, gensim.models.KeyedVectors):
            words = [word for word in words if word in model]
            if not words:
                return np.zeros(model.vector_size)
            return np.mean([model[word] for word in words], axis=0)
        else:
            words = [word for word in words if word in model.wv]
            if not words:
                return np.zeros(model.vector_size)
            return np.mean([model.wv[word] for word in words], axis=0)

    vec1 = get_vector(str1)
    vec2 = get_vector(str2)
    return cosine_similarity([vec1], [vec2])[0][0]


def distilbert_similarity(sentence1, sentence2):
    # Tokenize sentences
    inputs1 = tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(sentence2, return_tensors="pt", padding=True, truncation=True)
    
    # Generate embeddings
    with torch.no_grad():
        outputs1 = distilbert_model(**inputs1).last_hidden_state.mean(dim=1)  # Mean pooling over tokens for sentence embedding
        outputs2 = distilbert_model(**inputs2).last_hidden_state.mean(dim=1)

    # Calculate cosine similarity
    similarity = cosine_similarity(outputs1, outputs2).item()
    return similarity


# Maximum depth of the WordNet taxonomy
MAX_DEPTH = 20

# Maximum possible LCH similarity score
MAX_LCH_SCORE = -math.log(1 / (2 * MAX_DEPTH))


# Wordnet similarity models
models = {
    0: wu_palmer,
    1: path_length,
    2: lch,
    3: web_jaccard,
    4: word2vec,
    5: fasttext,
    6: glove,
    7: distilbert,
    8: snippet_similarity,
    9: fuzzywuzzy,
}


def get_search_count(query):
    """Get the number of search results for a query using the Google Custom Search API."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": API_KEY, "cx": CX, "q": query}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        logger.error(f"Error: {response.json().get('error', {}).get('message')}")
        return 0

    data = response.json()
    with open("data.json", "w") as f:
        json.dump(data, f, indent=4)

    logger.debug(f"Response for query '{query}': {data}")
    # Check if search information is present in the response
    try:
        count = int(data["searchInformation"]["totalResults"])
        logger.info(f"Result count for query '{query}': {count}")
    except KeyError:  # Handle cases with no results
        count = 0
        logger.warning(f"No results found for query '{query}'")
    return count


def get_snippets(query):
    """Get search snippets for a query using the Google Custom Search API."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": API_KEY, "cx": CX, "q": query}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        logger.error(f"Error: {response.json().get('error', {}).get('message')}")
        return 0

    data = response.json()
    with open("data.json", "w") as f:
        json.dump(data, f, indent=4)

    logger.debug(f"Response for query '{query}': {data}")

    try:
        items = data.get("items", [])
        snippets = [item.get("snippet") for item in items]

        # Append snippets to a text file
        with open("snippets.txt", "a", encoding="utf-8") as f:
            f.write(f"Query: {query}\n")
            f.write("\n".join(snippets))
            f.write("\n\n")
    except KeyError:
        logger.warning(f"No snippets found for query '{query}'")
    return snippets


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


def calculate_wordnet_correlations(datasets: dict):
    # Compute and print correlations for each dataset
    results = []
    for name, path in datasets.items():
        df = pd.read_csv(path, delimiter=";")
        web_jaccard_scores = df["web_jaccard_score"].values
        human_scores = df["human_score"].values
        snippet_scores_1 = df["snippet_score1"].values
        snippet_scores_2 = df["snippet_score2"].values
        wu_palmer_scores = []
        path_length_scores = []
        lch_scores = []
        word2vec_scores = []
        fasttext_scores = []
        glove_scores = []
        distilbert_scores = []

        for _, row in df.iterrows():
            P, Q = row["word1"], row["word2"]

            wu_palmer = wordnet_similarity(P, Q, 0)
            path_length = wordnet_similarity(P, Q, 1)
            lch = wordnet_similarity(P, Q, 2)
            word2vec = wordnet_similarity(P, Q, 4)
            fasttext = wordnet_similarity(P, Q, 5)
            glove = wordnet_similarity(P, Q, 6)
            distilbert = wordnet_similarity(P, Q, 7)

            wu_palmer_scores.append(wu_palmer)
            path_length_scores.append(path_length)
            lch_scores.append(lch)
            word2vec_scores.append(word2vec)
            fasttext_scores.append(fasttext)
            glove_scores.append(glove)
            distilbert_scores.append(distilbert)

        df["web_jaccard_similarity"] = web_jaccard_scores
        df["wu_palmer_similarity"] = wu_palmer_scores
        df["path_length_similarity"] = path_length_scores
        df["lch_similarity"] = lch_scores
        df["word2vec_similarity"] = word2vec_scores
        df["fasttext_similarity"] = fasttext_scores
        df["glove_similarity"] = glove_scores
        df["distilbert_similarity"] = distilbert_scores
        df["snippet_similarity"] = snippet_scores_1

        # Calculate Pearson correlations
        web_jaccard_corr, _ = pearsonr(df["web_jaccard_similarity"], human_scores)
        wu_palmer_corr, _ = pearsonr(df["wu_palmer_similarity"], human_scores)
        path_length_corr, _ = pearsonr(df["path_length_similarity"], human_scores)
        lch_corr, _ = pearsonr(df["lch_similarity"], human_scores)
        word2vec_corr, _ = pearsonr(df["word2vec_similarity"], human_scores)
        fasttext_corr, _ = pearsonr(df["fasttext_similarity"], human_scores)
        glove_corr, _ = pearsonr(df["glove_similarity"], human_scores)
        distilbert_corr, _ = pearsonr(df["distilbert_similarity"], human_scores)
        snippet_corr, _ = pearsonr(df["snippet_similarity"], human_scores)

        results.append(
            {
                "Dataset": name,
                "WebJaccard": web_jaccard_corr,
                "WuPalmer": wu_palmer_corr,
                "PathLength": path_length_corr,
                "LCH": lch_corr,
                "Word2Vec": word2vec_corr,
                "FastText": fasttext_corr,
                "Glove": glove_corr,
                "DistilBERT": distilbert_corr,
                "Snippet": snippet_corr,
            }
        )

        logger.info(
            f"Pearson correlation for {name} - WebJaccard: {web_jaccard_corr:.2f}, WuPalmer: {wu_palmer_corr:.2f}, PathLength: {path_length_corr:.2f}, LCH: {lch_corr:.2f}, Word2Vec: {word2vec_corr:.2f}, FastText: {fasttext_corr:.2f}, Glove: {glove_corr:.2f}, DistilBERT: {distilbert_corr:.2f}, Snippet: {snippet_corr:.2f}"
        )

    # Summarize results in a table
    results_df = pd.DataFrame(results).round(2)
    logger.info(results_df)


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
