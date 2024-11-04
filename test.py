import os

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access Google API key and Search Engine ID
API_KEY = os.getenv("API_KEY")
CX = os.getenv("CX")


def get_search_count(query):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": API_KEY, "cx": CX, "q": query}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error: {response.json().get('error', {}).get('message')}")
        return 0

    data = response.json()
    try:
        count = int(data["searchInformation"]["totalResults"].replace(",", ""))
    except KeyError:
        count = 0
    return count


def web_jaccard_similarity(P, Q):
    N11 = get_search_count(f"{P} AND {Q}")
    N10 = get_search_count(f"{P} -{Q}")
    N01 = get_search_count(f"{Q} -{P}")

    similarity = N11 / (N11 + N10 + N01) if (N11 + N10 + N01) > 0 else 0
    return similarity


# Example usage
P = "apple"
Q = "fruit"
similarity_score = web_jaccard_similarity(P, Q)
print(f"WebJaccard Similarity between '{P}' and '{Q}': {similarity_score}")
