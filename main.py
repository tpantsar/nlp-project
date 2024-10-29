import os

import requests
from dotenv import load_dotenv

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
    data = response.json()
    # Check if search information is present in the response
    try:
        count = int(data["searchInformation"]["totalResults"])
    except KeyError:
        count = 0  # Handle cases with no results
    return count


# Define the WebJaccard similarity function
def web_jaccard_similarity(P, Q):
    # Calculate N11: the count of results for "P AND Q"
    N11_query = f"{P} AND {Q}"
    N11 = get_search_count(N11_query)

    # Calculate N10: the count of results for "P AND NOT Q"
    N10_query = f"{P} -{Q}"
    N10 = get_search_count(N10_query)

    # Calculate N01: the count of results for "Q AND NOT P"
    N01_query = f"{Q} -{P}"
    N01 = get_search_count(N01_query)

    # Calculate the WebJaccard similarity
    similarity = N11 / (N11 + N10 + N01) if (N11 + N10 + N01) > 0 else 0
    return similarity


# Example usage
P = "apple"
Q = "fruit"
similarity_score = web_jaccard_similarity(P, Q)
print(f"WebJaccard Similarity between '{P}' and '{Q}': {similarity_score}")
