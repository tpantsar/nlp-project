# nlp-project

Project work for Natural Language Processing and Text Mining course 2024.

This project aims to investigate the similarity between two phrases using internet search results only.

WebJaccard similarity is a measure used in natural language processing (NLP) to quantify the similarity between two sets of web search results.
It is based on the Jaccard similarity coefficient, which is a statistical measure of the similarity between two sets.
The WebJaccard similarity specifically leverages web search engines to determine the overlap between the sets of web pages returned for two different queries.

## Add these variables to the .env file:

```
API_KEY=your_google_api_key (The API key for the Google Custom Search API)
CX=your_google_cx (The identifier of the Programmable Search Engine)
```

## Install dependencies:

```
conda env create --file environment.yml
conda activate nlp-project
conda env update --file environment.yml

Git Bash:
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

PowerShell (Administrator):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

datasets/ directory contains three standard human judgments datasets used in the project to calculate the similarity between two phrases.<br>
Pearson correlation coefficient is used to evaluate the similarity between WebJaccard similarity and human similarity judgments.<br>

<strong>The following datasets have human judgments similarity scores ranging between 0 and 10</strong>:

- MC (Miller and Charles, 91) - 30 pairs of terms
- RG (Rubenstein and Goodenougth, 1965) - 65 pairs of terms
- WordSim353 (Finkelstein et al., 2001) - 353 pairs of terms

## To directly compare WebJaccard with other measures:

- Ensure each similarity score is normalized to fall between 0 and 1.
- Use specific normalizations for cosine and other embedding-based similarities.
- Check if path-based measures like Wu-Palmer are already within [0, 1], and rescale if they’re not.

## References:

https://github.com/alexanderpanchenko/sim-eval

### Google Custom Search API

https://console.cloud.google.com/apis/api/customsearch.googleapis.com

https://console.cloud.google.com/apis/api/customsearch.googleapis.com/quotas?project=nlp-project-34673

### Create a Custom Search Engine

https://programmablesearchengine.google.com/controlpanel/create
