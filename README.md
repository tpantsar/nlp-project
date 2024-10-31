# nlp-project

Project work for Natural Language Processing and Text Mining course 2024.

This project aims to investigate the similarity between two phrases using internet search results only.

## Add these variables to the .env file:

```
API_KEY=your_google_api_key
CX=your_google_cx
```

## Install dependencies:

```
conda env create --file environment.yml
conda activate nlp-project

conda env update --file environment.yml

python -m venv .venv
source .venv/bin/activate
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
