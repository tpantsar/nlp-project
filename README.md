# nlp-project

Project work for Natural Language Processing and Text Mining course 2024.

This project aims to investigate the similarity between two phrases using internet search results only.

## Google Custom Search API

https://console.cloud.google.com/apis/api/customsearch.googleapis.com

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

datasets/ directory contains the corpus datasets used in the project to calculate the similarity between two phrases.<br>
Pearson correlation coefficient is used to evaluate the similarity between WebJaccard similarity and human similarity judgments.<br>
The datasets are:

- MC (Miller and Charles, 91)
- RG (Rubenstein and Goodenougth, 1965)
- WordSim353 (Finkelstein et al., 2001)
