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
