# import numpy as np
import requests
import spacy
import yfinance as yf
from bs4 import BeautifulSoup
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")
model = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
sentiment_pipe = pipeline("sentiment-analysis", model=model)
MAX_LENGTH = 512

def fetch_text_from_url(url):
    """ Fetch and return the text content from a given URL. """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    return text

def get_stock_name(symbol):
    """ Retrieve the full name of a stock based on its symbol using Yahoo Finance. """
    stock = yf.Ticker(symbol)
    return stock.info['longName']

def analyze_stock_sentiment(url, stock_symbol):
    """Extract sentiment about a given stock symbol from a given URL."""
    text = fetch_text_from_url(url)
    stock_long_name = get_stock_name(stock_symbol)

    # Removeing common suffixes/articles from the stock name
    stock_long_name_truncated = stock_long_name.replace("The", "").strip()
    stock_long_name_truncated = stock_long_name_truncated.replace("Inc.", "").strip()
    stock_long_name_truncated = stock_long_name_truncated.replace(
        "Corporation", ""
    ).strip()
    stock_short_name = stock_long_name_truncated.split(" ")[0].split(",")[0]
    stock_base_name = stock_short_name.split(".")[0]
    entities = [
        stock_long_name,
        stock_long_name_truncated,
        stock_short_name,
        stock_base_name,
        stock_symbol,
    ]
    entities = [entity.lower() for entity in entities]
    text = text.lower()

    doc = nlp(text)

    text_snippets = [
        sent.text
        for sent in doc.sents
        if any(entity in sent.text for entity in entities)
    ]
    # for sent in doc.sents:
    #     snippet = sent.text[:MAX_LENGTH]
    #     results = sentiment_pipe(snippet)
    #     print(results)
    pos_count = []
    neg_count = []
    scores = []
    sentiments = []
    for snippet in text_snippets:
        if len(snippet) > MAX_LENGTH:
            snippet = snippet[:MAX_LENGTH]
        results = sentiment_pipe(snippet)
        # print(results)
        sentiments.append(results[0]["label"])
        if results[0]["label"] == "positive":
            pos_count.append(len(snippet))
            # print(f"Positive snippet length: {len(snippet)}")
        elif results[0]["label"] == "negative":
            neg_count.append(len(snippet))
            # print(f"Negative snippet length: {len(snippet)}")
        scores.append(results[0]["score"])
    # scores = np.array(scores)
    # print(pos_count)
    # print(neg_count)

    pos_sum = sum(pos_count)
    neg_sum = sum(neg_count)
    # print(pos_sum)
    # print(neg_sum)
    sentiment_score = sum(
        [
            1 * (pos_sum / (pos_sum + neg_sum))
            if sentiment == "positive"
            else -1 * (neg_sum / (pos_sum + neg_sum))
            if sentiment == "negative"
            else 0
            for (score, sentiment) in zip(scores, sentiments)
        ]
    )  # / len(sentiments)

    # print(sentiment_score / (len(pos_count) + len(neg_count)))
    sentiment_score = float(sentiment_score / (len(pos_count) + len(neg_count)))
    return sentiment_score

# Example usage
url = 'https://finance.yahoo.com/news/salesforce-gives-weak-sales-outlook-201103268.html'  # Replace with actual URL
stock_symbol = 'CRM'
sentiment_score = analyze_stock_sentiment(url, stock_symbol)
print(f'Sentiment Score for {stock_symbol}: {sentiment_score:.2f}')