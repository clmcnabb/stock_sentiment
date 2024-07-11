"""
NLP Stock Sentiment Analysis Module
This module provides functionality to analyze sentiment towards a given stock symbol based on text from a URL.
"""

import requests
import spacy
import yfinance as yf
from bs4 import BeautifulSoup
from transformers import pipeline


def _fetch_text_from_url(url):
    """Fetch and return the text content from a given URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text()
    return text


def _get_stock_name(symbol):
    """Retrieve the full name of a stock based on its symbol using Yahoo Finance."""
    stock = yf.Ticker(symbol)
    return stock.info["longName"]


class StockSentimentAnalyzer:
    """Class to analyze stock sentiment from text."""

    def __init__(self, model, max_length, nlp):
        self.max_length = max_length
        self.nlp = nlp
        self.sentiment_pipe = pipeline("sentiment-analysis", model=model)

    def analyze_stock_sentiment(self, url, stock_symbol):
        """Extract sentiment about a given stock symbol from a given URL."""
        text = _fetch_text_from_url(url)
        stock_long_name = _get_stock_name(stock_symbol)
        # Removing common suffixes/articles from the stock name
        stock_long_name_truncated = (
            stock_long_name.replace("The", "")
            .replace("Inc.", "")
            .replace("Corporation", "")
            .strip()
        )
        stock_short_name = (
            stock_long_name_truncated.split(" ")[0].split(",")[0].split(".")[0]
        )
        entities = [
            stock_long_name,
            stock_long_name_truncated,
            stock_short_name,
            stock_symbol,
        ]
        entities = [entity.lower() for entity in entities]

        text = text.lower()

        doc = self.nlp(text)

        text_snippets = [
            sent.text
            for sent in doc.sents
            if any(entity in sent.text for entity in entities)
        ]

        pos_count = 0
        neg_count = 0
        neu_count = 0
        sentiments = []

        for snippet in text_snippets:
            if len(snippet) > self.max_length:
                snippet = snippet[: self.max_length]
            results = self.sentiment_pipe(snippet)
            sentiment = results[0]["label"]
            sentiments.append(sentiment)
            if sentiment == "positive":
                pos_count += len(snippet)
            elif sentiment == "negative":
                neg_count += len(snippet)
            elif sentiment == "neutral":
                neu_count += len(snippet)

        total_count = pos_count + neg_count + neu_count

        sentiment_score = (
            (pos_count / total_count) - (neg_count / total_count)
            if total_count > 0
            else 0
        )

        return sentiment_score


if __name__ == "__main__":
    # Example usage

    # Constants
    MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    MAX_LENGTH = 512

    # Initialize NLP tools
    nlp = spacy.load("en_core_web_sm")

    analyzer = StockSentimentAnalyzer(MODEL, MAX_LENGTH, nlp)
    url = "https://finance.yahoo.com/news/big-techs-next-milestone-4-185011643.html"
    stock_symbol = "NVDA"
    sentiments = analyzer.analyze_stock_sentiment(url, stock_symbol)
    print(sentiments)
