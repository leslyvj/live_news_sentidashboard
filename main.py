# main.py
import streamlit as st
import pandas as pd
import requests
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# --- Setup ---

# For Streamlit Cloud, put your API key in .streamlit/secrets.toml and use: st.secrets["API_KEY"]
API_KEY = st.secrets["API_KEY"] if "API_KEY" in st.secrets else "your_api_key_here"
BASE_URL = 'https://newsdata.io/api/1/latest'

@st.cache_resource
def get_spark_session():
    return SparkSession.builder.appName("NewsSentimentApp").getOrCreate()

spark = get_spark_session()

# --- Functions ---

def fetch_news(query="a", max_pages=2):
    """Fetch articles from NewsData.io API. Returns list of dicts."""
    all_articles = []
    params = {
        'apikey': API_KEY,
        'language': 'en',
        'q': query
    }
    page_token = None
    pages_fetched = 0

    while True:
        if page_token:
            params['page'] = page_token

        response = requests.get(BASE_URL, params=params)
        if response.status_code != 200:
            st.warning(f"API Error: {response.status_code}")
            break

        data = response.json()
        articles = data.get('results', [])
        all_articles.extend(articles)
        pages_fetched += 1

        page_token = data.get('nextPage', None)
        if not page_token or pages_fetched >= max_pages:
            break
        time.sleep(1)  # Respect API rate limits
    return all_articles

# VADER setup once for use in UDF
analyzer = SentimentIntensityAnalyzer()
def get_sentiment(text):
    try:
        score = analyzer.polarity_scores(str(text))['compound']
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    except Exception:
        return 'Neutral'

spark_sentiment_udf = udf(get_sentiment, StringType())

# --- Streamlit UI ---

st.title("📰 Live News Sentiment Dashboard (PySpark Edition)")
st.write(
    """Fetch the latest news from NewsData.io, analyze sentiment distributed via PySpark, 
    and visualize the results live!"""
)

max_pages = st.slider("Number of pages to fetch", min_value=1, max_value=10, value=2, step=1)
query = st.text_input("Keyword (required, e.g. 'a', 'AI', 'finance')", value="a")

if st.button('Update News Now'):
    with st.spinner('Fetching news and analyzing...'):
        articles = fetch_news(query=query, max_pages=max_pages)
        if articles:
            sdf = spark.createDataFrame(articles)
            # Use fillna to avoid NoneType issues
            from pyspark.sql.functions import coalesce, lit
            sdf = sdf.withColumn(
                "text",
                coalesce(sdf.title, lit("")) + lit(". ") + coalesce(sdf.description, lit(""))
            )
            sdf = sdf.withColumn("sentiment", spark_sentiment_udf(sdf.text))
            news_pd = sdf.select("title", "sentiment").toPandas()
            st.success(f"Fetched and analyzed {len(news_pd)} articles.")
            st.dataframe(news_pd.head(20), use_container_width=True)
            st.bar_chart(news_pd['sentiment'].value_counts())
        else:
            st.warning("No articles found or API limit reached.")
else:
    st.info("Set your parameters and click the button to fetch and analyze live news.")

st.write(
    "Tip: On Streamlit Cloud, put your NewsData.io API key in `.streamlit/secrets.toml` as `API_KEY = \"...\"`"
)
