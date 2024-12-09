import requests
from bs4 import BeautifulSoup
import feedparser
import nltk
from transformers import pipeline
from dateutil import parser as date_parser
import sqlite3
import logging
import json
from tqdm import tqdm
from datetime import datetime, timedelta
import hashlib
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# NLTK setup
nltk.download('punkt', quiet=True)

def initialize_summarizer():
    """Initialize the Hugging Face summarizer pipeline"""
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception as e:
        logging.error(f"Failed to initialize summarizer: {e}")
        raise

def setup_article_cache():
    """Initialize SQLite database for caching articles"""
    db_path = "article_cache.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS articles
                 (id TEXT PRIMARY KEY, source TEXT, title TEXT, 
                  text TEXT, url TEXT, date TEXT)''')
    conn.commit()
    return conn

def cache_article(conn, source, title, text, url, date):
    """Cache an article to avoid reprocessing"""
    article_id = hashlib.md5((title + url).encode()).hexdigest()
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO articles
                 (id, source, title, text, url, date)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (article_id, source, title, text, url, date))
    conn.commit()

def get_article_content(url):
    """Fetch and clean article content from a URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text()]
        return ' '.join(paragraphs).strip()
    except requests.exceptions.RequestException as e:
        logging.warning(f"Request failed for {url}: {e}")
        return None
    except Exception as e:
        logging.warning(f"Failed to process content for {url}: {e}")
        return None

def truncate_text(text, max_tokens=300):
    """Truncate text to a safe length for summarization"""
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens[:max_tokens])

def summarize_article(text, summarizer):
    """Summarize article using NLP"""
    try:
        truncated_text = truncate_text(text)
        summary = summarizer(truncated_text, max_length=100, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        logging.warning(f"Summarization failed: {e}")
        return text[:300]  # Fallback to first 300 characters

def parse_date(date_str):
    """Parse and normalize dates"""
    try:
        return date_parser.parse(date_str).isoformat()
    except Exception:
        return datetime.now().isoformat()

def process_feed(feed_url, source, conn, summarizer):
    """Process a single RSS feed and return processed snippets"""
    feed = feedparser.parse(feed_url)
    if not feed.entries:
        logging.warning(f"No entries found for feed: {source}")
        return []

    snippets = []
    for entry in feed.entries[:1]:  # Process only 1 entry per source for efficiency
        try:
            article_text = get_article_content(entry.link)
            if not article_text:
                continue
            title = entry.title.strip()
            date = parse_date(entry.get('published', ''))
            cache_article(conn, source, title, article_text, entry.link, date)
            summary = summarize_article(article_text, summarizer)
            snippets.append({
                'title': title,
                'date': date,
                'summary': summary,
                'source': source,
                'link': entry.link
            })
        except Exception as e:
            logging.error(f"Failed to process entry from {source}: {e}")
            continue
    return snippets

def save_snippets(output_file, snippets):
    """Save snippets to JSON file"""
    try:
        if os.path.exists(output_file):
            with open(output_file, "r+") as f:
                existing_data = json.load(f)
                f.seek(0)
                f.write(json.dumps(existing_data + snippets, indent=2))
        else:
            with open(output_file, "w") as f:
                json.dump(snippets, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save snippets to file: {e}")

def monitor_rss_feeds(rss_feeds, output_file, summarizer):
    """Monitor RSS feeds and save processed snippets"""
    conn = setup_article_cache()
    try:
        all_snippets = []
        for source, feed_url in tqdm(rss_feeds.items(), desc="Processing feeds"):
            snippets = process_feed(feed_url, source, conn, summarizer)
            all_snippets.extend(snippets)
        save_snippets(output_file, all_snippets)
        logging.info(f"Snippets saved to {output_file}")
    finally:
        conn.close()

if __name__ == "__main__":
    try:
        logging.info("Initializing summarizer pipeline...")
        summarizer = initialize_summarizer()
        logging.info("Summarizer initialized.")

        rss_feeds = {
            "BBC": "http://feeds.bbci.co.uk/news/world/rss.xml",
            "Reuters": "https://openrss.org/www.reuters.com/world",
            "AP": "https://openrss.org/apnews.com/apf-intl"
        }
        monitor_rss_feeds(rss_feeds, "conflict_news.json", summarizer)
    except Exception as e:
        logging.error(f"Failed to run RSS feed monitoring: {e}")
