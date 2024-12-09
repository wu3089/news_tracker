import requests
from bs4 import BeautifulSoup
import nltk
from dateutil import parser as date_parser
import sqlite3
import logging
import json
from tqdm import tqdm
from datetime import datetime
import hashlib
import os
import feedparser
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# NLTK setup
nltk.download('punkt', quiet=True)

# Get API key from environment variable
API_KEY = os.getenv('GOOGLE_API_KEY')
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

US_KEYWORDS = ["Trump", "USA", "United States", "Congress", "Washington", "Pentagon"]

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
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text()]
        content = ' '.join(paragraphs).strip()
        logging.debug(f"Fetched content for {url}: {content[:200]}...")
        return content
    except Exception as e:
        logging.warning(f"Failed to process content for {url}: {e}")
        return None

def truncate_text(text, max_tokens=500):
    """Truncate text to a safe length for summarization"""
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens[:max_tokens])

def summarize_with_google_api(text):
    """Summarize text using Google API with complete sentences"""
    try:
        truncated_text = truncate_text(text)
        url = f"https://generativeai.googleapis.com/v1beta2/models/text-bison-001:generateText?key={API_KEY}"
        payload = {
            "prompt": f"Summarize this article in two to three concise and complete sentences. Focus on the key events, their global implications, and the perspectives of involved parties:\n\n{truncated_text}",
            "temperature": 0.7,
            "maxOutputTokens": 256
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        summary = result.get("candidates", [{}])[0].get("output", "")
        if summary and not summary.endswith('.'):
            summary += "."
        return summary.strip()
    except Exception as e:
        logging.warning(f"Summarization with Google API failed: {e}")
        return text[:300]  # Fallback to first 300 characters

def format_date(date_str):
    """Format date as 'Month Day, Year'"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S").strftime("%B %d, %Y")
    except Exception:
        return datetime.now().strftime("%B %d, %Y")

def filter_articles(articles):
    """Filter out U.S.-centric content and duplicates"""
    filtered = []
    seen_hashes = set()
    for article in articles:
        if any(keyword.lower() in article['title'].lower() for keyword in US_KEYWORDS):
            logging.info(f"Excluding U.S.-centric article: {article['title']}")
            continue
        article_hash = hashlib.md5((article['title'] + article['link']).encode()).hexdigest()
        if article_hash in seen_hashes:
            logging.info(f"Excluding duplicate article: {article['title']}")
            continue
        seen_hashes.add(article_hash)
        filtered.append(article)
    return filtered

def scrape_ap_news():
    """Scrape AP World News articles"""
    url = "https://apnews.com/world-news"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = []
    for article in soup.select('a[href^="/article"]'):
        title = article.get_text(strip=True)
        link = "https://apnews.com" + article['href']
        articles.append({'title': title, 'link': link, 'source': 'AP'})
    return articles

def scrape_bbc_news():
    """Scrape BBC World News RSS feed"""
    url = "http://feeds.bbci.co.uk/news/world/rss.xml"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:5]:  # Limit to the top 5 entries
        articles.append({
            'title': entry.title,
            'link': entry.link,
            'source': 'BBC'
        })
    return articles

def scrape_nyt_news():
    """Scrape New York Times World RSS feed"""
    url = "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:5]:  # Limit to the top 5 entries
        articles.append({
            'title': entry.title,
            'link': entry.link,
            'source': 'NYT'
        })
    return articles

def process_articles(articles, conn):
    """Process articles scraped from websites"""
    articles = filter_articles(articles)
    snippets = []
    for article in articles:
        try:
            logging.info(f"Processing article: {article['title']}")
            article_text = get_article_content(article['link'])
            if not article_text:
                logging.warning(f"No content fetched for article: {article['title']}")
                continue
            date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            date_str = format_date(date)
            cache_article(conn, article['source'], article['title'], article_text, article['link'], date)
            summary = summarize_with_google_api(article_text)
            snippets.append({
                'title': article['title'],
                'date_str': date_str,
                'summary': summary,
                'source': article['source'],
                'link': article['link']
            })
        except Exception as e:
            logging.error(f"Failed to process article: {e}")
            continue
    return snippets

def monitor_news_sources(output_file):
    """Monitor news sources (BBC, AP, NYT) and save processed snippets"""
    conn = setup_article_cache()
    try:
        all_snippets = []
        ap_articles = scrape_ap_news()
        bbc_articles = scrape_bbc_news()
        nyt_articles = scrape_nyt_news()
        all_snippets.extend(process_articles(ap_articles, conn))
        all_snippets.extend(process_articles(bbc_articles, conn))
        all_snippets.extend(process_articles(nyt_articles, conn))
        save_snippets(output_file, all_snippets)
        logging.info(f"Snippets saved to {output_file}")
    finally:
        conn.close()

def save_snippets(output_file, snippets):
    """Save snippets to JSON file"""
    try:
        temp_file = output_file + ".tmp"
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                existing_data = json.load(f)
            snippets = existing_data + snippets
        with open(temp_file, "w") as f:
            json.dump(snippets, f, indent=2)
        os.replace(temp_file, output_file)
        logging.info(f"Saved {len(snippets)} articles to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save snippets: {e}")
        raise

if __name__ == "__main__":
    try:
        logging.info("Monitoring news sources...")
        load_dotenv()
        monitor_news_sources("conflict_news.json")
    except Exception as e:
        logging.error(f"Failed to run news monitoring: {e}")
