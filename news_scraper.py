import requests
from bs4 import BeautifulSoup
import nltk
from dateutil import parser as date_parser
import sqlite3
import logging
import json
from tqdm import tqdm
from datetime import datetime, timedelta
import hashlib
import os
import feedparser
from dotenv import load_dotenv
import time
import google.generativeai as genai  # Add this import at the top of the file

# Load environment variables first
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# NLTK setup
nltk.download('punkt', quiet=True)

# Get API keys from environment variables
API_KEY = os.getenv('GOOGLE_API_KEY')
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

US_KEYWORDS = ["Trump", "USA", "United States", "Congress", "Washington", "Pentagon"]
CONFLICT_KEYWORDS = [
    "war", "conflict", "attack", "military", "fighting", "battle", "troops",
    "violence", "killed", "casualties", "missile", "strike", "combat",
    "rebellion", "uprising", "insurgency", "crisis", "protest"
]

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
    """Fetch the full content of an article from its URL"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Use BeautifulSoup to parse the HTML and extract the article content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Example: Extract content from a specific tag or class
        content = soup.find_all('p')  # Adjust this to match the NYT article structure
        article_text = ' '.join([p.get_text() for p in content])
        
        return article_text.strip()
    except requests.exceptions.HTTPError as e:
        logging.warning(f"Failed to process content for {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error fetching article content: {e}")
        return None

def truncate_text(text, max_tokens=500):
    """Truncate text to a safe length for summarization"""
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens[:max_tokens])

def summarize_with_google_api(text):
    """Summarize text using Google's Generative AI"""
    try:
        truncated_text = truncate_text(text)
        
        # Configure the Generative AI library
        genai.configure(api_key=API_KEY)
        
        # Create a GenerativeModel instance
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate the summary
        prompt = f"Summarize this article in two to three concise and complete sentences. Focus on the key events, their global implications, and the perspectives of involved parties in the style of the Council on Foreign Relations:\n\n{truncated_text}"
        
        response = model.generate_content(prompt)
        
        # Get the generated text
        if response.text:
            summary = response.text.strip()
            if not summary.endswith('.'):
                summary += "."
            return summary
            
    except Exception as e:
        logging.warning(f"Summarization with Google API failed: {e}")
        
        # Fallback to sentence-based summary
        sentences = nltk.sent_tokenize(text)
        if sentences:
            return ' '.join(sentences[:2])  # Return first two sentences
        return text[:300]  # Fallback to first 300 characters only if sentence tokenization fails

def format_date(date_str):
    """Format date as 'Month Day, Year'"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S").strftime("%B %d, %Y")
    except Exception:
        return datetime.now().strftime("%B %d, %Y")

def filter_articles(articles):
    """Filter out U.S.-centric content, non-conflict related content, and duplicates"""
    filtered = []
    seen_hashes = set()
    for article in articles:
        # Skip U.S.-centric articles
        if any(keyword.lower() in article['title'].lower() for keyword in US_KEYWORDS):
            logging.info(f"Excluding U.S.-centric article: {article['title']}")
            continue
        
        # Check for conflict-related keywords in both title and summary
        title_lower = article['title'].lower()
        summary_lower = article.get('summary', '').lower()
        matched_keywords = [keyword for keyword in CONFLICT_KEYWORDS if keyword.lower() in title_lower or keyword.lower() in summary_lower]
        
        if not matched_keywords:
            logging.info(f"Excluding non-conflict article: {article['title']}")
            logging.debug(f"Title: {title_lower}, Summary: {summary_lower}")
            continue
        else:
            logging.debug(f"Article '{article['title']}' matched keywords: {matched_keywords}")
        
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
    cutoff_date = datetime.now() - timedelta(days=3)
    
    for entry in feed.entries:
        # Parse the published date
        try:
            pub_date = datetime(*entry.published_parsed[:6])
            if pub_date < cutoff_date:
                continue
        except (AttributeError, TypeError):
            # If we can't parse the date, skip this check
            pass
            
        # Check if the article is relevant based on conflict keywords
        if any(keyword.lower() in entry.title.lower() for keyword in CONFLICT_KEYWORDS):
            articles.append({
                'title': entry.title,
                'link': entry.link,
                'source': 'BBC',
                'pub_date': entry.get('published', datetime.now().strftime("%Y-%m-%d"))
            })
        # Limit to 5 relevant articles
        if len(articles) >= 5:
            break
    return articles

def process_nyt_article(item):
    """Process a single NYT article"""
    try:
        title = item.get('title', '')
        abstract = item.get('abstract', '')
        url = item.get('url', '')
        pub_date = item.get('published_date', '')
        
        # Skip if missing essential fields
        if not all([title, abstract, url, pub_date]):
            logging.warning(f"Missing required fields for NYT article: {title}")
            return None
            
        # Format the date
        try:
            pub_date = datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%S%z")
            pub_date = pub_date.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            pub_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        
        return {
            'title': title,
            'text': abstract,
            'link': url,
            'date': pub_date,
            'source': 'NYT'
        }
    except Exception as e:
        logging.error(f"Error processing NYT article: {e}")
        return None

def scrape_nyt_news():
    """Fetch NYT World News articles using Top Stories API"""
    try:
        nyt_api_key = os.getenv('NYT_API_KEY')
        if not nyt_api_key:
            logging.warning("NYT_API_KEY environment variable is not set")
            return []

        url = f"https://api.nytimes.com/svc/topstories/v2/world.json?api-key={nyt_api_key}"
        response = requests.get(url)
        response.raise_for_status()
        
        # Log the raw response for debugging
        logging.debug(f"NYT API Response: {response.text[:500]}...")
        
        data = response.json()
        if 'results' not in data:
            logging.warning("No 'results' found in NYT API response")
            return []
        
        articles = []
        cutoff_date = datetime.now() - timedelta(days=3)
        
        for item in data.get('results', []):
            try:
                # Parse the published date
                pub_date = datetime.strptime(item.get('published_date', ''), "%Y-%m-%dT%H:%M:%S%z")
                if pub_date.replace(tzinfo=None) < cutoff_date:
                    continue
                
                # Check if article is conflict-related
                title = item.get('title', '').lower()
                if any(keyword.lower() in title for keyword in CONFLICT_KEYWORDS):
                    processed_article = process_nyt_article(item)
                    if processed_article:
                        articles.append(processed_article)
                        logging.info(f"Added NYT article: {processed_article['title']}")
                
                # Limit to 5 relevant articles
                if len(articles) >= 5:
                    break
                    
            except (KeyError, ValueError) as e:
                logging.warning(f"Error processing NYT article: {e}")
                continue
                
        logging.info(f"Found {len(articles)} relevant NYT articles")
        return articles
        
    except Exception as e:
        logging.error(f"Failed to fetch NYT articles: {e}")
        return []

def process_articles(articles, conn, max_articles=5):
    """Process articles scraped from websites"""
    snippets = []
    for article in articles:
        try:
            logging.info(f"Processing article: {article['title']}")
            article_text = get_article_content(article['link'])
            
            if not article_text:
                logging.warning(f"No content fetched for article: {article['title']}")
                continue
            
            # Log the length of the article text
            logging.debug(f"Article text length: {len(article_text.split())} words")
            
            # Adjust the length check if necessary
            if len(article_text.split()) < 30:  # Adjusted from 50 to 30 words
                logging.warning(f"Article too short, skipping: {article['title']}")
                continue
                
            date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            date_str = format_date(date)
            
            # Only add articles that have both title and content
            if article['title'] and article_text:
                cache_article(conn, article['source'], article['title'], article_text, article['link'], date)
                summary = summarize_with_google_api(article_text)
                if summary:  # Only add if we got a summary
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
        
        # Process AP articles
        ap_articles = scrape_ap_news()
        all_snippets.extend(process_articles(ap_articles, conn))
        
        # Process BBC articles
        bbc_articles = scrape_bbc_news()
        all_snippets.extend(process_articles(bbc_articles, conn))
        
        # Process NYT articles with additional logging
        nyt_articles = scrape_nyt_news()
        if nyt_articles:
            logging.info(f"Processing {len(nyt_articles)} NYT articles")
            all_snippets.extend(process_articles(nyt_articles, conn))
        else:
            logging.warning("No NYT articles to process")
        
        save_snippets(output_file, all_snippets)
        logging.info(f"Snippets saved to {output_file}")
    finally:
        conn.close()

def save_snippets(output_file, snippets):
    """Save snippets to JSON file, keeping only recent and limited number of articles"""
    try:
        # Start fresh each time - don't load existing articles
        unique_snippets = []
        seen_titles = set()
        cutoff_date = datetime.now() - timedelta(days=3)
        
        # Only keep new snippets that have actual content and are recent
        for snippet in snippets:
            if snippet.get('summary') and snippet['title'] not in seen_titles:
                try:
                    article_date = datetime.strptime(snippet['date_str'], "%B %d, %Y")
                    if article_date >= cutoff_date:
                        seen_titles.add(snippet['title'])
                        unique_snippets.append(snippet)
                except (ValueError, KeyError):
                    # If we can't parse the date, skip this article
                    continue
        
        # Keep only the most recent 10 articles
        unique_snippets = unique_snippets[-10:]
        
        # Save directly to file (no merging with existing)
        with open(output_file, "w") as f:
            json.dump(unique_snippets, f, indent=2)
        logging.info(f"Saved {len(unique_snippets)} articles to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save snippets: {e}")
        raise

if __name__ == "__main__":
    try:
        logging.info("Monitoring news sources...")
        monitor_news_sources("conflict_news.json")
    except Exception as e:
        logging.error(f"Failed to run news monitoring: {e}")
