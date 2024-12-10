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
def setup_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        # Create NLTK data directory if it doesn't exist
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
    except Exception as e:
        logging.warning(f"NLTK setup warning: {e}")

# Call setup at start
setup_nltk()

# Get API keys from environment variables
API_KEY = os.getenv('GOOGLE_API_KEY')
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

US_KEYWORDS = [
    "Trump",
    "Donald Trump",
    "Trump Administration",
    "Congress",
    "Washington",
    "Trump's",
    "Crypto",
    "cryptocurrency",
    "US policy",
    "American",
    "United States",
    "White House"
]
CONFLICT_KEYWORDS = [
    "war",
    "conflict", 
    "attack",
    "military",
    "fighting",
    "battle",
    "troops",
    "violence",
    "killed",
    "casualties", 
    "missile",
    "strike",
    "combat",
    "rebellion",
    "uprising",
    "insurgency", 
    "crisis",
    "protest"
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
    """Get article content with proper headers"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logging.warning(f"Failed to process content for {url}: {e}")
        return None

def truncate_text(text, max_tokens=500):
    """Truncate text to a safe length for summarization"""
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens[:max_tokens])

def summarize_with_google_api(text):
    """Summarize text using Google's Generative AI with error handling"""
    try:
        truncated_text = truncate_text(text)
        
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""Summarize this news article in 2-3 sentences, focusing on key facts and events only.
        Article text: {truncated_text}"""
        
        response = model.generate_content(prompt, safety_settings={
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE"
        })
        
        if response and response.text:
            return response.text.strip()
            
    except Exception as e:
        logging.warning(f"Summarization failed: {e}")
        return None

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
    
    # Specific article to remove (as a temporary fix)
    banana_article_url = "https://www.bbc.com/news/articles/c4gzz5wdg41o"
    
    for article in articles:
        # Skip the specific article we want to remove
        if article['link'] == banana_article_url:
            logging.info(f"Excluding specific article: {article['title']}")
            continue
            
        title_lower = article['title'].lower()
        summary_lower = article.get('summary', '').lower()
        content = title_lower + " " + summary_lower
        
        # Count how many US keywords match
        us_matches = [keyword.lower() for keyword in US_KEYWORDS if keyword.lower() in content]
        
        # If we find any US keyword, filter it out
        if us_matches:
            logging.info(f"Excluding U.S.-centric article: {article['title']}")
            logging.debug(f"Matched US keywords: {us_matches}")
            continue
            
        # Rest of the filtering logic...
        matched_keywords = [keyword for keyword in CONFLICT_KEYWORDS if keyword.lower() in content]
        if not matched_keywords:
            logging.info(f"Excluding non-conflict article: {article['title']}")
            continue
        
        article_hash = hashlib.md5((article['title'] + article['link']).encode()).hexdigest()
        if article_hash in seen_hashes:
            continue
        
        seen_hashes.add(article_hash)
        filtered.append(article)
    
    return filtered

def paraphrase_title(original_title):
    """Paraphrase article title using Gemini"""
    try:
        # Configure the Generative AI library
        genai.configure(api_key=API_KEY)
        
        # Create a GenerativeModel instance
        model = genai.GenerativeModel('gemini-pro')
        
        title_prompt = f"""Rewrite this news headline following these rules:
        1. Use title case (capitalize all words except articles, conjunctions, and prepositions under 4 letters)
        2. Make it a complete sentence (no ellipsis or trailing dots)
        3. Keep it concise but informative
        4. Maintain the key information but rephrase it
        
        Original headline: {original_title}"""
        
        response = model.generate_content(title_prompt)
        return response.text.strip().rstrip('.')
    except Exception as e:
        logging.error(f"Error paraphrasing title: {e}")
        return original_title  # Fallback to original if paraphrasing fails

def process_nyt_article(item):
    """Process a single NYT article"""
    try:
        original_title = item.get('title', '')
        abstract = item.get('abstract', '')
        url = item.get('url', '')
        pub_date = item.get('published_date', '')
        
        if not all([original_title, abstract, url, pub_date]):
            logging.warning(f"Missing required fields for NYT article: {original_title}")
            return None
            
        try:
            pub_date = datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%S%z")
            pub_date = pub_date.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            pub_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        
        title = paraphrase_title(original_title)
        summary = summarize_with_google_api(abstract)
        
        return {
            'title': title,
            'text': abstract,
            'link': url,
            'date': pub_date,
            'source': 'NYT',
            'summary': summary
        }
    except Exception as e:
        logging.error(f"Error processing NYT article: {e}")
        return None

def scrape_nyt_news():
    """Fetch NYT World News articles using Top Stories API"""
    try:
        nyt_api_key = os.getenv('NYT_API_KEY')
        if not nyt_api_key:
            logging.error("NYT_API_KEY environment variable is not set")  # Changed to error level
            return []

        url = f"https://api.nytimes.com/svc/topstories/v2/world.json?api-key={nyt_api_key}"
        logging.info(f"Fetching NYT articles from: {url}")  # Log the URL (without API key)
        
        response = requests.get(url)
        
        # Log the response status and headers
        logging.info(f"NYT API Response Status: {response.status_code}")
        logging.debug(f"NYT API Response Headers: {response.headers}")
        
        # Raise for status before processing
        response.raise_for_status()
        
        data = response.json()
        if 'results' not in data:
            logging.error(f"Unexpected API response structure. Keys found: {data.keys()}")
            if 'fault' in data:
                logging.error(f"API error: {data['fault']}")
            return []
        
        articles = []
        cutoff_date = datetime.now() - timedelta(days=3)
        
        logging.info(f"Found {len(data['results'])} total NYT articles")
        
        for item in data.get('results', []):
            try:
                # Parse the published date
                pub_date = datetime.strptime(item.get('published_date', ''), "%Y-%m-%dT%H:%M:%S%z")
                if pub_date.replace(tzinfo=None) < cutoff_date:
                    logging.debug(f"Skipping old article: {item.get('title', 'No title')} ({pub_date})")
                    continue
                
                # Check if article is conflict-related
                title = item.get('title', '').lower()
                if any(keyword.lower() in title for keyword in CONFLICT_KEYWORDS):
                    processed_article = process_nyt_article(item)
                    if processed_article:
                        articles.append(processed_article)
                        logging.info(f"Added NYT article: {processed_article['title']}")
                else:
                    logging.debug(f"Skipping non-conflict article: {item.get('title', 'No title')}")
                
                # Limit to 5 relevant articles
                if len(articles) >= 5:
                    break
                    
            except (KeyError, ValueError) as e:
                logging.warning(f"Error processing NYT article: {e}")
                continue
                
        logging.info(f"Found {len(articles)} relevant NYT articles")
        return articles
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch NYT articles - Request error: {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse NYT response - JSON error: {e}")
        return []
    except Exception as e:
        logging.error(f"Failed to fetch NYT articles - Unexpected error: {e}")
        return []

def scrape_bbc_news():
    """Scrape BBC World News RSS feed"""
    logging.info("Starting BBC news scraping...")
    url = "http://feeds.bbci.co.uk/news/world/rss.xml"
    feed = feedparser.parse(url)
    articles = []
    cutoff_date = datetime.now() - timedelta(days=3)
    
    for entry in feed.entries:
        logging.debug(f"Processing BBC entry: {entry.title}")
            
        try:
            pub_date = datetime(*entry.published_parsed[:6])
            if pub_date < cutoff_date:
                continue
        except (AttributeError, TypeError):
            pass
            
        title_lower = entry.title.lower()
        if (any(keyword.lower() in title_lower for keyword in CONFLICT_KEYWORDS) and 
            not any(keyword.lower() in title_lower for keyword in US_KEYWORDS)):
            
            # Paraphrase the title
            paraphrased_title = paraphrase_title(entry.title)
            
            articles.append({
                'title': paraphrased_title,
                'link': entry.link,
                'source': 'BBC',
                'pub_date': entry.get('published', datetime.now().strftime("%Y-%m-%d"))
            })
            
        if len(articles) >= 5:
            break
    return articles

def process_articles(articles, conn, max_articles=5):
    """Process articles scraped from websites"""
    snippets = []
    for article in articles:
        try:
            logging.info(f"Processing article: {article['title']}")
            
            # Clean up title
            title = article['title'].strip()
            if title.endswith('.'):
                title = title[:-1]
            
            # Format source names consistently
            source_mapping = {
                'Associated Press': 'AP',
                'AP': 'AP',
                'BBC': 'BBC',
                'Reuters': 'Reuters',
                'The New York Times': 'NYT',
                'NYT': 'NYT'
            }
            
            # Get the abbreviated source name
            source = source_mapping.get(article['source'], article['source'])
            
            if article['source'] == 'NYT':
                date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                date_str = format_date(date)
                
                snippets.append({
                    'title': title,
                    'date_str': date_str,
                    'summary': article.get('summary', ''),
                    'source': source,
                    'link': article['link']
                })
                continue
            
            # For non-NYT articles
            article_text = get_article_content(article['link'])
            
            if not article_text:
                logging.warning(f"No content fetched for article: {title}")
                continue
            
            date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            date_str = format_date(date)
            
            if title and article_text:
                cache_article(conn, source, title, article_text, article['link'], date)
                summary = summarize_with_google_api(article_text)
                if summary:
                    snippets.append({
                        'title': title,
                        'date_str': date_str,
                        'summary': summary,
                        'source': source,
                        'link': article['link']
                    })
        except Exception as e:
            logging.error(f"Failed to process article: {e}")
            continue
    return snippets

def monitor_news_sources(output_file):
    """Monitor news sources and save processed snippets"""
    conn = setup_article_cache()
    try:
        all_snippets = []
        
        # Get Reuters/AP articles from NewsAPI
        newsapi_articles = get_news_from_newsapi()
        all_snippets.extend(process_articles(newsapi_articles, conn))
        
        # Process BBC articles
        bbc_articles = scrape_bbc_news()
        all_snippets.extend(process_articles(bbc_articles, conn))
        
        # Process NYT articles
        nyt_articles = scrape_nyt_news()
        if nyt_articles:
            logging.info(f"Processing {len(nyt_articles)} NYT articles")
            all_snippets.extend(process_articles(nyt_articles, conn))
        
        save_snippets(output_file, all_snippets)
        logging.info(f"Snippets saved to {output_file}")
    finally:
        conn.close()

def save_snippets(output_file, snippets):
    """Save snippets to JSON file, keeping only recent and limited number of articles"""
    try:
        unique_snippets = []
        seen_titles = set()
        cutoff_date = datetime.now() - timedelta(days=3)
        
        # Sort snippets by date (newest first)
        sorted_snippets = sorted(
            snippets,
            key=lambda x: datetime.strptime(x['date_str'], "%B %d, %Y"),
            reverse=True
        )
        
        for snippet in sorted_snippets:
            if snippet.get('summary') and snippet['title'] not in seen_titles:
                try:
                    article_date = datetime.strptime(snippet['date_str'], "%B %d, %Y")
                    if article_date >= cutoff_date:
                        seen_titles.add(snippet['title'])
                        # Ensure consistent formatting
                        snippet['title'] = snippet['title'].strip()
                        snippet['summary'] = snippet['summary'].strip()
                        unique_snippets.append(snippet)
                except (ValueError, KeyError):
                    continue
        
        # Keep only the most recent 10 articles
        unique_snippets = unique_snippets[:10]
        
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(unique_snippets, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved {len(unique_snippets)} articles to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save snippets: {e}")
        raise

def get_news_from_newsapi():
    """Get news from NewsAPI.org focusing on Reuters and AP world news"""
    logging.info("Fetching from NewsAPI...")
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        logging.warning("No NewsAPI key found")
        return []
        
    url = "https://newsapi.org/v2/everything"
    
    # Specific query to target world news and conflicts
    params = {
        'apiKey': api_key,
        'domains': 'reuters.com,apnews.com',  # Specifically target these domains
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 100,  # Get more articles to filter through
        'q': '(world OR international OR global) AND (conflict OR war OR crisis OR tension)'  # Focus on world news and conflicts
    }
    
    try:
        logging.info(f"Making NewsAPI request with params: {params}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'ok':
            logging.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return []
            
        total_results = data.get('totalResults', 0)
        logging.info(f"Found {total_results} total articles from NewsAPI")
        
        articles = []
        for article in data.get('articles', []):
            title = article.get('title', '')
            source = article.get('source', {}).get('name', '')
            
            if not title or not source:
                continue
                
            # Only process Reuters and AP articles
            if source.lower() not in ['reuters', 'associated press', 'ap']:
                continue
                
            # Apply our usual filtering
            title_lower = title.lower()
            if (any(keyword.lower() in title_lower for keyword in CONFLICT_KEYWORDS) and 
                not any(keyword.lower() in title_lower for keyword in US_KEYWORDS)):
                articles.append({
                    'title': title,
                    'link': article.get('url'),
                    'source': source,
                    'pub_date': article.get('publishedAt', datetime.now().strftime("%Y-%m-%d"))
                })
                logging.info(f"Added {source} article: {title}")
                
            if len(articles) >= 10:  # Get more articles since we're combining sources
                break
                
        logging.info(f"Found {len(articles)} relevant Reuters/AP articles")
        return articles
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error fetching from NewsAPI: {e}")
        return []
    except Exception as e:
        logging.error(f"Error fetching from NewsAPI: {e}")
        return []

def save_to_json(articles, output_file='conflict_news.json'):
    try:
        if not articles:
            logging.error("No articles to save!")
            # Create minimal valid JSON with error message
            articles = [{
                "title": "Error: No articles found",
                "url": "",
                "source": "system",
                "published": datetime.now().isoformat(),
                "summary": "The news scraper encountered issues. Please check the logs."
            }]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved {len(articles)} articles to {output_file}")
            
    except Exception as e:
        logging.error(f"Failed to save articles: {e}")
        raise

if __name__ == "__main__":
    try:
        logging.info("Starting news monitoring...")
        
        # Force remove the existing file
        output_file = "conflict_news.json"
        try:
            if os.path.exists(output_file):
                os.remove(output_file)
                logging.info(f"Removed existing {output_file}")
            time.sleep(1)  # Small delay to ensure file system sync
        except Exception as e:
            logging.error(f"Error removing file: {e}")
        
        # Run the monitoring with fresh start
        monitor_news_sources(output_file)
        
        # Verify the file was created
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                data = json.load(f)
                logging.info(f"Successfully saved {len(data)} articles")
                
    except Exception as e:
        logging.error(f"Failed to run news monitoring: {e}")

