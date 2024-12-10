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
        # Only download what we actually use
        nltk.download('punkt', quiet=True)
        nltk.download('words', quiet=True)
        
        # Create NLTK data directory if it doesn't exist
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Verify downloads
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/words')
            logging.info("NLTK resources verified successfully")
        except LookupError as e:
            logging.error(f"NLTK resource verification failed: {e}")
            
    except Exception as e:
        logging.warning(f"NLTK setup warning: {e}")

# Call setup at start
setup_nltk()

# Get API keys from environment variables
API_KEY = os.getenv('GOOGLE_API_KEY')
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

US_KEYWORDS = [
    # Domestic Politics
    "Congress debate",
    "Senate hearing",
    "House Republicans",
    "House Democrats",
    "domestic policy",
    "state legislature",
    
    # US Internal Affairs
    "Supreme Court ruling",
    "federal court",
    "gubernatorial",
    "state election",
    
    # US Business/Economy (when purely domestic)
    "Wall Street",
    "Federal Reserve",
    "US housing market",
    "US inflation",
]
CONFLICT_REGIONS = [
    # Major Active Conflicts
    "Ukraine", "Russia", "Kyiv", "Moscow", "Donbas", "Crimea",
    "Syria", "Damascus", "Assad", "Aleppo",
    "Yemen", "Houthi", "Sanaa", "Red Sea",
    "Israel", "Gaza", "Palestine", "Hamas", "West Bank", "Netanyahu",
    "Sudan", "Khartoum", "RSF", "Darfur",
    
    # Regional Conflicts
    "Libya", "Tripoli", "Benghazi",
    "Afghanistan", "Taliban", "Kabul",
    "Nagorno-Karabakh", "Armenia", "Azerbaijan",
    "Somalia", "Al-Shabaab", "Mogadishu",
    "Congo", "DRC", "Kinshasa",
    "Central African Republic", "CAR", "Bangui",
    
    # Strategic Tensions
    "South China Sea", "Taiwan Strait", "Beijing", "Taipei",
    "Iran", "Tehran", "IRGC", "Strait of Hormuz",
    "North Korea", "Pyongyang", "DPRK",
    "Kashmir", "India Pakistan border",
    
    # Regional Instability
    "Sahel", "Mali", "Burkina Faso", "Niger",
    "Ethiopia", "Tigray", "Amhara", "Oromia",
    "Haiti", "Port-au-Prince",
    "Venezuela", "Caracas", "Maduro",
    "Myanmar", "Burma", "Rohingya",
]

CONFLICT_KEYWORDS = [
    # Military Actions
    "war", "invasion", "offensive", "counteroffensive",
    "airstrike", "bombardment", "missile", "drone attack",
    "military operation", "combat", "fighting",
    
    # Political Violence
    "conflict", "crisis", "uprising", "insurgency",
    "rebellion", "coup", "civil war", "unrest",
    
    # Security Issues
    "terrorism", "extremist", "militant", "insurgent",
    "armed group", "militia", "paramilitary",
    
    # Humanitarian Impact
    "refugee", "displaced", "humanitarian crisis",
    "civilian casualties", "war crimes", "genocide",
    
    # Peace Process
    "ceasefire", "peace talks", "negotiation",
    "diplomatic crisis", "sanctions", "resolution"
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
    try:
        # Simple word-based truncation
        words = text.split()
        if len(words) > max_tokens:
            return ' '.join(words[:max_tokens])
        return text
    except Exception as e:
        logging.warning(f"Error in truncate_text: {e}")
        return text[:2000]  # Fallback to character-based truncation

def summarize_with_google_api(text):
    """Summarize text using Google's Generative AI with error handling"""
    try:
        truncated_text = truncate_text(text)
        
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""Please provide a comprehensive summary in active voice of this news article in 3-5 sentences in the style of the Council on Foreign Relations. Include:
        - Main events and their significance
        - Key players involved
        - Important context or background
        - Any notable geopolitical implications
        
        Article text: {truncated_text}
        
        Format the summary as a single paragraph focusing on factual information."""
        
        response = model.generate_content(prompt, safety_settings={
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        })
        
        if response.text:
            return response.text.strip()
        return None
        
    except Exception as e:
        logging.error(f"Summarization failed: {e}")
        return None

def format_date(date_str):
    """Format date as 'Month Day, Year'"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S").strftime("%B %d, %Y")
    except Exception:
        return datetime.now().strftime("%B %d, %Y")

def filter_articles(articles):
    """Filter out purely domestic U.S. content while keeping international stories"""
    filtered = []
    seen_hashes = set()
    
    for article in articles:
        title_lower = article['title'].lower()
        summary_lower = article.get('summary', '').lower()
        content = title_lower + " " + summary_lower
        
        # Count how many US keywords match
        us_matches = [keyword.lower() for keyword in US_KEYWORDS if keyword.lower() in content]
        
        # Only filter out if multiple domestic keywords are found
        if len(us_matches) > 2:  # Allow some US mentions before filtering
            logging.info(f"Excluding U.S.-centric article: {article['title']}")
            logging.debug(f"Matched US keywords: {us_matches}")
            continue
            
        # Rest of the filtering logic remains the same...
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
        
        # Parse the actual publication date from the NYT article
        pub_date = item.get('published_date', '')
        date_str = datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%S%z").strftime("%B %d, %Y")
        
        return {
            'title': title,
            'text': abstract,
            'link': url,
            'date_str': date_str,  # Use the actual formatted date
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
        cutoff_date = datetime.now() - timedelta(days=7)
        
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
    cutoff_date = datetime.now() - timedelta(days=7)
    
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

def process_articles(articles, conn, max_articles=20):
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
            
            # Get the article's publication date
            pub_date = article.get('pub_date')
            if pub_date:
                try:
                    date_str = datetime.strptime(pub_date, "%Y-%m-%d").strftime("%B %d, %Y")
                except ValueError:
                    date_str = datetime.now().strftime("%B %d, %Y")
            else:
                date_str = datetime.now().strftime("%B %d, %Y")
            
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
    """Monitor all news sources and save results"""
    all_articles = []
    
    # Fetch from each source
    newsapi_articles = get_news_from_newsapi()
    bbc_articles = scrape_bbc_news()
    nyt_articles = scrape_nyt_news()
    
    # Combine all articles
    all_articles.extend(newsapi_articles)
    all_articles.extend(bbc_articles)
    all_articles.extend(nyt_articles)
    
    # Process and save
    try:
        if all_articles:
            # Create a connection for article caching
            conn = sqlite3.connect('article_cache.db')
            # Process the articles to get summaries
            processed_articles = process_articles(all_articles, conn)
            conn.close()
            # Save the processed articles
            save_snippets(output_file, processed_articles)
        else:
            logging.warning("No articles found from any source")
            save_snippets(output_file, [])
    except Exception as e:
        logging.error(f"Error in monitor_news_sources: {e}")
        save_snippets(output_file, [])

def save_snippets(output_file, snippets):
    """Save news snippets to JSON file"""
    try:
        if not isinstance(snippets, list):
            logging.error(f"Expected list of snippets, got {type(snippets)}")
            snippets = []
            
        if not snippets:
            snippets = [{
                "title": "No Recent Articles Found",
                "summary": "No conflict-related articles were found in the past 3 days.",
                "source": "system",
                "url": "",
                "date_str": datetime.now().strftime("%B %d, %Y")
            }]
        
        # Ensure all required fields exist and are strings
        for snippet in snippets:
            if not isinstance(snippet, dict):
                continue
            snippet['date_str'] = str(snippet.get('date_str', datetime.now().strftime("%B %d, %Y")))
            snippet['summary'] = str(snippet.get('summary', "No summary available"))
            snippet['url'] = str(snippet.get('url', ""))
            snippet['source'] = str(snippet.get('source', "unknown"))
            snippet['title'] = str(snippet.get('title', "Untitled Article"))

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(snippets, f, indent=2, ensure_ascii=False)
        
        # Verify the file was written correctly
        with open(output_file, 'r', encoding='utf-8') as f:
            verification = json.load(f)
            if not verification:
                raise ValueError("JSON file was written but is empty")
            
        logging.info(f"Successfully saved {len(snippets)} snippets to {output_file}")
        
    except Exception as e:
        logging.error(f"Error saving snippets: {e}")
        # Create emergency fallback content
        fallback = [{
            "title": "Error Saving Articles",
            "summary": f"An error occurred while saving articles: {str(e)}",
            "source": "system",
            "url": "",
            "date_str": datetime.now().strftime("%B %d, %Y")
        }]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fallback, f, indent=2, ensure_ascii=False)

def get_news_from_newsapi():
    """Get news from NewsAPI.org focusing on Reuters and AP world news"""
    logging.info("Fetching from NewsAPI...")
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        logging.warning("No NewsAPI key found")
        return []
        
    url = "https://newsapi.org/v2/everything"
    
    params = {
        'apiKey': api_key,
        'domains': 'reuters.com,apnews.com',
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 100,
        'q': '(world OR international OR global) AND (conflict OR war OR crisis OR tension)',
        'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    }
    
    try:
        logging.info(f"Making NewsAPI request with params: {params}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'ok':
            logging.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return []
            
        articles = []
        for article in data.get('articles', []):
            title = article.get('title', '')
            source = article.get('source', {}).get('name', '')
            
            if source.lower() in ['reuters', 'associated press', 'ap']:
                # Paraphrase the title like we do for NYT
                paraphrased_title = paraphrase_title(title)
                
                articles.append({
                    'title': paraphrased_title,
                    'link': article.get('url', ''),
                    'source': source,
                    'pub_date': article.get('publishedAt', '').split('T')[0]  # Get the date part of ISO format
                })
                logging.info(f"Added {source} article: {paraphrased_title}")
                
        logging.info(f"Found {len(articles)} relevant Reuters/AP articles")
        return articles
        
    except Exception as e:
        logging.error(f"Failed to fetch from NewsAPI: {e}")
        return []
    
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
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not data:
                    raise ValueError("JSON file exists but is empty")
                logging.info(f"Successfully saved {len(data)} articles")
                
    except Exception as e:
        logging.error(f"Failed to run news monitoring: {e}")

