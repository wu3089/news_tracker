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
import google.generativeai as genai

# Load environment variables first
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# NLTK setup
def setup_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('words', quiet=True)
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
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

# Configure Gemini API
API_KEY = os.getenv('GOOGLE_API_KEY')
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")
genai.configure(api_key=API_KEY)

# Keywords for filtering
US_KEYWORDS = [
    "Congress debate", "Senate hearing", "House Republicans", "House Democrats",
    "domestic policy", "state legislature", "Supreme Court ruling", "federal court",
    "gubernatorial", "state election", "Wall Street", "Federal Reserve",
    "US housing market", "US inflation"
]

CONFLICT_REGIONS = [
    "Ukraine", "Russia", "Kyiv", "Moscow", "Donbas", "Crimea",
    "Syria", "Damascus", "Assad", "Aleppo", "Yemen", "Houthi",
    "Israel", "Gaza", "Palestine", "Hamas", "West Bank",
    "Sudan", "Libya", "Afghanistan", "Somalia", "Congo", "Myanmar"
]

# Update the keywords section at the top of the file
CONFLICT_CATEGORIES = {
    "MILITARY_ACTIONS": [
        "war", "invasion", "offensive", "counteroffensive", "airstrike",
        "bombardment", "missile", "drone attack", "military operation",
        "combat", "fighting", "troops", "forces", "military",
        "artillery", "tank", "warplane", "warship", "army"
    ],
    "VIOLENCE_AND_CASUALTIES": [
        "killed", "wounded", "casualties", "death toll", "civilian deaths",
        "massacre", "attack", "bombing", "explosion", "strike",
        "destroyed", "damage", "violent", "bloodshed"
    ],
    "POLITICAL_CONFLICT": [
        "conflict", "crisis", "tension", "dispute", "clash",
        "hostilities", "warfare", "insurgency", "rebellion",
        "coup", "overthrow", "uprising", "unrest"
    ],
    "PEACE_AND_DIPLOMACY": [
        "ceasefire", "peace talks", "negotiation", "diplomatic",
        "mediation", "agreement", "treaty", "resolution",
        "settlement", "accord", "truce"
    ],
    "WEAPONS_AND_SECURITY": [
        "nuclear", "missile test", "weapons", "arms", "military aid",
        "defense system", "ammunition", "arsenal", "armed forces",
        "military base", "strategic", "tactical"
    ]
}


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
        words = text.split()
        if len(words) > max_tokens:
            return ' '.join(words[:max_tokens])
        return text
    except Exception as e:
        logging.warning(f"Error in truncate_text: {e}")
        return text[:2000]

def summarize_with_google_api(text):
    """Summarize text using Google's Generative AI with error handling"""
    try:
        truncated_text = truncate_text(text)
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        prompt = f"""Please provide a comprehensive summary in active voice of this news article in 3-5 sentences in the style of the Council on Foreign Relations. Include:
        - Main events and their significance
        - Key players involved
        - Important context or background
        
        Article text: {truncated_text}
        
        Format the summary as a single paragraph focusing on factual information."""
        
        response = model.generate_content(prompt, safety_settings={
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        })
        
        if response and response.text:
            return response.text.strip()
        else:
            logging.warning("Empty response from Gemini API")
            return "No summary available"
            
    except Exception as e:
        logging.error(f"Summarization failed: {e}")
        return "No summary available"

def format_date(date_str):
    """Format date as 'Month Day, Year'"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S").strftime("%B %d, %Y")
    except Exception:
        return datetime.now().strftime("%B %d, %Y")

# Flatten CONFLICT_KEYWORDS for easy searching
CONFLICT_KEYWORDS = [keyword for category in CONFLICT_CATEGORIES.values() for keyword in category]

def is_conflict_related(title, summary=''):
    """
    Determine if an article is sufficiently conflict-related by checking for
    keyword combinations and context.
    """
    text = (title + ' ' + summary).lower()
    
    # Check for region mentions
    has_region = any(region.lower() in text for region in CONFLICT_REGIONS)
    
    # Count keywords by category
    category_matches = {
        category: sum(1 for keyword in keywords if keyword.lower() in text)
        for category, keywords in CONFLICT_CATEGORIES.items()
    }
    
    # Article must have:
    # 1. At least one region mention AND
    # 2. Either:
    #    a. 2+ keywords from MILITARY_ACTIONS or VIOLENCE_AND_CASUALTIES
    #    b. 1 keyword from those categories + 2 from other categories
    military_violence_matches = (
        category_matches['MILITARY_ACTIONS'] +
        category_matches['VIOLENCE_AND_CASUALTIES']
    )
    other_matches = sum(
        count for category, count in category_matches.items()
        if category not in ['MILITARY_ACTIONS', 'VIOLENCE_AND_CASUALTIES']
    )
    
    is_conflict = (
        has_region and (
            military_violence_matches >= 2 or
            (military_violence_matches >= 1 and other_matches >= 2)
        )
    )
    
    return is_conflict

def filter_articles(articles):
    """Filter out non-conflict articles and duplicates"""
    filtered = []
    seen_hashes = set()
    
    for article in articles:
        title = article['title']
        summary = article.get('summary', '')
        
        # Skip if not conflict-related
        if not is_conflict_related(title, summary):
            logging.info(f"Excluding non-conflict article: {title}")
            continue
        
        # Skip if US-centric
        content = (title + " " + summary).lower()
        us_matches = [keyword.lower() for keyword in US_KEYWORDS if keyword.lower() in content]
        if len(us_matches) > 2:
            logging.info(f"Excluding U.S.-centric article: {title}")
            continue
        
        # Skip if duplicate
        article_hash = hashlib.md5((title + article['link']).encode()).hexdigest()
        if article_hash in seen_hashes:
            continue
        
        seen_hashes.add(article_hash)
        filtered.append(article)
    
    return filtered

def paraphrase_title(original_title):
    """Paraphrase article title using Gemini"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        prompt = f"""Rewrite this news headline following these rules:
        1. Use title case
        2. Make it a complete sentence
        3. Keep it concise but informative
        4. Maintain the key information but rephrase it
        5. Return ONLY the rewritten headline, nothing else

        Original headline: {original_title}"""
        
        response = model.generate_content(prompt)
        title = response.text.strip()
        title = title.replace('*', '').strip()
        if ":" in title:
            title = title.split(":")[-1].strip()
        return response.text.strip().rstrip('.')
    except Exception as e:
        logging.error(f"Error paraphrasing title: {e}")
        return original_title

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
        
        date_str = format_date(pub_date)
        
        return {
            'title': title,
            'text': abstract,
            'link': url,
            'date_str': date_str,
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
            logging.error("NYT_API_KEY environment variable is not set")
            return []

        url = f"https://api.nytimes.com/svc/topstories/v2/world.json?api-key={nyt_api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if 'results' not in data:
            logging.error(f"Unexpected API response structure. Keys found: {data.keys()}")
            return []
        
        articles = []
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for item in data.get('results', []):
            try:
                pub_date = datetime.strptime(item.get('published_date', ''), "%Y-%m-%dT%H:%M:%S%z")
                if pub_date.replace(tzinfo=None) < cutoff_date:
                    continue
                
                title = item.get('title', '').lower()
                if any(keyword.lower() in title for keyword in CONFLICT_KEYWORDS):
                    processed_article = process_nyt_article(item)
                    if processed_article:
                        articles.append(processed_article)
                
                if len(articles) >= 5:
                    break
                    
            except (KeyError, ValueError) as e:
                logging.warning(f"Error processing NYT article: {e}")
                continue
                
        return articles
        
    except Exception as e:
        logging.error(f"Failed to fetch NYT articles: {e}")
        return []

def scrape_bbc_news():
    """Scrape BBC World News RSS feed"""
    logging.info("Starting BBC news scraping...")
    url = "http://feeds.bbci.co.uk/news/world/rss.xml"
    feed = feedparser.parse(url)
    articles = []
    cutoff_date = datetime.now() - timedelta(days=7)
    
    for entry in feed.entries:
        try:
            pub_date = datetime(*entry.published_parsed[:6])
            if pub_date < cutoff_date:
                continue
        except (AttributeError, TypeError):
            pass
            
        title_lower = entry.title.lower()
        if (any(keyword.lower() in title_lower for keyword in CONFLICT_KEYWORDS) and 
            not any(keyword.lower() in title_lower for keyword in US_KEYWORDS)):
            
            paraphrased_title = paraphrase_title(entry.title)
            article_text = get_article_content(entry.link)
            if article_text:
                summary = summarize_with_google_api(article_text)
            else:
                summary = "No summary available"
            
            articles.append({
                'title': paraphrased_title,
                'link': entry.link,
                'source': 'BBC',
                'pub_date': entry.get('published', datetime.now().strftime("%Y-%m-%d")),
                'summary': summary
            })
            
        if len(articles) >= 5:
            break
    return articles

def get_news_from_newsapi():
    """Get news from NewsAPI.org focusing on Reuters and AP world news"""
    logging.info("Fetching from NewsAPI...")
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        logging.warning("No NewsAPI key found")
        return []
        
    url = "https://newsapi.org/v2/everything"
    # Define the domains list
    domains = [
        'reuters.com',
        'apnews.com',
        'aljazeera.com',
        'france24.com',
        'theguardian.com'
    ]
    params = {
        'apiKey': api_key,
        'domains': ','.join(domains),
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 50,
        'q': '(war OR conflict OR military OR attack OR fighting OR troops OR forces OR killed OR casualties OR missile OR airstrike OR bombing) AND (Ukraine OR Russia OR Israel OR Gaza OR Hamas OR Syria OR Yemen OR Sudan OR Libya OR Myanmar)',
        'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    }
    
    try:
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
                paraphrased_title = paraphrase_title(title)
                article_text = get_article_content(article.get('url', ''))
                summary = summarize_with_google_api(article_text) if article_text else "No summary available"
                
                try:
                    pub_date_str = article.get('publishedAt', '')
                    pub_date = date_parser.parse(pub_date_str).strftime("%B %d, %Y")
                except (ValueError, TypeError):
                    pub_date = datetime.now().strftime("%B %d, %Y")
                
                articles.append({
                    'title': paraphrased_title,
                    'link': article.get('url', ''),
                    'source': source,
                    'date_str': pub_date,
                    'summary': summary
                })
                
        return articles
        
    except Exception as e:
        logging.error(f"Failed to fetch from NewsAPI: {e}")
        return []

def monitor_news_sources(output_file):
    """Monitor all news sources and save results"""
    all_articles = []
    
    try:
        newsapi_articles = get_news_from_newsapi()
        all_articles.extend(newsapi_articles)
    except Exception as e:
        logging.error(f"Failed to fetch from NewsAPI: {e}")
    
    try:
        bbc_articles = scrape_bbc_news()
        all_articles.extend(bbc_articles)
    except Exception as e:
        logging.error(f"Failed to fetch from BBC: {e}")
    
    try:
        nyt_articles = scrape_nyt_news()
        all_articles.extend(nyt_articles)
    except Exception as e:
        logging.error(f"Failed to fetch from NYT: {e}")
    
    # Add this line to filter all articles before saving
    all_articles = filter_articles(all_articles)
    
    if all_articles:
        save_snippets(output_file, all_articles)
    else:
        save_snippets(output_file, [{
            "title": "No Articles Found",
            "summary": "No conflict-related articles were found.",
            "source": "system",
            "date_str": datetime.now().strftime("%B %d, %Y")
        }])

def save_snippets(output_file, snippets):
    """Save news snippets to JSON file"""
    try:
        if not isinstance(snippets, list):
            snippets = []
            
        if not snippets:
            snippets = [{
                "title": "No Recent Articles Found",
                "summary": "No conflict-related articles were found in the past 3 days.",
                "source": "system",
                "date_str": datetime.now().strftime("%B %d, %Y")
            }]
        
        for snippet in snippets:
            if not isinstance(snippet, dict):
                continue
            snippet['date_str'] = str(snippet.get('date_str', datetime.now().strftime("%B %d, %Y")))
            snippet['summary'] = str(snippet.get('summary', "No summary available"))
            snippet['source'] = str(snippet.get('source', "unknown"))
            snippet['title'] = str(snippet.get('title', "Untitled Article"))
            snippet['link'] = str(snippet.get('link', ""))

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(snippets, f, indent=2, ensure_ascii=False)
            
        logging.info(f"Successfully saved {len(snippets)} snippets to {output_file}")
        
    except Exception as e:
        logging.error(f"Error saving snippets: {e}")
        fallback = [{
            "title": "Error Saving Articles",
            "summary": f"An error occurred while saving articles: {str(e)}",
            "source": "system",
            "date_str": datetime.now().strftime("%B %d, %Y")
        }]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fallback, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    try:
        logging.info("Starting news monitoring...")
        
        output_file = "conflict_news.json"
        try:
            if os.path.exists(output_file):
                os.remove(output_file)
                logging.info(f"Removed existing {output_file}")
            time.sleep(1)
        except Exception as e:
            logging.error(f"Error removing file: {e}")
        
        monitor_news_sources(output_file)
        
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not data:
                    raise ValueError("JSON file exists but is empty")
                logging.info(f"Successfully saved {len(data)} articles")
                
    except Exception as e:
        logging.error(f"Failed to run news monitoring: {e}")