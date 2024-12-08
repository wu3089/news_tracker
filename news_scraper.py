import requests
from bs4 import BeautifulSoup
import feedparser
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import pipeline
from dateutil import parser as date_parser
from collections import defaultdict
from tqdm import tqdm
import re
import json

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def truncate_text_for_summarization(text, max_tokens=800):
    tokens = word_tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return ' '.join(tokens)

def get_article_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text().replace('\xa0', ' ') for p in paragraphs if p])
        if not article_text.strip():
            raise ValueError("No article content found.")
        return article_text
    except Exception:
        return None

def clean_text(text):
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def summarize_with_nlp(text, min_length=50, max_length=200):
    text = truncate_text_for_summarization(text, max_tokens=800)
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        summary = clean_text(summary)
        return summary
    except Exception:
        sents = sent_tokenize(text)
        return ' '.join(sents[:5])

def is_conflict_related(title, text, keywords):
    combined_text = (title + " " + text).lower()
    for keyword in keywords:
        if keyword.lower() in combined_text:
            keyword_index = combined_text.find(keyword.lower())
            context_start = max(0, keyword_index - 50)
            context_end = min(len(combined_text), keyword_index + len(keyword) + 50)
            context = combined_text[context_start:context_end]
            if any(cw in context for cw in ["war", "battle", "attack", "kill", "death", "crisis", "rebel", "bomb", "fight"]):
                return True
    return False

def is_us_domestic_only(title, text):
    combined_text = (title + " " + text).lower()

    us_markers = [
        "america ", "american ", " united states ", " u.s.", "usa", "washington", "white house", "pentagon",
        "congress", "senate", "donald trump", "trump administration", "biden administration", "us president", "american politics"
    ]

    global_markers = [
        "syria", "ukraine", "russia", "afghanistan", "iraq", "iran", "yemen", "libya", "somalia",
        "mali", "congo", "pakistan", "india", "kashmir", "kurd", "palestine", "israel", "lebanon",
        "taiwan", "china", "myanmar", "eritrea", "south sudan", "colombia", "venezuela",
        "international", "global", "cross-border"
    ]

    has_us = any(um in combined_text for um in us_markers)
    has_global = any(gm in combined_text for gm in global_markers)

    if has_us and not has_global:
        return True
    return False

def is_global_conflict(title, text):
    combined_text = (title + " " + text).lower()
    global_markers = [
        "syria", "ukraine", "russia", "afghanistan", "iraq", "iran", "yemen", "libya", "somalia",
        "mali", "congo", "pakistan", "india", "kashmir", "kurd", "palestine", "israel", "lebanon",
        "taiwan", "china", "myanmar", "eritrea", "south sudan", "colombia", "venezuela",
        "international", "global", "cross-border"
    ]
    return any(gm in combined_text for gm in global_markers)

def parse_date(published_str):
    if not published_str:
        return None
    try:
        dt = date_parser.parse(published_str)
        return dt
    except:
        return None

def get_region(title, text):
    combined_text = (title + " " + text).lower()
    regions = {
        "Middle East": ["syria", "iran", "iraq", "yemen", "libya", "lebanon", "palestine", "israel"],
        "Africa": ["somalia", "mali", "congo", "eritrea", "south sudan"],
        "Asia": ["afghanistan", "pakistan", "india", "kashmir", "kurd", "taiwan", "china", "myanmar"],
        "Europe": ["russia", "ukraine"],
        "Latin America": ["colombia", "venezuela"],
        "Global": ["international", "global", "cross-border"]
    }

    for region_name, markers in regions.items():
        if any(m in combined_text for m in markers):
            return region_name
    return None

lowercase_words = {"a", "an", "the", "and", "but", "or", "nor", "for", "in", "on", "at", "to", "from", "by", "of"}

def headline_style(title):
    words = title.strip().split()
    if not words:
        return title

    words[0] = words[0].capitalize()

    for i in range(1, len(words)):
        w = words[i].lower()
        if w not in lowercase_words or w == "i":
            w = w.capitalize()
        words[i] = w

    return ' '.join(words)

def create_news_snippet(entry, source, summarizer, conflict_keywords):
    link = entry.link
    article_text = get_article_content(link)
    if article_text is None:
        return None

    raw_title = entry.title.strip()
    dt = parse_date(getattr(entry, 'published', None))
    if dt is None:
        dt = date_parser.parse(time.strftime('%Y-%m-%d'))

    if not is_conflict_related(raw_title, article_text, conflict_keywords):
        return None
    if not is_global_conflict(raw_title, article_text):
        return None
    if is_us_domestic_only(raw_title, article_text):
        return None

    region = get_region(raw_title, article_text)
    if not region:
        return None

    cleaned_title = clean_text(raw_title)
    styled_title = headline_style(cleaned_title)
    date_str = dt.strftime('%B %d, %Y')

    summary = summarize_with_nlp(article_text)

    snippet = {
        "title": styled_title,
        "date": dt.isoformat(),
        "date_str": date_str,
        "summary": summary,
        "source": source,
        "link": link,
        "region": region
    }
    return snippet

def monitor_rss_feeds(feeds, keywords, summarizer, interval=3600):
    # Instead of looping forever, let's just run once.
    # If you want it to run forever, revert to while True.
    # For GitHub Actions, one run is enough per trigger.
    
    print(f"Checking RSS feeds at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    by_source = defaultdict(list)

    for source, url in feeds.items():
        feed = feedparser.parse(url)
        entries = feed.entries[:20]  # Only consider first 20 entries per source
        if not entries:
            print(f"No entries found for {source} feed: {url}")
        for entry in tqdm(entries, desc=f"Processing {source}", unit="article"):
            snippet = create_news_snippet(entry, source, summarizer, keywords)
            if snippet:
                by_source[source].append(snippet)

    selected_articles = []
    for source, articles in by_source.items():
        articles.sort(key=lambda x: x['date'], reverse=True)
        top_ten = articles[:10]
        selected_articles.extend(top_ten)

    # Save selected_articles to conflict_news.json
    with open("conflict_news.json", "w") as f:
        json.dump(selected_articles, f, indent=2)

    if not selected_articles:
        print("No matching articles found.")
    else:
        print(f"Generated conflict_news.json with {len(selected_articles)} articles.")


conflict_keywords = [
    "war", "conflict", "battle", "attack", "military", "rebel", "bombing", "airstrike",
    "insurgent", "troops", "violence", "casualties", "killed", "died", "wounded", "injured",
    "displaced", "refugee", "crisis", "unrest", "clashes", "fighting", "offensive", "invasion",
    "occupation", "coup", "rebellion", "terrorism", "extremism", "militant", "insurgency",
    "civil war", "armed conflict", "protest", "riot", "demonstration", "firefight", "shelling",
    "explosion", "massacre", "atrocity"
]

rss_feeds = {
    "BBC": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "Reuters": "https://feeds.feedburner.com/reuters/worldNews",
    "AP": "https://feeds.apnews.com/apf-world"
}

print("Initializing summarizer pipeline...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("Summarizer initialized.")

monitor_rss_feeds(rss_feeds, conflict_keywords, summarizer, interval=1800)
