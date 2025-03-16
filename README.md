Global Conflict News Tracker

A automated news aggregation and summarization system that automatically tracks, filters, and summarizes global conflict-related news.

## Features

- **Automated News Collection**: Aggregates news from multiple reliable sources:
  - BBC News
  - Reuters
  - Associated Press
  - Al Jazeera
  - The New York Times
  - The Guardian
  - France 24

- **Smart Filtering**: 
  - Focuses on conflict-related news using sophisticated keyword categorization
  - Filters out non-conflict and US-domestic news
  - Removes duplicate stories across different sources
  - Prioritizes primary news sources

- **AI-Powered Processing**:
  - Uses Google's Gemini 2.0 Flash-Lite model for:
    - Article summarization
    - Headline paraphrasing
  - Maintains factual accuracy while improving readability

- **Clean Web Interface**:
  - Responsive design
  - Article navigation minimap
  - Copy functionality for each article
  - Bulk copy option for all articles
  - Mobile-friendly layout

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key
NEWS_API_KEY=your_newsapi_key
NYT_API_KEY=your_nyt_api_key
```

4. Run the news scraper:
```bash
python news_scraper.py
```

5. Serve the web interface (you can use any HTTP server):
```bash
python -m http.server 8000
```

## Automated Updates

The project includes GitHub Actions workflow that:
- Runs every 12 hours to fetch fresh news
- Commits updated news to the repository
- Maintains a consistent JSON format
- Includes error logging and monitoring


## Notes

- This is a demonstration version showcasing automated news summarization capabilities
- Articles are filtered based on comprehensive conflict-related keyword categories
- Summaries are generated in an objective, fact-focused style

## License

MIT License

