import json
import re
import string
import argparse

import praw
from textblob import TextBlob
import textstat
import emoji

CLIENT_ID = "iiLVOdpeJZQlNYJkoBqCeA"
CLIENT_SECRET = "Vl9QiYx9yo7CnofMGWjFs_DS3H3s2Q"
USER_AGENT = "web:assignment1:1.0 (by /u/Abject_Dog_6540)"

dd = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

SUBREDDITS = [
    "AmIOverreacting",
    "AITAH",
    "AmItheAsshole",
    "ShowerThoughts",
]

def compute_meta_feats(title: str, body: str):
    text = title + ' ' + body
    # sentence count
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    num_sentences = len(sentences)
    # tokens & avg length
    tokens = text.split()
    avg_token_len = sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0
    # uppercase ratio
    upper_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0.0
    # punctuation count
    punct_count = sum(1 for c in text if c in string.punctuation)
    # emoji count
    emoji_count = len(emoji.emoji_list(text))
    # sentiment polarity
    sentiment_score = TextBlob(text).sentiment.polarity
    # readability
    readability_flesch = textstat.flesch_reading_ease(text)
    # lengths
    title_len = len(title)
    body_len = len(body)
    # question flag
    is_question = int(title.strip().endswith('?'))

    return [
        num_sentences,
        avg_token_len,
        upper_ratio,
        punct_count,
        emoji_count,
        sentiment_score,
        readability_flesch,
        title_len,
        body_len,
        is_question,
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--limit', type=int, default=5,
        help='Number of latest posts to fetch per subreddit'
    )
    parser.add_argument(
        '--output', type=str, default='inference_input.json',
        help='Path to save JSON'
    )
    args = parser.parse_args()

    results = []
    for sub in SUBREDDITS:
        for submission in dd.subreddit(sub).new(limit=args.limit):
            title = submission.title
            body = submission.selftext or ''
            feats = compute_meta_feats(title, body)

            results.append({
                'text': title + '\n' + body,
                'meta_feats': feats,
                'subreddit': submission.subreddit.display_name,
                'score': submission.score,
                'num_comments': submission.num_comments,
            })

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} posts from {len(SUBREDDITS)} subreddits to {args.output}")
