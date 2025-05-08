import praw
import json
import time

# Reddit API credentials (You must fill in these fields with your own credentials)
CLIENT_ID = 
CLIENT_SECRET = 
USER_AGENT = 

# Initialize Reddit API
reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)

def scrape_reddit(query, num_posts=100, output_file="data.json"):
    subreddit = reddit.subreddit("all")
    posts_data = []

    for submission in subreddit.search(query, limit=num_posts):
        post_info = {
            "id": submission.id,
            "title": submission.title,
            "score": submission.score,
            "url": submission.url,
            "num_comments": submission.num_comments,
            "created_utc": submission.created_utc,
            "selftext": submission.selftext,
            "comments": []
        }
        print('submission: ', post_info['id'])

        submission.comments.replace_more(limit=0)  # Retrieve only top-level comments
        for comment in submission.comments.list():
            post_info["comments"].append({
                "id": comment.id,
                "score": comment.score,
                "body": comment.body,
                "created_utc": comment.created_utc
            })
            print('\tcomment: ', post_info['comments'][0]['id'])

	
        posts_data.append(post_info)

        # Delay to avoid rate limiting
        #time.sleep(1)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(posts_data, f, indent=4)

    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    user_query = input("Enter a search query: ")
    scrape_reddit(user_query)
