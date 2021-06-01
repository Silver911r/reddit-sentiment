import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import praw
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import config


reddit = praw.Reddit(
    client_id=config.CLIENT_ID,
    client_secret=config.CLIENT_SECRET,
    user_agent=config.USER_AGENT
)

def get_sentiment_r(reddit_group):
    # hot new rising top
    headlines = set()
    for submission in reddit.subreddit(reddit_group).hot(limit=None):
        headlines.add(submission.title)
    print(len(headlines))

    headlines_df = pd.DataFrame(headlines)

    headlines_df.to_csv(reddit_group + '.csv', header=False, encoding='utf-8', index=False)

    nltk.download('vader_lexicon')

    sia = SIA()
    results = []

    for line in headlines:
        pol_score = sia.polarity_scores(line) # -> dict
        pol_score['headline'] = line
        results.append(pol_score)
        
    print(results[:5], width=100)

    df = pd.DataFrame.from_records(results)

    df['label'] = 0
    df.loc[df['compound'] > 0.2, 'label'] = 1
    df.loc[df['compound'] < -0.2, 'label'] = -1

    df2 = df[['headline','label']]

    df2.to_csv(reddit_group+'_headlines_labels.csv', encoding='utf-8', index=False)

    df.label.value_counts()

    df.label.value_counts(normalize=True) * 100

    print("Positive "+ reddit_group +" headlines:\n")
    print(list(df[df['label'] == 1 ].headline)[:5], width=200)

    print("\nNegative " + reddit_group + " headlines:\n")
    print(list(df[df['label'] == -1].headline)[:5], width=200)

    fig, ax = plt.subplots(figsize=(8, 8))
    counts = df.label.value_counts(normalize=True) * 100
    sns.barplot(x=counts.index, y=counts, ax=ax)

get_sentiment_r('investing')