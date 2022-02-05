import snscrape.modules.twitter as sntwitter
import pandas as pd
import datetime
import pytz

utc = pytz.UTC
# Creating list to append tweet data to
tweets_list = []

# defining limits
stopdate = datetime.datetime(2021, 1, 1, tzinfo=utc)
hard_limit = 10000
users = ["elonmusk",
         "VitalikButerin",
         "rogerkver",
         "APompliano",
         "brian_armstrong",
         "BarrySilbert",
         "aantonop",
         "saylor",
         "TimDraper",
         "SatoshiLite"]
# TwitterSearchScraper
for usertag in users:
    print(f'scraping {usertag}')
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'from:{usertag}').get_items()):
        if tweet.date < stopdate or i > hard_limit:
            break
        tweets_list.append([tweet.date, tweet.user.username, tweet.content])

    # Creating a dataframe from the tweets list above
    tweets_df = pd.DataFrame(tweets_list, columns=[
                              'Datetime', 'Username', 'Text'])
    tweets_df.to_csv(f'./output/{usertag}.csv')
    tweets_list = []
