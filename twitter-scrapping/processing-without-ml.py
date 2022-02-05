from asyncio.windows_events import NULL
import pandas
import json
import re
import os
import pandas as pd
import numpy as np
import re
import string
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from datetime import datetime, timedelta
import traceback
from textblob import TextBlob
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))  # stop words for tweets analysys


def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset


def remove_unwanted_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset


def sentiment_analysis(tweet):
    # Create a function to get the polarity
    def getPolarity(text):
        return TextBlob(text).sentiment.polarity

    # Create two new columns 'Subjectivity' & 'Polarity'
    polarity = getPolarity(tweet)

    def getAnalysis(score):
        if score <= -0.45:
            return 'Negative'
        elif score >= 0.45:
            return 'Positive'
        return 'Neutral'
        
    return getAnalysis(polarity)


def preprocess_tweet_text(tweet):
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]

    return " ".join(filtered_words)


def loadJSON(fileName):
    file = open(fileName)
    data = json.load(file)
    return data


def createDir(path):
    try:
        os.makedirs(path)
        print(f'{path} created')

    except OSError as e:
        print(f'{path} exists, skipping')


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


def getDataByKeywords():
    '''
    generate data by keywords for all users for all coins.
    '''
    coins = loadJSON('coins.json').get('coins')
    for coin in coins:
        symbol = coin.get('symbol')
        createDir(f'./output/{symbol}')  # creates a dir if does not exist.
        keywords = coin.get('keywords')
        print(f'on coin {symbol}')
        for user in users:
            print(f'reading user {user}')
            csvData = pandas.read_csv(f'./users_tweets/{user}.csv')
            keys = str.join('|', keywords)
            matchingData = csvData.set_index('Text').filter(
                regex=re.compile(keys, re.IGNORECASE), axis=0)
            print(f'writing data to file ./output/{symbol}/{user}.csv')
            matchingData.to_csv(f'./output/{symbol}/{user}.csv')


def getTweetSentiment(tweetText):
    '''
    function to get a tweet sentiment
    '''
    # create TextBlob object of tweet
    cleanTweet = preprocess_tweet_text(tweetText)
    return sentiment_analysis(cleanTweet)


def addSentimentToFile(filePath):
    '''
    for a given csv files of users comment about a coin,
    calculate and add the sentiment polarity value of the tweet
    '''
    csvData = pandas.read_csv(filePath)
    polarity = [getTweetSentiment(tweet)
                for tweet in csvData['Text']]
    csvData['polarity'] = polarity
    csvData.to_csv(filePath)


def calculatePolarityForAllData():
    coins = loadJSON('coins.json').get('coins')
    for coin in coins:
        symbol = coin.get('symbol')
        print(f'-------on coin {symbol}-------')
        for user in users:
            print(f'\treading {symbol}@{user}')
            addSentimentToFile(f'./output/{symbol}/{user}.csv')


def getTweetsByPolarity(polarity, symbol, user):
    '''
    returns the posts of a user with the topic of a specific coin
    that are higer
    '''
    tweets = pandas.read_csv(f'./output/{symbol}/{user}.csv')
    return tweets[tweets['polarity'] == polarity]


def decreseDays(dateString, n):
    tmp = datetime.strptime(dateString, "%Y-%m-%d")
    days = timedelta(n)
    tmp -= days
    return datetime.strftime(tmp, "%Y-%m-%d")


def calculateDeltas(symbol, user):
    '''
    this function will return the delta in volume and close-open gap within the given range
    for a specific coin symbol,user,date.
    will return array of :
    {user,date,polarity,volume,valueDelta}
    '''
    # return object:
    data = []
    # load coin data:
    coinData = pandas.read_csv(f'./coinsData/{symbol}.csv')
    # get relevant posts:
    positiveTweets = getTweetsByPolarity(
        'Positive', symbol, user).sort_values(by='polarity', ascending=True)
    negativeTweets = getTweetsByPolarity('Negative', symbol,
                                         user).sort_values(by='Datetime', ascending=True)
    neutralTweets = getTweetsByPolarity('Neutral', symbol,
                                        user).sort_values(by='Datetime', ascending=True)
    # extract the dates:
    # we need to convert to DD/MM/YYYY
    positiveDates = [date.split(' ')[0] for date in positiveTweets['Datetime']]
    positiveDates.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))
    negativeDates = [date.split(' ')[0] for date in negativeTweets['Datetime']]
    negativeDates.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))
    neutralDates = [date.split(' ')[0] for date in neutralTweets['Datetime']]
    neutralDates.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))
    # calculate avarage in volume change within the given range (in days)
    try:
        for idx, date in enumerate(positiveDates):
            valueDelta = (coinData[coinData['Time'] == date].Close -
                          coinData[coinData['Time'] == date].Open)
            if not len(valueDelta.values):
                continue
            valueDelta = valueDelta.values[0]
            yesterday = decreseDays(date, 1)
            volume = coinData[coinData['Time'] == date].Volume.values[0]
            yesterdayVolume = coinData[coinData['Time'] == yesterday].Volume
            if not len(yesterdayVolume.values):
                continue
            data.append(
                {'user': user,
                 'date': date,
                 'tweet': positiveTweets.iloc[idx].Text,
                 'polarity': positiveTweets.iloc[idx].polarity,
                 'valueDelta': valueDelta,
                 'volume': volume,
                 'volumeDelta': volume - yesterdayVolume.values[0]})

        for idx, date in enumerate(negativeDates):
            valueDelta = (coinData[coinData['Time'] ==
                                   date].Close-coinData[coinData['Time'] == date].Open)
            if not len(valueDelta.values):
                continue
            valueDelta = valueDelta.values[0]
            volume = coinData[coinData['Time'] == date].Volume.values[0]
            yesterday = decreseDays(date, 1)
            yesterdayVolume = coinData[coinData['Time'] == yesterday].Volume
            if not len(yesterdayVolume.values):
                continue
            data.append(
                {'user': user,
                 'date': date,
                 'tweet': negativeTweets.iloc[idx].Text,
                 'polarity': negativeTweets.iloc[idx].polarity,
                 'valueDelta': valueDelta,
                 'volume': volume,
                 'volumeDelta': volume - volume - yesterdayVolume.values[0]})

        for idx, date in enumerate(neutralDates):
            valueDelta = (coinData[coinData['Time'] ==
                                   date].Close-coinData[coinData['Time'] == date].Open)
            if not len(valueDelta.values):
                continue
            valueDelta = valueDelta.values[0]
            volume = coinData[coinData['Time'] == date].Volume.values[0]
            yesterday = decreseDays(date, 1)
            yesterdayVolume = coinData[coinData['Time'] == yesterday].Volume
            if not len(yesterdayVolume.values):
                continue
            data.append(
                {'user': user,
                 'date': date,
                 'tweet': neutralTweets.iloc[idx].Text,
                 'polarity': neutralTweets.iloc[idx].polarity,
                 'valueDelta': valueDelta,
                 'volume': volume,
                 'volumeDelta': volume - volume - yesterdayVolume.values[0]})
    except Exception as e:
        print(traceback.format_exc())
        print('missing data, skipped.')
    return data


def calculateResultsForEachUser():
    coins = loadJSON('coins.json').get('coins')
    for coin in coins:
        symbol = coin.get('symbol')
        print(f'------- calculate Results on coin {symbol}-------')
        for user in users:
            print(f'\treading {symbol}@{user}')
            data = calculateDeltas(symbol, user)
            df = pd.json_normalize(data)
            df.to_csv(f'./output/{symbol}/{user}_results.csv')


def calculateResultsCombined():
    coins = loadJSON('coins.json').get('coins')
    for coin in coins:
        symbol = coin.get('symbol')
        data = []
        print(f'------- calculate Results on coin {symbol}-------')
        for user in users:
            print(f'\treading {symbol}@{user}')
            data += calculateDeltas(symbol, user)
        df = pd.json_normalize(data).sort_values(by=["date"])
        df = df[df['volume'] != 0]
        df.to_csv(f'./output/{symbol}/all_results.csv')


def generateDataset():
    coins = loadJSON('coins.json').get('coins')
    for coin in coins:
        symbol = coin.get('symbol')
        data = []
        target = []
        resDF = pd.read_csv(f'./output/{symbol}/all_results.csv')
        print(f'------- calculate Results on coin {symbol}-------')
        for index, row in resDF.iterrows():
            data.append(f"{row['user']},{row['polarity']}")
            target.append(f"{1 if row['valueDelta']>0 else 0}")
        df = pd.DataFrame()
        df['data'] = data
        df['target'] = target
        df.to_csv(f'./output/{symbol}/dataset.csv', index=False)


def convertDateFormat(dateString):
    # we need to convert to dd/mm/yyyy
    # from yyyy-mm-dd
    tmp = datetime.strptime(dateString, '%Y-%m-%d')
    return tmp.strftime('%d/%m/%Y')


# getDataByKeywords()
# calculatePolarityForAllData()
# calculateResultsForEachUser()
# calculateResultsCombined()
