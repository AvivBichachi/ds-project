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
# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from datetime import datetime, timedelta
import traceback
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


def loadAndTrainModel():
    global LR_model
    global tf_vector
    model_filename = './models/nlp_model.sav'
    tf_vector_filename = './models/tf_vector_nlp_model.sav'
    # check if there is a saved trained model:
    createDir('./models')
    if os.path.isfile(model_filename) and os.path.isfile(tf_vector_filename):
        print('[NLP-MODEL] Saved model found !')
        LR_model = pickle.load(open(model_filename, 'rb'))
        tf_vector = pickle.load(open(tf_vector_filename, 'rb'))
        return
    print('[NLP-MODEL] Saved model was not found, training a new one')
    print('[NLP-MODEL] Loading tweets db')
    dataset = load_dataset(
        "data/tweets_clean_training.csv", ['row', 'target', 'text'])
    # Remove unwanted columns from dataset
    print('[NLP-MODEL] removing cols')
    n_dataset = remove_unwanted_cols(
        dataset, ['row'])
    # Preprocess data
    print('[NLP-MODEL] Cleaning text')
    # Split dataset into Train, Test
    print('[NLP-MODEL] get_feature_vector')
    # Same tf vector will be used for Testing sentiments on unseen trending data
    tf_vector = get_feature_vector(
        np.array(dataset.iloc[:, 1].values.astype('U')).ravel())
    print('[NLP-MODEL] tfVector = ', tf_vector)
    print('[NLP-MODEL] transform')
    X = tf_vector.transform(
        np.array(dataset.iloc[:, 1].values.astype('U')).ravel())
    y = np.array(dataset.iloc[:, 0]).ravel()
    print('[NLP-MODEL] train_test_split')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=30)
    print('[NLP-MODEL] make_pipeline')
    pipe = make_pipeline(StandardScaler(with_mean=False),
                         LogisticRegression(solver='lbfgs', max_iter=300))
    print('[NLP-MODEL] scaling ')
    LR_model = pipe.fit(X_train, y_train)
    # save the model to disk
    pickle.dump(LR_model, open(model_filename, 'wb'))
    pickle.dump(tf_vector, open(tf_vector_filename, 'wb'))
    pipe.score(X_test, y_test)
    print('[NLP-MODEL] DONE scaling')
    y_predict_lr = LR_model.predict(X_test)
    print('[NLP-MODEL] Accuracy score: ', accuracy_score(y_test, y_predict_lr))


def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector


def int_to_string(sentiment):
    if sentiment == 0:
        return "Negative"
    elif sentiment == 2:
        return "Neutral"
    else:
        return "Positive"


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
    test_feature = tf_vector.transform(np.array(cleanTweet).ravel())
    test_prediction_lr = LR_model.predict(test_feature)
    return int_to_string(test_prediction_lr)


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


# loadAndTrainModel()
# getDataByKeywords()
# calculatePolarityForAllData()
# calculateResultsForEachUser()
# calculateResultsCombined()