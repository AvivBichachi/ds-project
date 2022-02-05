import requests
import json

cryptoConfig = open("config.json")
config = json.load(cryptoConfig)
coins = config['coins']

#https://api.coinmarketcap.com/data-api/v3/cryptocurrency/historical?id=1&convertId=2781&timeStart=1041379200&timeEnd=1642809600
def constructApiUrl(coin):
    base = config.get('api').get('base')
    finalUrl =  f'{base}/?id={coin["id"]}&convertId=2781&timeStart=1041379200&timeEnd=1642809600'
    return finalUrl

for coin in coins:
    request = requests.get(constructApiUrl(coin))
    with open(f'./responses/{coin["symbol"]}.json', 'w') as f:
        json.dump(request.json(),f)

