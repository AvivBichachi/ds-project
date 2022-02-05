import json
import csv
from os import listdir
from os.path import isfile, join

jsonsPath = './responses'


def loadJSON(fileName):
    file = open(f"{jsonsPath}/{fileName}")
    data = json.load(file)
    return data


def makeCsvsFromResponses(jsonFiles):
    for file in jsonFiles:
        data = loadJSON(file)["data"]
        #                 "open": 0.8320050115,
        #                 "high": 1.3134865957,
        #                 "low": 0.6941868976,
        #                 "close": 0.9510539576,
        #                 "volume": 87364276.1190757,
        #                 "marketCap": 0.0,
        #                 "timestamp": "2020-04-10T23:59:59.999Z"
        header = ['Time', 'Open', 'High', 'Low',
                  'Close', 'Volume', 'Market Cap']
        dataToWrite = [header]
        with open(f'./csvs/{data["symbol"]}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            # extract data
            for row in data["quotes"]:
                quote = row["quote"]
                dataToWrite.append([quote["timestamp"].split("T")[0],
                                    quote["open"], quote["high"],
                                    quote["low"], quote["close"],
                                    quote["volume"], quote["marketCap"]])
            # write the data
            writer.writerows(dataToWrite)


if __name__ == "__main__":
    jsonFiles = [f for f in listdir(jsonsPath) if isfile(join(jsonsPath, f))]
    makeCsvsFromResponses(jsonFiles)
