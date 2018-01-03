import gcp
import utils
import scrapper
import json
import re
import os
import operator
import shutil


def predict(result):
    i = 0
    index_key = {}
    predictions = {}
    DATA_DIR = '../data/output/features/'
    shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)
    PRED_PATH = '../data/output/prediction.txt'

    #Create json representation of feature vector for prediction
    for key, value in result.items():
        with open(DATA_DIR+'test_'+str(i)+'.json', 'wb') as outfile:
            if ( value is not None ):
                json.dump(value, outfile)
        index_key[str(i)] = key
        i += 1

    for filename in os.listdir(DATA_DIR):
        gcp.get_prediction(DATA_DIR, filename)
        prob = utils.get_prob(PRED_PATH)
        p = re.compile('\d+')
        index = p.findall(filename)[0]
        predictions[index_key[index]] = prob
        print('Predicting entity: '+index_key[index])

    labels = get_top_labels(predictions)
    return labels

def get_top_labels(preds):
    i = 0
    labels = []
    sorted_preds = sorted(preds.items(),
        key=operator.itemgetter(1),
        reverse=True)
    #print(sorted_preds)
    labels = sorted_preds[:3]
    #logging.debug('Top 3 labels: '+str(labels))
    return labels


def infer(url):
    try:
        content, title, keywords = scrapper.getURLContent(url)
        try:
            entities = utils.process_entities(content, title)
            try:
                pos_tags1 = predict(entities)
                pos_tags2, neg_tags = utils.get_tags(entities, title, 3)
            except:
                pos_tags2, neg_tags = utils.get_tags(entities, title, 3)
        except:
            print("Unable to get entities.")
    except:
        content = "Unable to parse the article."
        pos_tags = "No tags found."
        title = "No title found."
        keywords = "No keywords found."
    return content, pos_tags2, title, keywords

def main():
    url = 'http://www.reuters.com/article/companyNewsAndPR/idUSTP13157220070102'
    infer(url)

if __name__== "__main__":
    main()
