import gcp
import utils
import scrapper
import json
import re
import os
import operator


def get_position_title(entity, title):
    if (entity.name.lower() in title.lower()):
        return title.lower().find(entity.name.lower())
    elif (entity.mentions[0].text.content.lower() in title.lower()):
        return title.lower().find(entity.mentions[0].text.content.lower())
    else:
        return 0

def get_prob(filename):
    with open(filename, 'r') as f:
        pred = f.read()
    return pred.split('\n')[1].split()[2][1:-1]

def process_entities(content, title):
    entities = gcp.get_entities(content)

    result = {}
    num_entities = len(entities)
    i = 0
    #logging.info(str(num_entities)+' entities identified.')
    for entity in entities:
        # Turn the categories into a dictionary of the form:
        # {entity.name: info}
        info = {}
        info['salience'] = entity.salience
        info['entity_type'] = entity.type
        info['mentions'] = len(entity.mentions)
        info['noun_type'] = entity.mentions[0].type
        info['pos_title'] = utils.get_position_title(entity, title)
        #logging.debug('Entity: '+entity.name+\
        #              ' has annotations: '+str(info))

        result[entity.name] = info
        if ( i >= 10 ): break
        i += 1
    return result

def get_tags(results, title, num_labels):
    pos_labels = []
    neg_labels = []
    salience = {}
    boost = 0

    for entity, info in results.iteritems():
        if ( info['noun_type'] == 2 ):
            boost += 0.2
        if ( entity.lower() in title.lower() ):
            boost += 0.4
        salience[entity] = float(info['salience'] * info['mentions'] + boost)
        boost = 0
    salience = sorted(salience.iteritems(), key=lambda (k,v): (v,k), reverse=True)

    top_items = salience[:num_labels]
    bottom_items = salience[-num_labels:]
    for item in top_items:
        pos_labels.append(item[0])
    for item in bottom_items:
        neg_labels.append(item[0])

    return pos_labels, neg_labels
