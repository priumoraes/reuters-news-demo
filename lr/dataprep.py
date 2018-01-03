import inference
import gcp
import utils
import csv

def generate_feature_vector():

    with open('../data/train3.csv', 'wb') as f:
        writer = csv.writer(f)
        with open('../data/news/train_data.csv', 'r') as csvfile:
            newsreader = csv.reader(csvfile, delimiter='|')

            for row in newsreader:
                content = row[4]
                title = row[2]
                try:
                    entities = utils.process_entities(content, title)
                except:
                    #logging.error("Unable to process content and title.")
                    continue
                token_pos = {}
                pos_tags, neg_tags = utils.get_tags(entities, title, 3)

                for tag in pos_tags:
                    entity = entities[tag]
                    pos_token = ''
                    writer.writerow([entity['salience'],entity['entity_type'], \
                        entity['mentions'],entity['noun_type'], \
                        entity['pos_title'], \
                        'yes'])

                for tag in neg_tags:
                    entity = entities[tag]
                    pos_token = ''
                    writer.writerow([entity['salience'],entity['entity_type'], \
                        entity['mentions'],entity['noun_type'], \
                        entity['pos_title'], \
                        'no'])
        csvfile.close()
    f.close()

def main():
    generate_feature_vector()

if __name__== "__main__":
    main()
