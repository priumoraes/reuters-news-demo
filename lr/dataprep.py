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
                #category_score = 0.0
                #similarity_category = 0.0
                #similarity_title = 0.0
                try:
                    entities = utils.process_entities(content, title)
                    #categories = gcp.get_categories(content)
                    #tokens, pos_tag = gcp.get_syntax(content)
                except:
                    print("Unable to process content and title.")
                    continue
                token_pos = {}
                #for token in tokens:
                #    token_pos[token.text.content.lower()] = \
                #        pos_tag[token.part_of_speech.tag]
                pos_tags, neg_tags = utils.get_tags(entities, title, 3)

                for tag in pos_tags:
                    #for category in categories:
                    #    if (tag.lower() in category.name.lower()):
                    #        category_score += category.confidence
                    entity = entities[tag]
                    pos_token = ''
                    #if (tag in token_pos):
                    #    pos_token = token_pos[tag.lower()]
                    writer.writerow([entity['salience'],entity['entity_type'], \
                        entity['mentions'],entity['noun_type'], \
                        entity['pos_title'], \
                        'yes'])
                    #category_score = 0.0
                    #similarity_category = 0.0

                for tag in neg_tags:
                    #for category in categories:
                    #    if (tag.lower() in category.name.lower()):
                    #        category_score += category.confidence
                    entity = entities[tag]
                    pos_token = ''
                    #if (tag in token_pos):
                    #    pos_token = token_pos[tag.lower()]
                    writer.writerow([entity['salience'],entity['entity_type'], \
                        entity['mentions'],entity['noun_type'], \
                        entity['pos_title'], \
                        'no'])
                    #category_score = 0.0
                    #similarity_category = 0.0
        csvfile.close()
    f.close()

def main():
    generate_feature_vector()

if __name__== "__main__":
    main()
