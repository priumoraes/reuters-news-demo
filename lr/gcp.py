from google.cloud import language
import os

def get_entities(content):
    client = language.LanguageServiceClient()
    document = language.types.Document(
        content=content,
        type=language.enums.Document.Type.PLAIN_TEXT,
    )
    response = client.analyze_entities(document=document, encoding_type='UTF32')
    return response.entities


def get_categories(content):
    client = language.LanguageServiceClient()

    document = language.types.Document(
        content=content,
        type=language.enums.Document.Type.PLAIN_TEXT
    )
    response = client.classify_text(document)
    return response.categories


def get_syntax(content):
    client = language.LanguageServiceClient()

    # Instantiates a plain text document.
    document = language.types.Document(
        content=content,
        type=language.enums.Document.Type.PLAIN_TEXT)

    tokens = client.analyze_syntax(document).tokens

    pos_tag = ('UNKNOWN', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM',
               'PRON', 'PRT', 'PUNCT', 'VERB', 'X', 'AFFIX')
    return tokens, pos_tag

def get_prediction(data_dir, filename):
    cmd = "gcloud ml-engine predict \
                --model lr_reuters \
                --version v1 \
                --json-instances \
                "+data_dir+filename+" \
                > ../data/output/prediction.txt"
    try:
        os.system(cmd)
        #logging.info('Prediction was successful.')
    except:
        #logging.error('Could not predict file '+filename)
        print('Could not predict file '+filename)
