from lxml import html, etree
import requests
import re

import csv
import json
import random

def striphtml(data):
    p = re.compile(r'<.*?>')
    data = p.sub('', data)
    p = re.compile(r'(&#[0-9]+;)')
    string = p.sub('', data)
    p = re.compile(r'\>')
    string = p.sub('', string)
    p = re.compile(r'\\')
    string = p.sub('', string)
    p = re.compile(r'\s\s+')
    string = p.sub(' ', string)
    return string

def stripDatareact(data):
    p = re.compile(r'(data-reactid=")[0-9]+(">)')
    return p.sub('', data)

def getURLContent(url):
    page = requests.get(url)
    tree = html.fromstring(page.content)

    #get keywords and title of article
    keywords = tree.xpath('//meta[@name="news_keywords"]')
    title = tree.xpath('//meta[@name="analyticsAttributes.contentTitle"]')
    content_pattern = re.compile('content="(.+?)"')
    keywords = content_pattern.findall(str(etree.tostring(keywords[0])))[0]
    title = content_pattern.findall(str(etree.tostring(title[0])))[0]

    renderable = tree.xpath('//div[@class="renderable"]')

    renderable = etree.tostring(renderable[1], pretty_print=True)
    pattern1 = re.compile('<p(.+?)</p>')
    pattern2 = re.compile('\(Reuters\)(.*)')
    paragraphs = pattern1.findall(str(renderable))
    paragraphs = ' '.join(paragraphs)
    paragraphs = pattern2.findall(paragraphs)
    paragraphs = ' '.join(paragraphs)
    return stripDatareact(striphtml(paragraphs)), title, keywords


def main():
    with open('../data/news/content1.csv', 'w') as f:
        with open('../data/metadata/reuters.csv', 'r', encoding='utf-8', errors='ignore') as csvfile:
            newsreader = csv.reader(csvfile)
            i = 0
            #filenumber = 1
            all_news = []
            for row in newsreader:
                #print(row[0])
                fields = row[0].split('\t')
                news = {}
                if len(fields) == 4:
                    id = fields[1].replace('"','')
                    clean_id = id.split()[0]
                    title = fields[2].replace('"','')
                    url = fields[3].replace('"','')
                    news['id'] = id
                    #print(id)
                    try:
                        content, title, keywords = getURLContent(url).replace('"','')
                        content_list = content.split()
                        random_tag1 = random.choice(content_list)
                        random_tag2 = random.choice(content_list)
                        random_tag3 = random.choice(content_list)
                        random_tags = random_tag1+', '+random_tag2+', '+random_tag3
                        line = id+' | '+clean_id+' | '+title+' | '+url+' | \
                            '+content+' | '+random_tags+ ' | '+keywords
                        f.write(line)
                        f.write('\n')
                    except:
                        print('Unexpected error when scrapping url: '+ url)

                else:
                    continue
                """
                i += 1
                if i == 500:
                    break
                """
            csvfile.close()
        f.close()


if __name__== "__main__":
    main()
