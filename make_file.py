import os
import re

abstracts_list = os.listdir('../Data/cnn/abstracts/train_set')
articles_list = os.listdir('../Data/cnn/articles/train_set')

r = re.compile('[0-9a-z]*\.story')

abstracts_list_list = list(filter(r.match, abstracts_list))
articles_list = list(filter(r.match, articles_list))

abstracts_list.sort()
articles_list.sort()

count = 0

src = open("../Data/src.txt", 'w')
tgt = open("../Data/tgt.txt", 'w')

for article_name, abstract_name in zip(articles_list, abstracts_list):

    if count == 4:
        break

    article = open('../Data/cnn/articles/train_set/' + article_name, 'r').readlines()
    abstract = open('../Data/cnn/abstracts/train_set/' + abstract_name, 'r').readlines()

    article_words = article[0].split()
    abstract_words = abstract[0].split()

    if len(article_words) > 400 or len(abstract_words) > 100:
        continue
    if len(article_words) == 0 or len(abstract_words) == 0:
        continue

    src.write(article[0])
    src.write("\n")
    tgt.write(abstract[0])
    tgt.write("\n")

    count += 1

src.close()
tgt.close()




