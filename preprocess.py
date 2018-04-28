import os
import re
import collections
import args


class Vocabulary(object):

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []


class Preprocess(object):

    def __init__(self):
        self.dictionary = Vocabulary()
        self.counter = collections.Counter()
        self.count = 0

    def load_dict(self, cnn_abs, daily_mail_abs, cnn_art, daily_mail_art):
        assert os.path.exists(cnn_abs)
        assert os.path.exists(daily_mail_abs)
        assert os.path.exists(cnn_art)
        assert os.path.exists(daily_mail_art)

        print('Loading dictionary...')

        for abs_dir, art_dir in zip([cnn_abs, daily_mail_abs], [cnn_art, daily_mail_art]):
            abs_list = os.listdir(abs_dir)
            art_list = os.listdir(art_dir)

            r = re.compile('[0-9a-z]*\.story')

            abs_list = list(filter(r.match, abs_list))
            art_list = list(filter(r.match, art_list))

            assert len(abs_list) == len(art_list)

            for abs_name, art_name in zip(abs_list, art_list):
                with open(abs_dir + '/' + abs_name, 'r') as abstract, open(art_dir + '/' + art_name, 'r') as article:

                    abs_words = abstract.readline().split()
                    art_words = article.readline().split()

                    # skip abstracts that are too long
                    if len(abs_words) > args.ABSTRACT_LENGTH_THRESHOLD or len(abs_words) == 0:
                        continue
                    if len(art_words) > args.ARTICLE_LENGTH_THRESHOLD or len(art_words) == 0:
                        continue

                    self.counter.update(abs_words)
                    self.counter.update(art_words)

                    self.count += 1

        # add padding word
        self.dictionary.word2idx[args.PAD] = args.PAD_TOKEN
        self.dictionary.idx2word.append(args.PAD)

        # add unknown word
        self.dictionary.word2idx[args.UNK] = args.UNK_TOKEN
        self.dictionary.idx2word.append(args.UNK)

        # add start word
        self.dictionary.word2idx[args.SOS] = args.SOS_TOKEN
        self.dictionary.idx2word.append(args.SOS)

        # add end word
        self.dictionary.word2idx[args.EOS] = args.EOS_TOKEN
        self.dictionary.idx2word.append(args.EOS)

        for i, (key, val) in enumerate(self.counter.most_common()):
            self.dictionary.word2idx[key] = i + 4
            self.dictionary.idx2word.append(key)

        print('Loading dictionary completed!')
        return self.dictionary

    def export_dict(self, dict_dir):
        assert os.path.exists(dict_dir)
        print('Exporting dictionary...')

        with open(dict_dir + '/dictionary.txt', 'w') as dict_file:
            for i, word in enumerate(self.dictionary.idx2word):
                if i > args.INDEX_THRESHOLD:
                    break

                dict_file.write("%s %d\n" % (word, i))

        print('Exporting dictionary completed!')

    def generate_files(self, cnn_abs, daily_mail_abs, cnn_art, daily_mail_art, data_dir):

        abs_file = open('%s/abstracts.txt' % data_dir, 'w')
        art_file = open('%s/articles.txt' % data_dir, 'w')

        for abs_dir, art_dir in zip([cnn_abs, daily_mail_abs], [cnn_art, daily_mail_art]):
            abs_list = os.listdir(abs_dir)
            art_list = os.listdir(art_dir)

            r = re.compile('[0-9a-z]*.story')

            abs_list = list(filter(r.match, abs_list))
            art_list = list(filter(r.match, art_list))

            assert len(abs_list) == len(art_list)

            for abs_name, art_name in zip(abs_list, art_list):
                with open(abs_dir + '/' + abs_name, 'r') as abstract, open(art_dir + '/' + art_name, 'r') as article:

                    abs_words = abstract.readline().split()
                    art_words = article.readline().split()

                    # skip abstracts that are too long
                    if len(abs_words) > args.ABSTRACT_LENGTH_THRESHOLD or len(abs_words) == 0:
                        continue
                    if len(art_words) > args.ARTICLE_LENGTH_THRESHOLD or len(art_words) == 0:
                        continue

                    abs_file.write(str(args.SOS_TOKEN))

                    for word in abs_words:
                        abs_file.write(", ")

                        try:
                            if self.dictionary.word2idx[word] <= args.INDEX_THRESHOLD:
                                abs_file.write(str(self.dictionary.word2idx[word]))
                            else:
                                abs_file.write(str(args.UNK_TOKEN))
                        except KeyError:
                            abs_file.write(str(args.UNK_TOKEN))

                    abs_file.write(", ")
                    abs_file.write(str(args.EOS_TOKEN))
                    abs_file.write("\n")
                    art_file.write(str(args.SOS_TOKEN))

                    for word in art_words:
                        art_file.write(", ")

                        try:
                            if self.dictionary.word2idx[word] <= args.INDEX_THRESHOLD:
                                art_file.write(str(self.dictionary.word2idx[word]))
                            else:
                                art_file.write(str(args.UNK_TOKEN))
                        except KeyError:
                            art_file.write(str(args.UNK_TOKEN))

                    art_file.write(", ")
                    art_file.write(str(args.EOS_TOKEN))
                    art_file.write("\n")

        abs_file.close()
        art_file.close()


def import_vocab(dict_name):
    assert os.path.exists(dict_name)

    print('Importing dictionary...')
    dict = Vocabulary()

    with open(dict_name, 'r') as dict_file:
        for line in dict_file.readlines():
            word, idx = line.split(' ')
            dict.idx2word.append(word)
            dict.word2idx[word] = int(idx)

    return dict


if __name__ == "__main__":
    data = Preprocess()

    # load train set
    # data.load_dict('../Data/cnn/abstracts/test_set',
    #                '../Data/dailymail/abstracts/test_set',
    #                '../Data/cnn/articles/test_set',
    #                '../Data/dailymail/articles/test_set')

    # data.export_dict('../Data/')

    data.dictionary = import_vocab('../Data/dictionary.txt')

    data.generate_files('../Data/cnn/abstracts/test_set',
                        '../Data/dailymail/abstracts/test_set',
                        '../Data/cnn/articles/test_set',
                        '../Data/dailymail/articles/test_set',
                        '../Data/test_set')
