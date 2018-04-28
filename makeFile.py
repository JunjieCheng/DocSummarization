import os
import subprocess


class Datafile(object):
    def __init__(self):
        data_dir = '/users/cheng/NLP/Data'

        # Create some new directories
        if not os.path.exists(data_dir + '/cnn/tokenized_stories'):
            os.makedirs(data_dir + '/cnn/tokenized_stories')
        if not os.path.exists(data_dir + '/dailymail/tokenized_stories'):
            os.makedirs(data_dir + '/dailymail/tokenized_stories')

        if not os.path.exists(data_dir + '/cnn/abstracts'):
            os.makedirs(data_dir + '/cnn/abstracts')
        if not os.path.exists(data_dir + '/cnn/articles'):
            os.makedirs(data_dir + '/cnn/articles')
        if not os.path.exists(data_dir + '/dailymail/abstracts'):
            os.makedirs(data_dir + '/dailymail/abstracts')
        if not os.path.exists(data_dir + '/dailymail/articles'):
            os.makedirs(data_dir + '/dailymail/articles')

        if not os.path.exists(data_dir + '/cnn/abstracts/train_set'):
            os.mkdir(data_dir + '/cnn/abstracts/train_set')
        if not os.path.exists(data_dir + '/cnn/abstracts/validation_set'):
            os.mkdir(data_dir + '/cnn/abstracts/validation_set')
        if not os.path.exists(data_dir + '/cnn/abstracts/test_set'):
            os.mkdir(data_dir + '/cnn/abstracts/test_set')
        if not os.path.exists(data_dir + '/dailymail/abstracts/train_set'):
            os.mkdir(data_dir + '/dailymail/abstracts/train_set')
        if not os.path.exists(data_dir + '/dailymail/abstracts/validation_set'):
            os.mkdir(data_dir + '/dailymail/abstracts/validation_set')
        if not os.path.exists(data_dir + '/dailymail/abstracts/test_set'):
            os.mkdir(data_dir + '/dailymail/abstracts/test_set')

        if not os.path.exists(data_dir + '/cnn/articles/train_set'):
            os.mkdir(data_dir + '/cnn/articles/train_set')
        if not os.path.exists(data_dir + '/cnn/articles/validation_set'):
            os.mkdir(data_dir + '/cnn/articles/validation_set')
        if not os.path.exists(data_dir + '/cnn/articles/test_set'):
            os.mkdir(data_dir + '/cnn/articles/test_set')
        if not os.path.exists(data_dir + '/dailymail/articles/train_set'):
            os.mkdir(data_dir + '/dailymail/articles/train_set')
        if not os.path.exists(data_dir + '/dailymail/articles/validation_set'):
            os.mkdir(data_dir + '/dailymail/articles/validation_set')
        if not os.path.exists(data_dir + '/dailymail/articles/test_set'):
            os.mkdir(data_dir + '/dailymail/articles/test_set')

        self.dm_single_close_quote = u'\u2019'  # unicode
        self.dm_double_close_quote = u'\u201d'
        self.END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', self.dm_single_close_quote, self.dm_double_close_quote,
                           ")"]

    def tokenize(self, stories_path, tokenized_stories_path):
        # Check if path exists
        assert (os.path.exists(stories_path))
        assert (os.path.exists(tokenized_stories_path))

        stories = os.listdir(stories_path)

        # Generate mapping path
        with open('mapping.txt', 'w') as file:
            for path in stories:
                file.write(
                    "%s \t %s\n" % (os.path.join(stories_path, path), os.path.join(tokenized_stories_path, path)))

        # Export CoreNLP path
        os.environ['CLASSPATH'] = '../stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar'

        # Tokenize stories by CoreNLP
        command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
        subprocess.call(command)

        os.remove("mapping.txt")

        num_orig = len(os.listdir(stories_path))
        num_tokenized = len(os.listdir(tokenized_stories_path))

        if num_orig != num_tokenized:
            print("The tokenized stories directory %s contains %i files, but it should contain the same \
                            number as %s (which has %i files). Was there an error during tokenization?\n"
                  % (tokenized_stories_path, num_tokenized, stories_path, num_orig))

        print("Successfully finished tokenizing %s to %s.\n" % (stories_path, tokenized_stories_path))

    def read_file(self, file_path, filename):
        lines = []

        try:
            with open(file_path + "/" + filename, "r") as file:
                for line in file:
                    lines.append(line.strip())
        except UnicodeDecodeError:
            print(file_path + '/' + filename + ' cannot be converted')
            return []

        return lines

    def fix_missing_period(self, line):
        """Adds a period to a line that is missing a period"""
        if "@highlight" in line:
            return line
        if line == "":
            return line
        if line[-1] in self.END_TOKENS:
            return line

        # print line[-1]
        return line + " ."

    def generate(self, tokenized_stories_path, article_path, abstract_path):
        # Check if path exists
        assert (os.path.exists(tokenized_stories_path))

        stories_filenames = os.listdir(tokenized_stories_path)

        train_set_index = int(len(stories_filenames) * 0.6)
        validation_set_index = int(len(stories_filenames) * 0.8)

        dir_name = "train_set/"

        for i, story_name in enumerate(stories_filenames):
            lines = self.read_file(tokenized_stories_path, story_name)

            if i == train_set_index:
                dir_name = "validation_set/"
            elif i == validation_set_index:
                dir_name = "test_set/"

            if len(lines) == 0:
                continue

            # Lowercase everything
            lines = [line.lower() for line in lines]

            # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many
            # image captions don't end in periods; consequently they end up in the body of the article as run-on
            # sentences)
            lines = [self.fix_missing_period(line) for line in lines]

            # Separate out article and abstract sentences
            article_lines = []
            highlights = []
            next_is_highlight = False

            for idx, line in enumerate(lines):
                if line == "":
                    continue  # empty line
                elif line.startswith("@highlight"):
                    next_is_highlight = True
                elif next_is_highlight:
                    highlights.append(line)
                    next_is_highlight = False
                else:
                    article_lines.append(line)

            # article is empty
                if len(article_lines) == 0:
                    continue

            with open(article_path + "/" + dir_name + story_name, "w") as article:
                for line in article_lines:
                    article.write(line)
                    article.write(' ')

            with open(abstract_path + "/" + dir_name + story_name, "w") as abstract:
                for line in highlights:
                    abstract.write(line)
                    abstract.write(' ')


if __name__ == "__main__":
    datafile = Datafile()

    print("Tokenizing data...")
    datafile.tokenize("/users/cheng/NLP/Data/cnn/stories", "/users/cheng/NLP/Data/cnn/tokenized_stories")
    datafile.tokenize("/users/cheng/NLP/Data/dailymail/stories", "/users/cheng/NLP/Data/dailymail/tokenized_stories")

    print("Generating files...")
    datafile.generate("/users/cheng/NLP/Data/cnn/tokenized_stories",
                      "/users/cheng/NLP/Data/cnn/articles", "/users/cheng/NLP/Data/cnn/abstracts")
    datafile.generate("/users/cheng/NLP/Data/dailymail/tokenized_stories",
                      "/users/cheng/NLP/Data/dailymail/articles", "/users/cheng/NLP/Data/dailymail/abstracts")