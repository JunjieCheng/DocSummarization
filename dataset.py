import os
import re
import torch
from torch.utils.data import Dataset, DataLoader


class FileDataSet(Dataset):
    def __init__(self, abstract_path, article_path):

        self.abstracts = []
        self.articles = []

        with open(abstract_path, 'r') as abs_file:
            for line in abs_file.readlines():
                self.abstracts.append([int(token) for token in line.split(', ')])

        with open(article_path, 'r') as art_file:
            for line in art_file.readlines():
                self.articles.append([int(token) for token in line.split(', ')])

    def __getitem__(self, index):
        return {'abstract': torch.LongTensor(self.abstracts[index]), 'abstract_length': len(self.abstracts[index]),
                'article': torch.LongTensor(self.articles[index]), 'article_length': len(self.articles[index])}

    def __len__(self):
        return len(self.abstracts)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        abs_max_len = max(map(lambda x: x['abstract'].shape[self.dim], batch))
        art_max_len = max(map(lambda x: x['article'].shape[self.dim], batch))

        abstracts_length = torch.LongTensor(list(map(lambda x: x['abstract_length'], batch)))
        articles_length = torch.LongTensor(list(map(lambda x: x['article_length'], batch)))

        # pad according to max_len
        batch = list(map(lambda x: (pad_tensor(x['abstract'], pad=abs_max_len, dim=self.dim),
                                    pad_tensor(x['article'], pad=art_max_len, dim=self.dim)), batch))

        # stack all
        abstracts = torch.stack(list(map(lambda x: x[0], batch)))
        articles = torch.stack(list(map(lambda x: x[1], batch)))

        return {'abstract': abstracts, 'abstract_length': abstracts_length,
                'article': articles, 'article_length': articles_length}

    def __call__(self, batch):
        return self.pad_collate(batch)


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size).long()])


if __name__ == '__main__':
    fileDataSet = FileDataSet('../Data/train_set/abstracts.txt', '../Data/train_set/articles.txt')
    dataLoader = DataLoader(fileDataSet, batch_size=2, shuffle=False, num_workers=4, collate_fn=PadCollate(dim=0))

    for batch in dataLoader:
        print(batch)
        break
