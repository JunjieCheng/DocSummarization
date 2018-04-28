import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import dataset
import preprocess
import args
from models import Models


def build_model(vocab_size, vocab):
    encoder = Models.RNNEncoder(vocab_size, vocab)
    decoder = Models.RNNDecoder(encoder.embeddings.weight, vocab_size, vocab)

    if args.USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    model = Models.DocSumModel(encoder, decoder, vocab)

    return model


def generate(model, file_data_set, vocab_size):

    data_loader = DataLoader(file_data_set, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=4,
                             collate_fn=dataset.PadCollate(dim=0))

    for i, batch in enumerate(data_loader):
        input_batch = batch['article']
        input_batch_length = batch['article_length']
        target_batch = batch['abstract']

        if args.USE_CUDA:
            input_batch = input_batch.cuda()
            target_batch = target_batch.cuda()

        input_batch = Variable(input_batch)
        target_batch = Variable(target_batch)

        outputs, dec_state = model(input_batch, target_batch, input_batch_length, test=True)


def main():
    fileDataSet = dataset.FileDataSet('../Data/test_set/abstracts.txt', '../Data/test_set/articles.txt')

    vocab = preprocess.import_vocab('../Data/dictionary.txt')
    vocab_size = len(vocab.idx2word)

    model = build_model(vocab_size, vocab)
    model.load_state_dict(torch.load('models/model_100_epochs.pt', map_location='cpu'))

    if args.USE_CUDA:
        model = model.cuda()

    model.eval()

    generate(model, fileDataSet, vocab_size)


if __name__ == "__main__":
    main()
