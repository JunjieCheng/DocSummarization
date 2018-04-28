import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import time

import model
import preprocess
import dataset
import log
import args


def train(input_batch, target_batch, input_batch_length, target_batch_length, model, optimizer, criterion):
    optimizer.zero_grad()

    loss = 0

    decoder_hiddens = model(input_batch, input_batch_length, target_batch, target_batch_length)

    if args.USE_CUDA:
        target_batch = target_batch.cuda()

    # eliminate <SOS>
    target_batch = Variable(target_batch[:, 1:])

    decoder_hiddens = decoder_hiddens.contiguous().view(-1, vocab_size)
    target_batch = target_batch.contiguous().view(-1)

    loss += criterion(decoder_hiddens, target_batch)
    loss.backward()

    torch.nn.utils.clip_grad_norm(model.parameters(), args.GRADIENT_CLIP)
    optimizer.step()

    return loss.data[0]


def train_iteration(model, file_data_set, learning_rate=0.01):
    start = time.time()
    loss_total = 0
    sample_processed = 0

    criterion = nn.NLLLoss()

    print("Start Training...")

    for epoch in range(args.EPOCH_NUMBER):

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        data_loader = DataLoader(file_data_set, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=4,
                                 collate_fn=dataset.PadCollate(dim=0))

        for i, batch in enumerate(data_loader):
            input_batch = batch['article']
            input_batch_length = batch['article_length']
            target_batch = batch['abstract']
            target_batch_length = batch['abstract_length']

            loss = train(input_batch, target_batch, input_batch_length, target_batch_length, model, optimizer,
                         criterion)

            loss_total += loss

            sample_processed += args.BATCH_SIZE

        loss_avg = loss_total / len(data_loader)
        loss_total = 0

        print('Epoch %d: %s (%d %.2f%%) %.4f' % (
            epoch + 1, log.time_since(start, sample_processed / len(file_data_set) / args.EPOCH_NUMBER),
            sample_processed, sample_processed / len(file_data_set) / args.EPOCH_NUMBER * 100,
            loss_avg))

        if epoch != 0 and epoch % 10 == 0:
            learning_rate /= 10

    with open('lr.txt', 'w') as file:
        file.write(str(float(learning_rate)))


if __name__ == '__main__':

    fileDataSet = dataset.FileDataSet('../Data/train_set/abstracts.txt', '../Data/train_set/articles.txt')

    vocab = preprocess.import_vocab('../Data/dictionary.txt')
    vocab_size = len(vocab.idx2word)
    learning_rate = args.LEARNING_RATE

    model = model.DocSummarizationModel(vocab_size, args.HIDDEN_SIZE)

    if args.USE_CUDA:
        model = model.cuda()

    model.train()

    train_iteration(model, fileDataSet, learning_rate=learning_rate)

    torch.save(model.state_dict(), './models/model_attn.pt')
