import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import args
import preprocess
import dataset
import log
from models import Models


def build_model(vocab_size, vocab):
    encoder = Models.RNNEncoder(vocab_size, vocab)
    decoder = Models.RNNDecoder(encoder.embeddings.weight, vocab_size, vocab)

    if args.USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    model = Models.DocSumModel(encoder, decoder, vocab)

    return model


def build_optim(model):
    return optim.SGD(model.parameters(), args.LEARNING_RATE)


def make_criterion(vocab_size):
    weight = torch.ones(vocab_size)
    weight[args.PAD_TOKEN] = 0
    criterion = nn.NLLLoss(weight)

    return criterion


def train_model(model, optim, file_data_set, vocab_size):
    start = time.time()
    criterion = make_criterion(vocab_size)

    lr = args.LEARNING_RATE

    if args.USE_CUDA:
        criterion = criterion.cuda()

    print("Start Training...")

    for epoch in range(args.EPOCH_NUMBER):

        data_loader = DataLoader(file_data_set, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=4,
                                 collate_fn=dataset.PadCollate(dim=0))

        train(model, data_loader, criterion, optim, start, epoch)

        if epoch != 0 and epoch % 20 == 0:
            lr /= 10
            optim.param_groups[0]['lr'] = lr
            torch.save(model.state_dict(), './models/model_epoch_' + str(epoch) + '.pt')


def train(model, data_loader, criterion, optim, start, epoch):
    loss = 0
    loss_total = 0
    trained_number = 0

    for i, batch in enumerate(data_loader):
        input_batch = batch['article']
        input_batch_length = batch['article_length']
        target_batch = batch['abstract']

        if args.USE_CUDA:
            input_batch = input_batch.cuda()
            target_batch = target_batch.cuda()

        input_batch = Variable(input_batch)
        target_batch = Variable(target_batch)

        model.zero_grad()

        outputs, dec_state = model(input_batch, target_batch, input_batch_length)

        scores = outputs.contiguous().view(-1, outputs.size(2))
        target_batch = target_batch[:, 1:].contiguous().view(-1)

        dec_state = Models.RNNDecoderState(dec_state)

        if dec_state is not None:
            dec_state.detach()

        loss = criterion(scores, target_batch)
        loss_total += loss
        loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm(model.parameters(), args.GRADIENT_CLIP)
        optim.step()

        trained_number += input_batch.size()[0]

        print('Epoch %d: %s (%d %.2f%%) %.4f' % (epoch + 1, log.time_since(start, trained_number / len(data_loader)),
                                                 trained_number,
                                                 trained_number / len(data_loader) * 100, loss))


def main():
    fileDataSet = dataset.FileDataSet('./Data/train_set/abstracts.txt', './Data/train_set/articles.txt')

    vocab = preprocess.import_vocab('./Data/dictionary.txt')
    vocab_size = len(vocab.idx2word)

    model = build_model(vocab_size, vocab)
    optim = build_optim(model)

    if args.USE_CUDA:
        model = model.cuda()

    model.train()

    train_model(model, optim, fileDataSet, vocab_size)

    torch.save(model.state_dict(), './models/model.pt')


if __name__ == "__main__":
    main()
