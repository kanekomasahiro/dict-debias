import os
import json
import random
import math
import optim
import model
import itertools
import argparse
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
from nltk.corpus import wordnet, stopwords
import torch
import torch.nn as nn
from torch import cuda
from torch.utils.data import DataLoader
import gensim
from gensim.models import KeyedVectors
from dataset import EmbDataset, EmbDictDataset
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save-prefix', type=str, required=True)
    parser.add_argument('--save-binary', action='store_true')
    parser.add_argument('--embedding', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dictionary', type=str, required=True,
                        help='json format')
    parser.add_argument('--remove-stop-words', action='store_true')
    parser.add_argument('--retrieve-original-case', action='store_true')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=1111)

    args = parser.parse_args()

    return args


def remove_stop_words(dictionary):
    stop_words = set(stopwords.words('english'))
    tmp_dictionary = {v: k - stop_words for (v, k) in dictionary.items()
                      if v not in stop_words}

    return tmp_dictionary


def retrieve_original_case(dictionary):
    tmp_dictionary = defaultdict(list)

    for word, definitions in dictionary.items():
        synsets = wordnet.synsets(word)
        assert(len(definitions) == len(synsets))
        for synset, definition in zip(synsets, definitions):
            cased_lemma = [c for c in synset.lemma_names()
                           if c.lower() == word]
            if len(cased_lemma) == 0:
                cased_lemma = [word]
            for c in cased_lemma:
                tmp_dictionary[c].append(definition)

    return tmp_dictionary


def make_optim(model, optimizer, learning_rate, lr_decay, max_grad_norm):
    model_optim = optim.Optim(optimizer, learning_rate,
                              lr_decay, max_grad_norm)
    model_optim.set_parameters(model.parameters())

    return model_optim


def save_word2vec_format(emb_dict, save_prefix, output):
    emb_num = len(emb_dict)
    emb_size = 300

    with gensim.utils.smart_open(save_prefix / output, 'wb') as fw:
        fw.write(gensim.utils.to_utf8(f'{emb_num} {emb_size}\n'))
        for word, vector in (emb_dict.items()):
            vector = vector.astype(np.float32).tostring()
            fw.write(gensim.utils.to_utf8(word) + b' ' + vector)


def preprocess_dictionary(dictionary, words, args):
    if args.retrieve_original_case:
        dictionary = retrieve_original_case(dictionary)
    words_set = set(words)
    dictionary_old = dictionary
    dictionary = {word: list(key for key in
                            itertools.chain.from_iterable(dictionary[word])
                            if key in words_set) for word in words}
    if args.remove_stop_words:
        dictionary = remove_stop_words(dictionary)
    dictionary = {key: value for key, value in dictionary.items()
                      if len(value) > 0}

    return dictionary


def debiasing_emb(emb, trainer, device, batch_size):

    vocab = [word for word in emb.vocab]
    vector = [emb[word] for word in vocab]
    emb_dict = {}
    trainer.change_model_mode('eval')

    for i in range(math.ceil(len(vocab) / batch_size)):
        words = vocab[i * batch_size: i * batch_size + batch_size]
        batch = np.stack(vector[i * batch_size: i * batch_size + batch_size])
        batch = torch.from_numpy(batch).to(device)
        trainer.encoder.zero_grad()
        hiddens = trainer.encoder(batch)
        hiddens = hiddens.cpu().tolist()
        for word, hidden in zip(words, hiddens):
            emb_dict[word] = np.array(hidden)

    return emb_dict


def main(args):
    config = json.load(open(args.config))

    if args.gpu >= 0:
        device = 'cuda:{}'.format(args.gpu)
        cuda.manual_seed_all(args.seed)
    else:
        device = 'cpu'
    logger.info(f'Device: {device}')

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    valid_size = config['valid_size']
    pre_valid_size = config['pre_valid_size']
    batch_size = config['batch_size']
    pre_batch_size = config['pre_batch_size']
    emb_size= config['emb_size']
    hidden_size = config['hidden_size']
    dropout_rate = config['dropout_rate']
    optimizer = config['optimizer']
    lr = config['lr']
    lr_decay = config['lr_decay']
    max_grad_norm = config['max_grad_norm']
    save_prefix = Path(args.save_prefix)
    save_binary = args.save_binary

    logger.info('Loading word embedding')
    if args.embedding.endswith('bin'):
        binary = True
    else:
        binary = False
    emb = KeyedVectors.load_word2vec_format(args.embedding,
                                        binary=binary)

    logger.info('Loading dictionary')
    dictionary = json.load(open(args.dictionary))
    emb_words = [word for  word in emb.vocab]
    words = [word for word in emb_words if word in dictionary]
    dictionary = preprocess_dictionary(dictionary, words, args)
    words = [word for word in emb_words if word in dictionary]
    random.shuffle(emb_words)
    random.shuffle(words)

    logger.info('Creating dataloader')
    pre_train_words = emb_words[pre_valid_size:]
    pre_valid_words = emb_words[:pre_valid_size]
    logger.info(f'Pre train data size:{len(pre_train_words)}')
    logger.info(f'Pre valid data size:{len(pre_valid_words)}')
    train_words, valid_words = words[valid_size:], words[:valid_size]
    logger.info(f'Train data size:{len(train_words)}')
    logger.info(f'Valid data size:{len(valid_words)}')

    pre_train_dataloader = DataLoader(EmbDataset(pre_train_words, emb),
                                  batch_size=pre_batch_size, shuffle=True)
    pre_valid_dataloader = DataLoader(EmbDataset(pre_valid_words, emb),
                                  batch_size=pre_batch_size, shuffle=False)

    train_dataloader = DataLoader(EmbDictDataset(train_words, emb, dictionary),
                                  batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(EmbDictDataset(valid_words, emb, dictionary),
                                  batch_size=batch_size, shuffle=False)

    logger.info('Creating models')
    encoder = model.Encoder(emb_size, hidden_size, dropout_rate)
    dictionary_decoder = model.Decoder(emb_size, hidden_size, dropout_rate)
    corpus_decoder = model.Decoder(emb_size, hidden_size, dropout_rate)
    encoder = encoder.to(device)
    dictionary_decoder = dictionary_decoder.to(device)
    corpus_decoder = corpus_decoder.to(device)
    encoder_optim = make_optim(encoder, optimizer, lr, lr_decay, max_grad_norm)
    dictionary_decoder_optim = make_optim(dictionary_decoder, optimizer, lr,
                                   lr_decay, max_grad_norm)
    corpus_decoder_optim = make_optim(corpus_decoder, optimizer, lr,
                                    lr_decay, max_grad_norm)
    criterion = nn.MSELoss()
    trainer = Trainer(args, config, encoder,
                      dictionary_decoder, corpus_decoder,
                      encoder_optim, dictionary_decoder_optim,
                      corpus_decoder_optim, criterion, device, save_prefix)

    logger.info('Pre-training dictionary & corpus decoder')
    trainer.run(pre_train_dataloader,
                pre_valid_dataloader, 'pre-train')
    logger.info('Debiasing word embedding')
    emb_dict = debiasing_emb(emb, trainer, device, batch_size)
    logger.info('Saving word embedding')
    save_word2vec_format(emb_dict, save_prefix, 'autoenc_emb.bin')

    logger.info('Training adversarial')
    trainer.run(train_dataloader, valid_dataloader, 'adversarial')
    logger.info('Debiasing word embedding')
    emb_dict = debiasing_emb(emb, trainer, device, batch_size)
    logger.info('Saving word embedding')
    save_word2vec_format(emb_dict, save_prefix, 'debiasing_emb.bin')


if __name__ == "__main__":
    args = parse_args()
    save_prefix = Path(args.save_prefix)
    os.makedirs(save_prefix, exist_ok=True)
    logging.config.fileConfig('logging.conf',
                              defaults={'logfilename':
                                        str(save_prefix / 'train.log')})
    logger = logging.getLogger()

    main(args)
