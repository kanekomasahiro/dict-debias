import torch
import os
import logging


logger = logging.getLogger()

class Trainer(object):

    def __init__(self, args, config, encoder,
                 dictionary_decoder, corpus_decoder,
                 encoder_optim, dictionary_decoder_optim,
                 corpus_decoder_optim, criterion, device,
                 save_prefix):

        self.encoder = encoder
        self.dictionary_decoder = dictionary_decoder
        self.corpus_decoder = corpus_decoder
        self.encoder_optim = encoder_optim
        self.dictionary_decoder_optim = dictionary_decoder_optim
        self.corpus_decoder_optim = corpus_decoder_optim
        self.criterion = criterion

        self.config = config
        self.args = args

        self.save_prefix = save_prefix
        self.device = device
        self.dictionary_weight = config['dictionary_weight']
        self.corpus_weight = config['corpus_weight']
        self.adversarial_weight = config['adversarial_weight']


    def set_hyperparameter(self, model_type):
        self.epoch = self.config[f'{model_type}_epoch']
        self.early_stop = self.config[f'{model_type}_early_stop']


    def change_model_mode(self, mode):
        assert mode in ['train', 'eval'], 'Please select mode [train, eval]'
        if mode == 'train':
            self.encoder.train()
            self.dictionary_decoder.train()
            self.corpus_decoder.train()
            self.criterion.train()
        elif mode == 'eval':
            self.encoder.eval()
            self.dictionary_decoder.eval()
            self.corpus_decoder.eval()
            self.criterion.eval()


    def dictionary_step(self, hidden, dict_emb):
        dictionary_output = self.dictionary_decoder(hidden)
        loss = self.criterion(dictionary_output, dict_emb)
        return dictionary_output, loss


    def corpus_step(self, word_emb):
        hidden = self.encoder(word_emb)
        corpus_output = self.corpus_decoder(hidden)
        loss = self.criterion(corpus_output, word_emb)
        return corpus_output, hidden, loss


    def step(self, dataloader, model_type, mode):

        self.change_model_mode(mode)

        total_dictionary_loss = 0
        total_corpus_loss = 0
        total_adversarial_loss = 0
        total_num = 0

        for batch in dataloader:
            if model_type == 'pre-train':
                word_emb = batch
                word_emb = word_emb.to(self.device)
            else:
                word_emb, dict_emb = batch
                word_emb = word_emb.to(self.device)
                dict_emb = dict_emb.to(self.device)
                bias_emb = word_emb - \
                           ((word_emb * dict_emb).sum(1, keepdim=True) * \
                           dict_emb / torch.norm(dict_emb))

            _, hidden, corpus_loss = self.corpus_step(word_emb)

            if model_type == 'pre-train':
                if mode == 'train':
                    self.encoder_optim.zero_grad()
                    self.corpus_decoder_optim.zero_grad()
                    corpus_loss.backward()
                    self.encoder_optim.step()
                    self.corpus_decoder_optim.step()
                total_corpus_loss += corpus_loss.item()

            elif model_type == 'adversarial':
                _, dictionary_loss = self.dictionary_step(hidden, bias_emb)
                adversarial_loss = torch.mean((hidden * self.encoder(bias_emb)).sum(1, keepdim=True)**2)
                loss = self.dictionary_weight * dictionary_loss \
                     + self.corpus_weight * corpus_loss \
                     + self.adversarial_weight * adversarial_loss
                if mode == 'train':
                    self.encoder_optim.zero_grad()
                    self.dictionary_decoder_optim.zero_grad()
                    self.corpus_decoder_optim.zero_grad()
                    loss.backward()
                    self.encoder_optim.step()
                    self.dictionary_decoder_optim.step()
                    self.corpus_decoder_optim.step()

                total_dictionary_loss += dictionary_loss.item()
                total_adversarial_loss += adversarial_loss.item()
            total_corpus_loss += corpus_loss.item()

            total_num += len(word_emb)

        return total_dictionary_loss / total_num, \
               total_corpus_loss / total_num, \
               total_adversarial_loss / total_num


    def run(self, train_dataloader, valid_dataloader, model_type):
        assert model_type in ['pre-train', 'adversarial'], \
        'Please select model_type [pre-train, adversarial]'

        self.best_epoch = 0
        self.best_loss = float('inf')
        self.set_hyperparameter(model_type)

        for epoch in range(1, self.epoch + 1):
            logger.info(f'Epoch: {epoch}')
            train_dictionary_loss, train_corpus_loss, train_adversarial_loss = self.step(train_dataloader, model_type, 'train')
            valid_dictionary_loss, valid_corpus_loss, valid_adversarial_loss = self.step(valid_dataloader, model_type, 'eval')
            logger.info(f'Train corpus loss: {train_corpus_loss}')
            logger.info(f'Valid corpus loss: {valid_corpus_loss}')

            if model_type == 'pre-train':
                valid_loss = valid_corpus_loss
            elif model_type == 'adversarial':
                train_loss = train_dictionary_loss * self.dictionary_weight + \
                             train_corpus_loss * self.corpus_weight + \
                             train_adversarial_loss * self.adversarial_weight
                valid_loss = valid_dictionary_loss * self.dictionary_weight + \
                             valid_corpus_loss * self.corpus_weight + \
                             valid_adversarial_loss * self.adversarial_weight
                logger.info(f'Train dictionary loss: {train_dictionary_loss}')
                logger.info(f'Valid dictionary loss: {valid_dictionary_loss}')
                logger.info(f'Train adversarial loss: {train_adversarial_loss}')
                logger.info(f'Valid adversarial loss: {valid_adversarial_loss}')
                logger.info(f'Train loss: {train_loss}')
                logger.info(f'Valid loss: {valid_loss}')

            if valid_loss < self.best_loss:
                self.best_epoch = epoch
                self.best_loss = valid_loss
                encoder_states = self.encoder.state_dict()
                dictionary_decoder_states = self.dictionary_decoder.state_dict()
                corpus_decoder_states = self.corpus_decoder.state_dict()
                checkpoint = {'encoder': encoder_states,
                              'dictionary_decoder': dictionary_decoder_states,
                              'corpus_decoder': corpus_decoder_states,
                              'encoder_optim': self.encoder_optim,
                              'dictionary_decoder_optim': self.dictionary_decoder_optim,
                              'corpus_decoder_optim': self.corpus_decoder_optim,
                              'config': self.config,
                              'args': self.args}
                torch.save(checkpoint, self.save_prefix / 'model_checkpoint')
            else:
                self.encoder_optim.decay_lr()
                self.dictionary_decoder_optim.decay_lr()
                self.corpus_decoder_optim.decay_lr()

            if epoch - self.best_epoch >= self.early_stop:
                break

        logger.info(f'Best epoch: {self.best_epoch}')
        checkpoint = torch.load(self.save_prefix / 'model_checkpoint')
        torch.save(checkpoint,
                   self.save_prefix / f'{model_type}_best_checkpoint')
        os.remove(self.save_prefix /  'model_checkpoint')
