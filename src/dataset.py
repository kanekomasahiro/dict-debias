import torch
from collections import defaultdict


class EmbDataset(torch.utils.data.Dataset):

    def __init__(self, words, emb):
        self.words = words
        self.emb = emb


    def __len__(self):
        return len(self.words)


    def __getitem__(self, idx):
        word = self.words[idx]
        word_emb = self.emb[word]

        return word_emb


class EmbDictDataset(torch.utils.data.Dataset):

    def __init__(self, words, emb, dict_def):
        self.words = words
        self.emb = emb
        self.dict_def = dict_def
        self.p = self.calculate_unigram_prob(self.dict_def)


    def __len__(self):
        return len(self.words)


    def calculate_unigram_prob(self, dict_def):
        total_num = 0
        count_d = defaultdict(int)
        for words in dict_def.values():
            for word in words:
                count_d[word] += 1
                total_num += 1

        return {key: value / total_num for key, value in count_d.items()}


    def create_SIF(self, words, a=0.0001):
        return sum([(a / (a + self.p[word])) * self.emb[word]
                    for word in words]) / len(words)


    def __getitem__(self, idx):
        word = self.words[idx]
        word_emb = self.emb[word]
        dict_emb = self.create_SIF(self.dict_def[word])

        return word_emb, dict_emb
