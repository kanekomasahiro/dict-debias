# Dictionary-based Debiasing of Pre-trained Word Embeddings

[Masahiro Kaneko](https://sites.google.com/view/masahirokaneko/english?authuser=0), [Danushka Bollegala](http://danushka.net/)


Code and debiased word embeddings for the paper: "Dictionary-based Debiasing of Pre-trained Word Embeddings" (In EACL 2021). If you use any part of this work, make sure you include the following citation:

```
@inproceedings{kaneko-bollegala-2021-dict,
    title={Dictionary-based Debiasing of Pre-trained Word Embeddings},
    author={Masahiro Kaneko and Danushka Bollegala},
    booktitle = {Proc. of the 16th European Chapter of the Association for Computational Linguistics (EACL)},
    year={2021}
}
```


### Requirements
- python==3.7.2
- torch==1.6.0
- gensim==3.7.3
- numpy==1.19.1
- nltk==3.4


### To debias your word embeddinngs
```
cd src
python train.py --embedding path/to/your/embeddings --dictionary ../data/dict_wn.json --config config/hyperparameter.json --save-prefix path/to/save/directory --gpu id --save-binary

```
Output is a debiased binary word embeddings saved in `--save-prefix`


### Our debiased word embeddings

You can directly download our debiased [word embeddings](https://drive.google.com/drive/folders/1qyFpqX7Wxz4uJj047enOOJbrbMGEMczK?usp=sharing).


### License
See the LICENSE file.
