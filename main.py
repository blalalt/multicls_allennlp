import torch
import torch.nn as nn
import logging
from torch import optim
from settings import config
from models import BaseModelWithoutKnowledge
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

logging.basicConfig(handlers=[logging.FileHandler("train.log", encoding="utf-8", mode='w')],
                    level=logging.DEBUG)

cuda_device = 0 if torch.cuda.is_available() else -1

# 1. 准备数据
from embeddings import get_token_utils, get_embedder
from dataloaders import ReutersDataSetReader, NewsGroupsDataSetReader

token_indexers, tokenizer = get_token_utils()
# reader = ReutersDataSetReader(tokenizer=tokenizer,  # TODO: token_indexer 的 key
#                               token_indexers={'tokens': token_indexers})
# train_ds, test_ds = [reader.read(fname) for fname in ['train.json', 'test.json']]
reader = NewsGroupsDataSetReader(tokenizer=tokenizer,  # TODO: token_indexer 的 key
                                 token_indexers={'tokens': token_indexers})
train_ds, test_ds = [reader.read(fname) for fname in ['train', 'test']]
val_ds = None

voc = Vocabulary()

iterator = BucketIterator(batch_size=config.batch_size,
                          sorting_keys=[('sentence', 'num_tokens')])
iterator.index_with(vocab=voc)

# 2. 搭建模型

word_embeddings = get_embedder()

encoder = PytorchSeq2VecWrapper(module=nn.LSTM(word_embeddings.get_output_dim(),
                                               dropout=config.dropout,
                                               hidden_size=config.lstm_hid_size,
                                               bidirectional=True,
                                               batch_first=True))

model = BaseModelWithoutKnowledge(voc=voc, word_embeddings=word_embeddings,
                                  encoder=encoder, out_sz=reader.label_length, multi=False)
model = model.cuda(cuda_device) if cuda_device > -1 else model
# 3. 训练
optimizer = optim.Adam(model.parameters(), lr=config.lr)

trainer = Trainer(model=model, optimizer=optimizer,
                  iterator=iterator, train_dataset=train_ds,
                  cuda_device=cuda_device,
                  num_epochs=config.epochs,
                  patience=5,
                  )

trainer.train()
