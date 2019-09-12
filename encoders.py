from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
import torch
from torch import nn
from settings import config
from allennlp.data.vocabulary import Vocabulary

class BertSentencePooler(Seq2VecEncoder):
    """
    sentence classification models on BERT we only use the embedding corresponding to the first token in the sentence.
    """

    def forward(self, embs: torch.tensor,
                mask: torch.tensor = None) -> torch.tensor:
        # extract first token tensor
        return embs[:, 0]

    def get_output_dim(self) -> int:
        return self.out_dim


def get_encoder(voc: Vocabulary, embed_out_dim: int, name: str=config.embedder):
    if name == 'bert':
        bert = BertSentencePooler(voc)
        bert.out_dim = embed_out_dim
        return bert
    else:
        return PytorchSeq2VecWrapper(module=nn.GRU(embed_out_dim,
                                                   dropout=config.dropout,
                                                   hidden_size=config.lstm_hid_size,
                                                   bidirectional=True,
                                                   batch_first=True))
