import torch
import torch.nn as nn
from typing import *
from settings import config
from metircs import AccuracyMultiLabel
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.training.metrics import F1Measure, CategoricalAccuracy


class BaseModelWithoutKnowledge(Model):
    def __init__(self, voc: Vocabulary, word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 out_sz: int, multi=True):
        super(BaseModelWithoutKnowledge, self).__init__(voc)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.projection = nn.Linear(in_features=self.encoder.get_output_dim(),
                                    out_features=out_sz)
        self.loss = nn.BCEWithLogitsLoss() if multi else nn.CrossEntropyLoss()
        self.accuracy = AccuracyMultiLabel() if multi else CategoricalAccuracy(top_k=3)

    def forward(self, id: Any, sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        state = self.encoder(embeddings, mask)
        class_logits = self.projection(state)
        output = {'class_logits': class_logits}
        if labels is not None:
            self.accuracy(predictions=class_logits, gold_labels=labels)
            output['loss'] = self.loss(class_logits, labels)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset=reset)}