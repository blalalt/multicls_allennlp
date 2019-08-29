import json
import numpy as np
from settings import config
from sklearn.datasets import fetch_20newsgroups
from typing import Callable, Dict, Iterator, List, Optional, Iterable
from allennlp.data import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, MetadataField, ArrayField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


class ReutersDataSetReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int] = config.max_seq_len) -> None:
        super().__init__(lazy=True)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer}
        self.max_seq_len = max_seq_len
        self.data_dir = config.data_home / 'reuters'
        self.label2idx = self.get_label_idx()
        self.label_length = len(self.label2idx)

    def get_label_idx(self):
        with open(self.data_dir / 'labels2idx.json', 'r', encoding='utf8') as f:
            label2idx = json.load(f)
        return label2idx

    def text_to_instance(self, tokens: List[Token], id: str = None,
                         labels: List = None) -> Instance:
        fields = {}
        sentence_field = TextField(tokens=tokens, token_indexers=self.token_indexers)
        fields['sentence'] = sentence_field

        id_field = MetadataField(id)
        fields['id'] = id_field

        # 预测集,或无监督
        labels_vec = [0] * len(self.label2idx)
        if labels:  # 有监督训练
            for label in labels:
                labels_vec[self.label2idx[label]] = 1
        label_field = ArrayField(array=np.array(labels_vec))
        fields['labels'] = label_field

        return Instance(fields=fields)

    def _read(self, file_name: str) -> Iterable[Instance]:
        file_path = self.data_dir / file_name
        with open(file_path, 'r', encoding='utf8') as f:
            news = json.load(f)
            for new in news:
                sentence = ' '.join(new['features'])
                tokens = [Token(w) for w in self.tokenizer(sentence)]
                _id = new['id']
                labels = new['labels']
                yield self.text_to_instance(
                    tokens=tokens,
                    id=_id,
                    labels=labels
                )


class NewsGroupsDataSetReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int] = config.max_seq_len) -> None:
        super().__init__(lazy=True)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer}
        self.max_seq_len = max_seq_len
        self.data_dir = config.data_home / 'newsgroups'
        self.label_length = 20
        self.label2idx = {k: k for k in range(self.label_length)}

    def text_to_instance(self, tokens: List[Token],
                         label: int = None, id: str = None) -> Instance:
        fields = {}
        sentence_field = TextField(tokens=tokens, token_indexers=self.token_indexers)
        fields['sentence'] = sentence_field

        id_field = MetadataField(id)
        fields['id'] = id_field

        label_field = LabelField(label=label, skip_indexing=True)
        fields['labels'] = label_field

        return Instance(fields=fields)

    def _read(self, file_path: str = None) -> Iterable[Instance]:
        data_set = fetch_20newsgroups(data_home=self.data_dir,  # 文件下载的路径
                                      subset=file_path,  # 加载那一部分数据集 train/test
                                      categories=None,  # 选取哪一类数据集[类别列表]，默认20类
                                      shuffle=True,  # 将数据集随机排序
                                      random_state=42,  # 随机数生成器
                                      remove=('headers', 'footers', 'quotes'),  # ('headers','footers','quotes') 去除部分文本
                                      download_if_missing=True  # 如果没有下载过，重新下载
                                      )
        for idx, (feature, label) in enumerate(zip(data_set['data'],
                                                   data_set['target'])):
            tokens = [Token(w) for w in self.tokenizer(feature)]
            yield self.text_to_instance(
                tokens=tokens,
                id=str(idx),
                label=int(label)
            )
