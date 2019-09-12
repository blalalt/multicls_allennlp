from settings import config
from allennlp.modules.text_field_embedders import (BasicTextFieldEmbedder,
                                                   TextFieldEmbedder)
from allennlp.modules.token_embedders import ElmoTokenEmbedder


def get_elmo_embedder():
    options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
    weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
    return word_embeddings


def get_bert_embedder():
    from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
    bert_embedder = PretrainedBertEmbedder(
        pretrained_model="bert-base-uncased",
        top_layer_only=True,  # conserve memory
    )
    word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                                                allow_unmatched_keys=True)
    return word_embeddings


def get_embedder(name: str = config.embedder):
    if name == 'elmo':
        return get_elmo_embedder()
    elif name == 'bert':
        return get_bert_embedder()


def get_token_utils(name: str = config.embedder):
    if name == 'elmo':
        from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
        from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer

        # the token indexer is responsible for mapping tokens to integers
        token_indexer = ELMoTokenCharactersIndexer()

        def tokenizer(x: str):
            return [w.text for w in
                    SpacyWordSplitter(language='en_core_web_sm',
                                      pos_tags=False).split_words(x)[:config.max_seq_len]]

        return token_indexer, tokenizer
    elif name == 'bert':
        from allennlp.data.token_indexers import PretrainedBertIndexer

        token_indexer = PretrainedBertIndexer(
            pretrained_model="bert-base-uncased",
            max_pieces=config.max_seq_len,
            do_lowercase=True,
        )

        def tokenizer(s: str):
            return token_indexer.wordpiece_tokenizer(s)[:config.max_seq_len - 2]

        return token_indexer, tokenizer
