from pathlib import Path


class Config:
    def __init__(self):
        self.data_home = Path('.') / 'dataset'
        self.max_seq_len = 512
        self.embedder = 'elmo'
        self.batch_size = 128
        self.lstm_hid_size = 200
        self.lr = 0.004
        self.epochs = 30
        self.threshold = 0.5
        self.dropout = 0.3


config = Config()
