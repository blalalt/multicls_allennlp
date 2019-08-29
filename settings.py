from pathlib import Path


class Config:
    def __init__(self):
        self.data_home = Path('.') / 'dataset'
        self.max_seq_len = 510
        self.embedder = 'elmo'
        self.batch_size = 8
        self.lstm_hid_size = 200
        self.lr = 0.003
        self.epochs = 20
        self.threshold = 0.5


config = Config()
