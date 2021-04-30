import json

import numpy as np
import paddle
from paddle.io import Dataset
# define a random dataset
class DuReaderDataset(Dataset):
    def __init__(self, examples):
        super(DuReaderDataset, self).__init__()
        self.examples = examples
        self.num_samples = len(examples)
    def __getitem__(self, idx):
        question_w = paddle.to_tensor(self.examples[idx]['question_w'], dtype=np.int64)
        context_w = paddle.to_tensor(self.examples[idx]['context_w'], dtype=np.int64)
        title_w = paddle.to_tensor(self.examples[idx]['title_w'], dtype=np.int64)
        question_c = paddle.to_tensor(self.examples[idx]['question_c'], dtype=np.int64)
        context_c = paddle.to_tensor(self.examples[idx]['context_c'], dtype=np.int64)
        title_c = paddle.to_tensor(self.examples[idx]['title_c'], dtype=np.int64)
        start_positions = self.examples[idx]['start_positions']
        end_positions = self.examples[idx]['end_positions']
        return (question_c,question_w,title_c,title_w,context_c,context_w),(start_positions,end_positions)
    def __len__(self):
        return self.num_samples


