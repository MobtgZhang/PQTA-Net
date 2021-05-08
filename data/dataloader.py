import json

import numpy as np
import paddle
from paddle.io import Dataset
# define a random dataset
class DuReaderDataset(Dataset):
    def __init__(self, examples,PAD_ID=None,UNK_ID=None):
        super(DuReaderDataset, self).__init__()
        self.examples = examples
        self.num_samples = len(examples)
        self.PAD_ID = PAD_ID
        self.UNK_ID = UNK_ID
    def __getitem__(self, idx):
        question_w = paddle.to_tensor(self.examples[idx]['question_w'], dtype=np.int64)
        context_w = paddle.to_tensor(self.examples[idx]['context_w'], dtype=np.int64)
        title_w = paddle.to_tensor(self.examples[idx]['title_w'], dtype=np.int64)
        question_c = paddle.to_tensor(self.examples[idx]['question_c'], dtype=np.int64)
        context_c = paddle.to_tensor(self.examples[idx]['context_c'], dtype=np.int64)
        title_c = paddle.to_tensor(self.examples[idx]['title_c'], dtype=np.int64)
        is_impossible = self.examples[idx]['is_impossible']
        start_positions = self.examples[idx]['start_positions']
        end_positions = self.examples[idx]['end_positions']
        if (self.PAD_ID is not None) and(self.UNK_ID is not None):
            return (question_c,question_w,title_c,title_w,context_c,context_w),(start_positions,end_positions,is_impossible),self.PAD_ID,self.UNK_ID
        else:
            return (question_c, question_w, title_c, title_w, context_c, context_w), (start_positions, end_positions,is_impossible)
    def __len__(self):
        return self.num_samples


