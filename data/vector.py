#!/usr/bin/env python3
# Copyright 2018-present, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Functions for putting examples into torch format."""

import numpy as np
import paddle
global PAD_ID
global UNK_ID
def batchify(batch):
    """Gather a batch of individual examples into one batch.
    (question_c,question_w,title_c,title_w,context_c,context_w),(start_positions,end_positions)
    """
    PAD_ID = batch[0][-2]
    ques = [ex[0][1] for ex in batch]
    ques_chars = [ex[0][0] for ex in batch]
    tite = [ex[0][3] for ex in batch]
    tite_chars = [ex[0][2] for ex in batch]
    docs = [ex[0][5] for ex in batch]
    docs_chars = [ex[0][4] for ex in batch]
    y_s = [ex[1][0] for ex in batch]
    y_e = [ex[1][1] for ex in batch]
    is_impossible = [ex[1][2] for ex in batch]
    # Batch documents
    max_length = max([d.shape[0] for d in docs])
    max_char_length = max([cs.shape[0] for cs in docs_chars])

    docs_w = paddle.fluid.layers.fill_constant(shape=(len(docs),max_length), value=PAD_ID, dtype='int64')
    docs_c = paddle.fluid.layers.fill_constant(shape=(len(docs),max_char_length), value=PAD_ID, dtype='int64')

    for i,dw in enumerate(docs):
        docs_w[i,:dw.shape[0]] = dw
    for i,dc in enumerate(docs_chars):
        docs_c[i,:dc.shape[0]] = dc
    # Batch questions
    max_length = max([q.shape[0] for q in ques])
    max_char_length = max([cs.shape[0] for cs in ques_chars])

    ques_w = paddle.fluid.layers.fill_constant(shape=(len(ques), max_length), dtype=np.int64,value=PAD_ID)
    ques_c = paddle.fluid.layers.fill_constant(shape=(len(ques_chars), max_char_length), dtype=np.int64,value=PAD_ID)

    for i, qw in enumerate(ques):
        ques_w[i, :qw.shape[0]] = qw
    for i, qc in enumerate(ques_chars):
        ques_c[i, :qc.shape[0]] = qc
    # Batch titles
    max_length = max([t.shape[0] for t in tite])
    max_char_length = max([cs.shape[0] for cs in tite_chars])

    tite_w = paddle.fluid.layers.fill_constant(shape=(len(tite), max_length), dtype=np.int64,value=PAD_ID)
    tite_c = paddle.fluid.layers.fill_constant(shape=(len(tite_chars), max_char_length), dtype=np.int64,value=PAD_ID)

    for i,tw in enumerate(ques):
        tite_w[i,:tw.shape[0]] = tw
    for i,tc in enumerate(ques_chars):
        tite_c[i,:tc.shape[0]] = tc

    # batch y_s,y_e,is_impossible
    y_s = paddle.to_tensor(y_s,dtype='int64')
    y_e = paddle.to_tensor(y_e,dtype='int64')
    is_impossible = paddle.to_tensor(is_impossible,dtype='int64')
    return docs_c,docs_w,tite_c,tite_w,ques_c,ques_w,y_s,y_e,is_impossible
