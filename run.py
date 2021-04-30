import json
import os
import random
import logging
import numpy as np
import paddle
import paddle.io
from paddlenlp.embeddings import TokenEmbedding
from config import override_model_args,parse_args,set_default_args
from data import process_data,batchify,DuReaderDataset

logger = logging.getLogger()
def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
def set_log(args):
    set_seeds(args)
    if True:#args.do_train:
        # preparing embeddings
        tokens_emb = TokenEmbedding("w2v.baidu_encyclopedia.target.word-word.dim300")
        # preparing train datasets
        assert args.raw_train_file != None, "--raw_train_file should be set when training!"
        if not os.path.exists(args.train_file):
            process_data(args.raw_train_file,args.train_file,tokens_emb)
        with open(args.train_file,mode="r",encoding="utf-8") as rfp:
            train_ex = json.load(rfp)
        train_dataset = DuReaderDataset(train_ex)
        train_batch_sampler = paddle.io.DistributedBatchSampler(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        train_data_loader = paddle.io.DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=batchify,
            return_list=True)
        # preparing dev datasets
        assert args.raw_dev_file != None, "--raw_dev_file should be set when training!"
        if not os.path.exists(args.dev_file):
            process_data(args.raw_dev_file,args.dev_file,tokens_emb)
        with open(args.train_file,mode="r",encoding="utf-8") as rfp:
            dev_ex = json.load(rfp)
        dev_dataset = DuReaderDataset(dev_ex)
        dev_batch_sampler = paddle.io.DistributedBatchSampler(
            dev_dataset, batch_size=args.dev_batch_size, shuffle=True)
        dev_data_loader = paddle.io.DataLoader(
            dataset=dev_dataset,
            batch_sampler=dev_batch_sampler,
            collate_fn=batchify,
            return_list=True)

        num_training_steps = args.max_steps if args.max_steps > 0 else len(
            train_data_loader) * args.num_train_epochs
        if paddle.distributed.get_rank() == 0:
            dev_count = paddle.fluid.core.get_cuda_device_count()
            logger.info("Device count: %d" % dev_count)
            logger.info("Num train examples: %d" % len(train_dataset))
            logger.info("Num dev examples: %d" % len(dev_dataset))
            logger.info("Max train steps: %d" % num_training_steps)
        model = DocReader(args)
        model.init_lr_scheduler(args, num_training_steps)
        model.init_optimizer(args)
        model.init_loss(args)
    if args.do_predict:
        # preparing test datasets
        pass
def run(args):
    pass
if __name__ == "__main__":
    args = parse_args()
    set_default_args(args)
    set_log(args)
    run(args)