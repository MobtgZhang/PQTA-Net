import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path",type=str,default="dureader_data/dataset",help="raw data path.")
    parser.add_argument("--processed-path",type=str,default="dureader_data/processed",help="Processed data path.")
    parser.add_argument("--device",type=str,default="gpu",help="Device for selecting for the training.")
    parser.add_argument("--batch-size",type=int,default=16,help="Batch size for the train.")
    parser.add_argument("--dev-batch-size", type=int, default=16, help="Batch size for the dev.")
    parser.add_argument("--seed",type=int,default=1234,help="The seeds of the model.")
    parser.add_argument("--doc_stride",type=int,default=128,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_steps",default=-1,type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--num_train_epochs",default=3,type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_seq_length",default=128,type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "+\
                                "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to train the model.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to predict.")
    args = parser.parse_args()
    return args
def set_default_args(args):
    if not os.path.exists(args.processed_path):
        os.mkdir(args.processed_path)
    args.raw_train_file = os.path.join(args.raw_path, "train.json")
    args.raw_dev_file = os.path.join(args.raw_path, "dev.json")
    args.raw_test_file = os.path.join(args.raw_path, "test.json")
    args.train_file = os.path.join(args.processed_path,"train.json")
    args.dev_file = os.path.join(args.processed_path,"dev.json")
    args.test_file = os.path.join(args.processed_path,"test.json")
def override_model_args(args, new_args):
    pass