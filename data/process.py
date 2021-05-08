import logging
import json
import random

from tqdm import tqdm
logger = logging.getLogger()
from paddlenlp.data import JiebaTokenizer
def get_idx_from_word(word, word_to_idx, unk_word):
    if word in word_to_idx:
        return word_to_idx[word]
    return word_to_idx[unk_word]
def chars_to_idx(sentence,vocab):
    return [get_idx_from_word(char,vocab.token_to_idx,vocab.unk_token) for char in list(sentence)]
def word_to_idx(sentence,vocab):
    return [get_idx_from_word(word,vocab.token_to_idx,vocab.unk_token) for word in sentence for _ in list(word)]
def sentence_to_idx(sentence,embedding):
    chars_list = []
    tokens = JiebaTokenizer(embedding)
    word_list = tokens.cut(sentence)
    for word in word_list:
        tp_w = get_idx_from_word(word, embedding.vocab.token_to_idx,embedding.vocab.unk_token)
        tp_list = [get_idx_from_word(ch, embedding.vocab.token_to_idx,embedding.vocab.unk_token) for ch in list(word)]
        chars_list.append({tp_w:tp_list})
    return chars_list
def process_data(loadfile,savefile,vocab):
    print(type(vocab))
    tokens = JiebaTokenizer(vocab)
    with open(loadfile, mode="r", encoding="utf8") as rfp:
        input_data = json.load(rfp)["data"]
    new_examples = []
    logger.info("Processing dataset %s."%loadfile)
    for entry in input_data:
        for paragraph in tqdm(entry["paragraphs"],desc="process"):
            title = paragraph["title"].strip()
            context = paragraph["context"].strip()
            for qa in paragraph["qas"]:
                qas_id = qa['id']
                question = qa["question"].strip()
                tmp_dict = {}
                tmp_dict['qas_id'] = qas_id

                tmp_dict['question_w'] = word_to_idx(tokens.cut(question),vocab)
                tmp_dict['context_w'] = word_to_idx(tokens.cut(context),vocab)
                tmp_dict['title_w'] = word_to_idx(tokens.cut(title),vocab)
                tmp_dict['question_c'] = chars_to_idx(question, vocab)
                tmp_dict['context_c'] = chars_to_idx(context, vocab)
                tmp_dict['title_c'] = chars_to_idx(title, vocab)
                tmp_dict['is_impossible'] = 1 if qa["is_impossible"] else 0
                length = len(tmp_dict['context_c'])
                for item in qa['answers']:
                    answer_start = int(item["answer_start"])
                    answer = item["text"].strip()
                    if answer_start==-1:
                        label = random.randint(0,length)
                        tmp_dict['start_positions'] = label
                        tmp_dict["end_positions"] = label
                    else:
                        # Start/end character index of the answer in the text.
                        start_char = answer_start
                        end_char = start_char + len(answer)
                        tmp_dict["start_positions"] = start_char
                        tmp_dict["end_positions"] = end_char
                    new_examples.append(tmp_dict)
    with open(savefile,mode="w",encoding="utf-8") as wfp:
        json.dump(new_examples,wfp)
    logger.info("Saved the processed dataset %s."%savefile)
    return new_examples
def process_test_data(loadfile,savefile,vocab):
    with open(loadfile, mode="r", encoding="utf8") as rfp:
        input_data = json.load(rfp)["data"]
    new_examples = []
    for entry in input_data:
        title = entry.get("title", "").strip()
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"].strip()
            for qa in paragraph["qas"]:
                qas_id = qa['id']
                question = qa["question"].strip()
                tmp_dict = {}
                tmp_dict['qas_id'] = qas_id
                tmp_dict['question'] = sentence_to_idx(question, vocab)
                tmp_dict['context'] = sentence_to_idx(context, vocab)
                tmp_dict['title'] = sentence_to_idx(title, vocab)
                new_examples.append(tmp_dict)
    with open(savefile, mode="w", encoding="utf-8") as wfp:
        json.dump(new_examples, wfp)
    return new_examples