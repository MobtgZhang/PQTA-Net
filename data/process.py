import logging
import json
logger = logging.getLogger()
from paddlenlp.data import JiebaTokenizer
def sentence_char_to_idx(sentence,tokens_emb):
    sentence = list(sentence)
    return [tokens_emb.get_idx_from_word(word) for word in sentence]
def process_data(loadfile,savefile,tokens_emb):
    with open(loadfile, mode="r", encoding="utf8") as rfp:
        input_data = json.load(rfp)["data"]
    new_examples = []
    tokens = JiebaTokenizer(tokens_emb.vocab)
    logger.info("Processing dataset %s."%loadfile)
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            title = paragraph["title"].strip()
            context = paragraph["context"].strip()
            for qa in paragraph["qas"]:
                qas_id = qa['id']
                question = qa["question"].strip()
                tmp_dict = {}
                tmp_dict['qas_id'] = qas_id
                tmp_dict['question_w'] = tokens.encode(question)
                tmp_dict['context_w'] = tokens.encode(context)
                tmp_dict['title_w'] = tokens.encode(title)
                tmp_dict['question_c'] = sentence_char_to_idx(question, tokens_emb)
                tmp_dict['context_c'] = sentence_char_to_idx(context, tokens_emb)
                tmp_dict['title_c'] = sentence_char_to_idx(title, tokens_emb)
                for item in qa['answers']:
                    answer_start = int(item["answer_start"])
                    answer = item["text"].strip()
                    if answer_start==-1:
                        tmp_dict['start_positions'] = -1
                        tmp_dict["end_positions"] = -1
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
def process_test_data(loadfile,savefile,tokens_emb):
    with open(loadfile, mode="r", encoding="utf8") as rfp:
        input_data = json.load(rfp)["data"]
    new_examples = []
    tokens = JiebaTokenizer(vocab=tokens_emb)
    for entry in input_data:
        title = entry.get("title", "").strip()
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"].strip()
            for qa in paragraph["qas"]:
                qas_id = qa['id']
                question = qa["question"].strip()
                tmp_dict = {}
                tmp_dict['qas_id'] = qas_id
                tmp_dict['question_w'] = tokens.encode(question)
                tmp_dict['context_w'] = tokens.encode(context)
                tmp_dict['title_w'] = tokens.encode(title)
                tmp_dict['question_c'] = sentence_char_to_idx(question, tokens_emb)
                tmp_dict['context_c'] = sentence_char_to_idx(context, tokens_emb)
                tmp_dict['title_c'] = sentence_char_to_idx(title, tokens_emb)
                new_examples.append(tmp_dict)
    with open(savefile, mode="w", encoding="utf-8") as wfp:
        json.dump(new_examples, wfp)
    return new_examples