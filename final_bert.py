import os
import torch
from torch.nn import functional as F
import string
from transformers import BertTokenizer, BertForMaskedLM, logging

logging.set_verbosity_error()

# Set global variables correctly
no_words_to_be_predicted = 5
select_model = "bert"
enter_input_text = "why are"

def set_model_config(**kwargs):
    global no_words_to_be_predicted, select_model, enter_input_text
    for key, value in kwargs.items():
        print("{0} = {1}".format(key, value))

    no_words_to_be_predicted = kwargs.get("no_words_to_be_predicted", 5)  # default to 5
    select_model = kwargs.get("select_model", "bert")  # default to 'bert'
    enter_input_text = kwargs.get("enter_input_text", "")  # default to empty string

    return no_words_to_be_predicted, select_model, enter_input_text

def load_model():
    try:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
        return bert_tokenizer, bert_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def encode_bert(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models don't predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'
    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx

def decode_bert(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])

def get_all_predictions(text_sentence, top_clean=5):
    tokenizer, model = load_model()
    input_ids, mask_idx = encode_bert(tokenizer, text_sentence)

    with torch.no_grad():
        predict = model(input_ids)[0]
    bert_predictions = predict[0, mask_idx, :].topk(top_clean)

    decoded_predictions = decode_bert(tokenizer, bert_predictions.indices.tolist(), top_clean)
    return {"bert": decoded_predictions}

def get_prediction_end_of_sentence(input_text):
    try:
        input_text += ' <mask>'
        print(input_text)
        res = get_all_predictions(input_text, top_clean=int(no_words_to_be_predicted))
        return res
    except Exception as error:
        print(f"Error in get_prediction_end_of_sentence: {error}")
        return None

try:
    print("Next Word Prediction with Pytorch using BERT")

    # Take user input
    enter_input_text = input("Enter the input text: ").strip()

    no_words_to_be_predicted, select_model, enter_input_text = set_model_config(no_words_to_be_predicted=5,
                                                                                select_model="bert",
                                                                                enter_input_text=enter_input_text)

    if select_model:
        res = get_prediction_end_of_sentence(enter_input_text)
        print("result is: {}".format(res))
        answer_bert = []
        print(res['bert'].split("\n"))
        for i in res['bert'].split("\n"):
            answer_bert.append(i)
            answer_as_string_bert = "    ".join(answer_bert)
            print("output answer is: {}".format(answer_as_string_bert))
except Exception as e:
    print(f'Some problem occurred: {e}')
