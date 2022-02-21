import spacy
import pandas as pd
import json
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_sm")
tqa_file_path = "/Users/mariatseytlin/Documents/Msc/Lab/data_qasrl/qasrl-v2/orig/dev.jsonl"
wiki_file_path = "/Users/mariatseytlin/Documents/Msc/Lab/data_qanom/dev_wiki.csv"
wikinews_file_path = "/Users/mariatseytlin/Documents/Msc/Lab/wikinews_dev/annot.final.wikinews.dev.csv"
MAX_SENTENCES = 100

'''
This code creates batch csv from data filepath
Usage: First run the correct preprocess function, and then run "process_file" function on the output.
'''

def spacy_example():
    text = ("""My name is Shaurya Uppal. 
    I enjoy writing articles on GeeksforGeeks checkout
    my other article by going to my profile section.""")

    # text=("An improvisational section was built around pieces by Mr. Douglas , beginning with `` Golden Rain , '' a lilting , laid-back lead in to the uptempo `` Sky , '' which gave Mr. Stoltzman the opportunity to wail in a high register and show off his fleet fingers .")
    text =  "Thanks to a new air-traffic agreement and the ability of Irish travel agents to issue Aeroflot tickets , tourists here are taking advantage of Aeroflot 's reasonable prices to board flights in Shannon for holidays in Havana , Kingston and Mexico City ."
    doc = nlp(text)

    # Token and Tag
    for token in doc:
        print(token, token.pos_)

    # You want list of Noun tokens
    print("Nouns:", [token.text for token in doc if token.pos_ == "NOUN"])
    print("preps", [token.text for token in doc if token.pos_ == "ADP"])
# spacy_example()


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


# this function gets a file with nes sentence in each line
def process_file(file_path, filename):
    '''
    This processes the txt file that is created in preprocess and creates a csv file for uploading to mturk
    :param file_path: txt file path that preprocess outputs
    :param filename: path of the newly created csv file
    '''
    to_df = []
    with open(file_path, 'r') as sentences_file:
        for line in sentences_file:
            split_line = line.split('#_#')
            sentence=split_line[0]
            id = split_line[1][:-1]
            preprocessed_sents = find_targets_and_preps(sentence, id)
            for sent in preprocessed_sents:
                to_df.append([sent['sentenceId'], sent['sentence'], sent['index'], sent['verbIndices']])
    df = pd.DataFrame(to_df, columns=['sentenceId', 'sentence', 'index', 'verbIndices'])
    df.to_csv(filename, index=False, encoding="utf-8", line_terminator='\r\n')


# def find_targets_and_preps(sentence, id):
#     nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
#     doc = nlp(sentence)
#     sent_dict = [{'id': id, 'sentence': sentence, 'index': i, 'prep': ''} for i in range(len(doc)) if doc[i].pos_ == "NOUN"]
#     prep_array = [(doc[i].text, i) for i in range(len(doc)) if doc[i].pos_ == "ADP"]
#     preprocessed_sents = find_preps(prep_array, sent_dict)
#     return preprocessed_sents

def find_targets_and_preps(sentence, id):
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    doc = nlp(sentence)
    verb_indices = find_verbs(doc)
    sent_dict = [{'sentenceId': id, 'sentence': sentence, 'index': i, 'verbIndices': verb_indices} for i in range(len(doc)) if doc[i].pos_ == "NOUN"]
    # prep_array = [(doc[i].text, i) for i in range(len(doc)) if doc[i].pos_ == "ADP"]
    # preprocessed_sents = find_preps(prep_array, sent_dict)
    return sent_dict


def find_verbs(doc):
    verbs_indices = '&'.join([str(verb_idx) for verb_idx in range(len(doc)) if doc[verb_idx].pos_ == "VERB"])
    return verbs_indices


def find_preps(prep_array, sent_dict):
    for sent in sent_dict:
        for prep in prep_array:
            if prep[1] - sent['index'] == 1:
                sent['prep'] = prep[0]
    return sent_dict


def preprocess_json(file_path, new_file_name, max_sentences):
    '''
    This is used for tqa
    :param file_path: filepath for data
    :param new_file_name: txt file path for output
    :param max_sentences: sentences to take from the data
    '''
    counter = 0
    with open(new_file_name, "w") as new_file:
        with open(file_path, 'r') as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            result = json.loads(json_str)
            new_file.write(' '.join(result["sentenceTokens"]))
            new_file.write('#_#' + result["sentenceId"])
            new_file.write("\n")
            counter += 1
            if counter == max_sentences:
                return


def preprocess_csv(file_path, new_file_name, max_sentences):
    '''
    This is used for wikinews and wikipedia
    :param file_path: filepath for data
    :param new_file_name: txt file path for output
    :param max_sentences: sentences to take from the data
    '''
    df = pd.read_csv(file_path)
    df = df.drop_duplicates(subset="qasrl_id")
    counter = 0
    with open(new_file_name, "w") as new_file:
        for index, row in df.iterrows():
            new_file.write(row["sentence"] + "#_#" + row["qasrl_id"] + '\n')
            counter += 1
            if counter == max_sentences:
                return


def find_chars():
    with open("html_file/batch3_with_prep.csv", "r") as file:
        for line in file:
            print(line)

# preprocess_csv(wikinews_file_path, "../batches/crowd_batch5/wikinews_sentences.txt", MAX_SENTENCES)
# preprocess_csv(wiki_file_path, "../batches/crowd_batch5/wiki_sentences.txt", MAX_SENTENCES)
# preprocess_json(tqa_file_path, "../batches/crowd_batch5/tqa_sentences.txt", MAX_SENTENCES)
# process_file("../batches/crowd_batch5/wikinews_sentences.txt", "../batches/crowd_batch5/wikinews_sentences_batch.csv")
process_file("../batches/training/crowd_batch1/wiki_sentences.txt", "../batches/training/crowd_batch1/crowd_batch1.csv")
# process_file("../batches/crowd_batch5/tqa_sentences.txt", "../batches/crowd_batch5/tqa_sentences_batch.csv")

