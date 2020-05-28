import os
from pymagnitude import Magnitude, MagnitudeUtils

base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__),'../data')))


class MagnitudeVectors():

    def __init__(self, emdim):

        #base_dir = os.path.join(os.path.dirname(__file__), 'data')

        self.fasttext_dim = 300
        self.glove_dim = emdim - 100

        assert self.glove_dim in [50, 100, 200,
                                  300], "Embedding dimension must be one of the following: 350, 400, 500, 600"

        glove = Magnitude(os.path.join(data_dir, "cc.ro.300.magnitude"))
        #fasttext = Magnitude(os.path.join(base_dir, "wiki.ro.magnitude"))
        #self.vectors = Magnitude(glove, fasttext)
        self.vectors = glove

    def load_vectors(self):
        return self.vectors


from keras import backend as K


def accuracy(y_true, y_pred):

    def calculate_accuracy(true_and_pred):
        y_true, y_pred_start, y_pred_end = true_and_pred

        start_probability = y_pred_start[K.cast(y_true[0], dtype='int32')]
        end_probability = y_pred_end[K.cast(y_true[1], dtype='int32')]
        return (start_probability + end_probability) / 2.0

    y_true = K.squeeze(y_true, axis=1)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]
    accuracy = K.map_fn(calculate_accuracy, (y_true, y_pred_start, y_pred_end), dtype='float32')
    return K.mean(accuracy, axis=0)

def negative_avg_log_error(y_true, y_pred):

    def sum_of_log_probabilities(true_and_pred):
        y_true, y_pred_start, y_pred_end = true_and_pred

        start_probability = y_pred_start[K.cast(y_true[0], dtype='int32')]
        end_probability = y_pred_end[K.cast(y_true[1], dtype='int32')]
        return K.log(start_probability) + K.log(end_probability)

    y_true = K.squeeze(y_true, axis=1)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]
    batch_probability_sum = K.map_fn(sum_of_log_probabilities, (y_true, y_pred_start, y_pred_end), dtype='float32')
    return -K.mean(batch_probability_sum, axis=0)


from keras.utils import Sequence
import os
import numpy as np


class BatchGenerator(Sequence):
    'Generates data for Keras'

    vectors = None

    def __init__(self, name, batch_size, emdim, max_passage_length, max_query_length, shuffle):
        'Initialization'

        base_dir = os.path.join(os.path.dirname(__file__), 'data')

        self.vectors = MagnitudeVectors(emdim).load_vectors()

        self.max_passage_length = max_passage_length
        self.max_query_length = max_query_length

        self.context_file = os.path.join(base_dir, 'squad', name + '.context')
        self.question_file = os.path.join(base_dir, 'squad', name + '.question')
        self.span_file = os.path.join(base_dir, 'squad', name + '.span')
        

        self.batch_size = batch_size
        i = 0
        with open(self.span_file, 'r', encoding='utf-8') as f:

            for i, _ in enumerate(f):
                pass
        self.num_of_batches = (i + 1) // self.batch_size
        self.indices = np.arange(i + 1)
        self.shuffle = shuffle

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.num_of_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        start_index = (index * self.batch_size) + 1
        end_index = ((index + 1) * self.batch_size) + 1

        inds = self.indices[start_index:end_index]

        contexts = []
        with open(self.context_file, 'r', encoding='utf-8') as cf:
            for i, line in enumerate(cf, start=1):
                line = line[:-1]
                if i in inds:
                    contexts.append(line.split(' '))

        questions = []
        with open(self.question_file, 'r', encoding='utf-8') as qf:
            for i, line in enumerate(qf, start=1):
                line = line[:-1]
                if i in inds:
                    questions.append(line.split(' '))

        answer_spans = []
        with open(self.span_file, 'r', encoding='utf-8') as sf:
            for i, line in enumerate(sf, start=1):
                line = line[:-1]
                if i in inds:
                    answer_spans.append(line.split(' '))

        context_batch = self.vectors.query(contexts, pad_to_length=self.max_passage_length)
        question_batch = self.vectors.query(questions, pad_to_length=self.max_query_length)
        if self.max_passage_length is not None:
            span_batch = np.expand_dims(np.array(answer_spans, dtype='float32'), axis=1).clip(0,
                                                                                              self.max_passage_length - 1)
        else:
            span_batch = np.expand_dims(np.array(answer_spans, dtype='float32'), axis=1)
        return [context_batch, question_batch], [span_batch]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)



def load_data_generators(batch_size, emdim, max_passage_length=None, max_query_length=None, shuffle=False):
    train_generator = BatchGenerator("train", batch_size, emdim, max_passage_length, max_query_length, shuffle)
    validation_generator = BatchGenerator("validation", batch_size, emdim, max_passage_length, max_query_length, shuffle)

    return train_generator, validation_generator


import random
import json
import nltk
from tqdm import tqdm
from six.moves.urllib.request import urlretrieve

random.seed(42)
np.random.seed(42)
nltk.download('punkt')


def write_to_file(out_file, line):
    """Take a line and file as input, encdes the line to utf-8 and then writes that line to the file"""
    out_file.write(line + '\n')


def data_from_json(filename):
    """Loads JSON data from filename and returns"""
    with open(filename,encoding = "utf-8") as data_file:
        data = json.load(data_file)
    return data


def tokenize(sequence, do_lowercase):
    """Tokenizes the input sequence using nltk's word_tokenize function, replaces two single quotes with a double quote"""

    if do_lowercase:
        tokens = [token.replace("``", '"').replace("''", '"').lower()
                  for token in nltk.word_tokenize(sequence)]
    else:
        tokens = [token.replace("``", '"').replace("''", '"')
                  for token in nltk.word_tokenize(sequence)]
    return tokens


def total_examples(dataset):
    """Returns the total number of (context, question, answer) triples, given the data loaded from the SQuAD json file"""
    total = 0
    for article in dataset['data']:
        for para in article['paragraphs']:
            total += len(para['qas'])
    return total


def get_char_word_loc_mapping(context, context_tokens):
    """
    Return a mapping that maps from character locations to the corresponding token locations.
    If we're unable to complete the mapping e.g. because of special characters, we return None.

    Inputs:
      context: string (unicode)
      context_tokens: list of strings (unicode)

    Returns:
      mapping: dictionary from ints (character locations) to (token, token_idx) pairs
        Only ints corresponding to non-space character locations are in the keys
        e.g. if context = "hello world" and context_tokens = ["hello", "world"] then
        0,1,2,3,4 are mapped to ("hello", 0) and 6,7,8,9,10 are mapped to ("world", 1)
    """
    acc = ''  # accumulator
    current_token_idx = 0  # current word loc
    mapping = dict()

    # step through original characters
    for char_idx, char in enumerate(context):
        if char != u' ' and char != u'\n':  # if it's not a space:
            acc += char  # add to accumulator
            context_token = context_tokens[current_token_idx]  # current word token
            if acc == context_token:  # if the accumulator now matches the current word token
                # char loc of the start of this word
                syn_start = char_idx - len(acc) + 1
                for char_loc in range(syn_start, char_idx + 1):
                    mapping[char_loc] = (acc, current_token_idx)  # add to mapping
                acc = ''  # reset accumulator
                current_token_idx += 1

    if current_token_idx != len(context_tokens):
        return None
    else:
        return mapping


def preprocess_and_write(dataset, name, out_dir, do_lowercase):
    """Reads the dataset, extracts context, question, answer, tokenizes them, and calculates answer span in terms of token indices.
    Note: due to tokenization issues, and the fact that the original answer spans are given in terms of characters, some examples are discarded because we cannot get a clean span in terms of tokens.

    This function produces the {train/dev}.{context/question/answer/span} files.

    Inputs:
      dataset: read from JSON
      tier: string ("train" or "dev")
      out_dir: directory to write the preprocessed files
    Returns:
      the number of (context, question, answer) triples written to file by the dataset.
    """

    num_exs = 0  # number of examples written to file
    num_mappingprob, num_tokenprob, num_spanalignprob = 0, 0, 0
    examples = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing"):

        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):

            context = article_paragraphs[pid]['context'].strip()  # string

            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context, do_lowercase=do_lowercase)  # list of strings (lowercase)

            if do_lowercase:
                context = context.lower()

            qas = article_paragraphs[pid]['qas']  # list of questions

            # charloc2wordloc maps the character location (int) of a context token to a pair giving (word (string), word loc (int)) of that token
            charloc2wordloc = get_char_word_loc_mapping(
                context, context_tokens)

            if charloc2wordloc is None:  # there was a problem
                num_mappingprob += len(qas)
                continue  # skip this context example

            # for each question, process the question and answer and write to file
            for qn in qas:

                # read the question text and tokenize
                question = qn['question'].strip()  # string
                question_tokens = tokenize(question, do_lowercase=do_lowercase)  # list of strings

                # of the three answers, just take the first
                # get the answer text
                # answer start loc (character count)
                
                ans_text = qn['answers'][0]['text']
                ans_start_charloc = qn['answers'][0]['answer_start']

                if do_lowercase:
                    ans_text = ans_text.lower()

                # answer end loc (character count) (exclusive)
                ans_end_charloc = ans_start_charloc + len(ans_text)

                # Check that the provided character spans match the provided answer text
                if context[ans_start_charloc:ans_end_charloc] != ans_text:
                    # Sometimes this is misaligned, mostly because "narrow builds" of Python 2 interpret certain Unicode characters to have length 2 https://stackoverflow.com/questions/29109944/python-returns-length-of-2-for-single-unicode-character-string
                    # We should upgrade to Python 3 next year!
                    num_spanalignprob += 1
                    continue

                # get word locs for answer start and end (inclusive)
                # answer start word loc
                ans_start_wordloc = charloc2wordloc[ans_start_charloc][1]
                # answer end word loc
                ans_end_wordloc = charloc2wordloc[ans_end_charloc - 1][1]
                assert ans_start_wordloc <= ans_end_wordloc

                # Check retrieved answer tokens match the provided answer text.
                # Sometimes they won't match, e.g. if the context contains the phrase "fifth-generation"
                # and the answer character span is around "generation",
                # but the tokenizer regards "fifth-generation" as a single token.
                # Then ans_tokens has "fifth-generation" but the ans_text is "generation", which doesn't match.
                ans_tokens = context_tokens[ans_start_wordloc:ans_end_wordloc + 1]
                if "".join(ans_tokens) != "".join(ans_text.split()):
                    num_tokenprob += 1
                    continue  # skip this question/answer pair

                
                examples.append((' '.join(context_tokens), ' '.join(question_tokens), ' '.join(ans_tokens), ' '.join([str(ans_start_wordloc), str(ans_end_wordloc)])))

                num_exs += 1

    print("Number of (context, question, answer) triples discarded due to char -> token mapping problems: ", num_mappingprob)
    print("Number of (context, question, answer) triples discarded because character-based answer span is unaligned with tokenization: ", num_tokenprob)
    print("Number of (context, question, answer) triples discarded due character span alignment problems (usually Unicode problems): ", num_spanalignprob)
    print("Processed %i examples of total %i\n" %
          (num_exs, num_exs + num_mappingprob + num_tokenprob + num_spanalignprob))

    # shuffle examples
    indices = list(range(len(examples)))
    np.random.shuffle(indices)

    with open(os.path.join(out_dir, name + '.context'), 'w', encoding='utf-8') as context_file, \
            open(os.path.join(out_dir, name + '.question'), 'w', encoding='utf-8') as question_file, \
            open(os.path.join(out_dir, name + '.answer'), 'w', encoding='utf-8') as ans_text_file, \
            open(os.path.join(out_dir, name + '.span'), 'w', encoding='utf-8') as span_file:

        
        for i in indices:

            (context, question, answer, answer_span) = examples[i]

            # write tokenized data to file
            write_to_file(context_file, context)
            write_to_file(question_file, question)
            write_to_file(ans_text_file, answer)
            write_to_file(span_file, answer_span)



def data_download_and_preprocess(do_lowercase=True):
    data_dir = os.path.join(base_dir, 'data', 'squad')

    print("Will put preprocessed SQuAD datasets in {}".format(data_dir))

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_filename = "train.json"

    # read train set
    train_data = data_from_json(os.path.join(data_dir, train_filename))
    print("Train data has %i examples total" % total_examples(train_data))

    # preprocess train set and write to file
    if not os.path.isfile(os.path.join(data_dir, 'train.context')):
        print("Preprocessing training data")
        preprocess_and_write(train_data, "train", data_dir, do_lowercase=do_lowercase)
    print("Train data preprocessed!")

    validation_filename = "validation.json"

    # read train set
    validation_data = data_from_json(os.path.join(data_dir, validation_filename))
    print("Validation data has %i examples total" % total_examples(validation_data))

    # preprocess train set and write to file
    if not os.path.isfile(os.path.join(data_dir, 'validation.context')):
        print("Preprocessing Validation data")
        preprocess_and_write(validation_data,"validation", data_dir, do_lowercase=do_lowercase)
    print("Validation data preprocessed!")
