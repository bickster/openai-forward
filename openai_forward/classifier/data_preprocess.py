from importlib.resources import open_text
import json
import numpy as np
from loguru import logger


class TextVectorization:
    def __init__(self, vocab):
        self.vocabulary = vocab
        self.output_length = 32

    def standardize(self, string):
        import string as edit
        string = string.translate(str.maketrans('', '', edit.punctuation))
        return '[START] '+string.lower()+' [END]'

    def __call__(self, string):
        stand_string = self.standardize(string)
        split_string = stand_string.split()
        tokens = []
        for word in split_string:
            if word in self.vocabulary:
                tokens.append(self.vocabulary.index(word))
            else:
                tokens.append(1)
        if tokens[-1] == 1: # assumes last word was spliced if [UNK]
            tokens = tokens[:-1]
        while len(tokens) < self.output_length:
            tokens.append(0)
        return np.array([tokens], dtype=np.int32)


vectorizer = None


def load_vectorizer():
    global vectorizer
    try:
        import openai_forward.classifier.data as data
        with open_text(data, 'tokens.json') as file:
            tokens = file.read()
        vectorizer = TextVectorization(list(json.loads(tokens).keys())[:20000])
    except Exception as error:
        logger.error("PRMOPT_PREPROCESSING_ERROR: vectorizer failed to initialize")
        logger.error(error)

def confirm_process_message(req_body):
    if 'model' in req_body:
        if 'gpt' in req_body['model']: # only two possible models, either gpt or dall-e. ignore if dall-e
            if 'tools' in req_body:
                for tool in req_body['tools']:
                    if tool['type'] == 'function':
                        if tool['function']['name'] == 'generateImage':
                            if 'tool_choice' not in tool['function']['parameters']:
                                return True
    return False

def preprocess_prompt(prompt_string):
    if type(prompt_string) != str:
        raise TypeError
        logger.error("PRMOPT_PREPROCESSING_ERROR: prompt not in string format")
        return False
    try:
        if vectorizer:
            token_prompt = vectorizer(prompt_string)
        else:
            logger.error("PRMOPT_PREPROCESSING_ERROR: unable to vectorize prompt, vectorizer not inistantiated")
            return False
    except Exception as error:
        logger.error(f"PRMOPT_PREPROCESSING_ERROR: vectorizer failed on {prompt}")
        logger.error(error)
        return False
    return token_prompt