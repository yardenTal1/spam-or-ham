import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import words
import pandas as pd
import string
from nltk.wsd import lesk
import enchant


def read_data():
    messages = pd.read_csv("./spam.csv", encoding='latin-1')
    # Drop the extra columns and rename columns
    messages = messages.drop(labels=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    messages.columns = ["category", "text"]
    return messages


def create_noslang_dict():
    noslang = {}
    with open('./Noslang.txt') as noslangfile:
        for line in noslangfile:
            slang, english = line.strip().split(' :', 1)
            noslang[slang] = english
    return noslang


def has_number(in_string):
    return any(char.isdigit() for char in in_string)


def is_number(in_string):
    return all(char.isdigit() for char in in_string)


def translate_slang_and_lower(message):
    message = message.translate(str.maketrans('', '', string.punctuation))
    words = [slang_dict[word.lower()].lower() if word.lower() in slang_dict else word.lower() for word in message.split()]
    message = " ".join(words)
    return message


def cleanText(message):
    #TODO remove words that are not in english? what about 'goooooood' for example?

    # remove punctuation
    words = []
    sentense = message.split()
    for word in sentense:
        if has_number(word):
            word = handle_nembers(word)
            if word is None:
                continue
        else:
            if not english_dict.check(word):
                print(word)
                continue
            # remove stop words
            if word in nltk_stop_words or word in signs_list:
                continue

            # find a sense that describe the current word
            word = find_sense(word, sentense)

        words.append(stemmer.stem(word))

    # join words to one string
    return " ".join(words)


def find_sense(word, sentense):
    new_word = lesk(sentense, word, 'n')
    if new_word is not None:
        #
        new_word = new_word.hypernyms()
        if new_word is not None and len(new_word) > 0:
            word = new_word[0]._name.split('.')[0]  # TODO
    return word
find_sense('call', 'please call me back as soon as possible i have great news :)')
def handle_nembers(number):
    number = number.translate(str.maketrans('', '', ','))
    number = number.translate(str.maketrans('', '', '-'))
    if is_number(number):
        if len(number) >= 7:
            return 'phonenumber'
        return 'othernumber'
    return None # TODO maybe add other signs - dollar, euro, etc


def pre_processing_stage_one(message):
    message_tokens = nltk.word_tokenize(message)


if __name__ == "__main__":
    nltk_stop_words = set(nltk.corpus.stopwords.words('english'))
    signs_list = [',', '/', '.', '"', "'", '?', '\\', ':', '(', ')', '*', '-', '=', '+', '&', '^', '$', '%', '#', '@',
                  '!', '`', '~', "'s"]
    stemmer = nltk.SnowballStemmer("english")
    # nltk_words = nltk.corpus.words.words()
    english_dict = enchant.Dict("en_US")
    slang_dict = create_noslang_dict()

    msg = read_data()
    msg['text'] = msg['text'].apply(translate_slang_and_lower)
    msg['text'] = msg['text'].apply(cleanText)

    pass # TODO do lesk just after we remove un english words