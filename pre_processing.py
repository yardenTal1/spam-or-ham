import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import words
import pandas as pd
import string
from nltk.wsd import lesk
import enchant


PHONE_NUMBER = 'phonenumber'
OTHER_NUMBER = 'othernumber'


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


def pre_process_msg(message):
    # remove punctuation # TODO check if use ' ' instead of ''
    message = message.translate(str.maketrans('', '', string.punctuation))
    # translate slang, and lower words
    words = [slang_dict[word.lower()].lower() if word.lower() in slang_dict else word.lower() for word in message.split()]
    # translate numbers to phonenumber and othernumber
    words = [handle_nembers(word) if has_number(word) and handle_nembers(word) is not None else word for word in words]

    message = " ".join(words)
    return message


def clean_text(message):
    #TODO remove words that are not in english? what about 'goooooood' for example?

    # remove punctuation
    words = []
    sentense = message.split()
    for word in sentense:
        if word in [PHONE_NUMBER, OTHER_NUMBER]:
            words.append(word)
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


def handle_nembers(number):
    number = number.translate(str.maketrans('', '', ','))
    number = number.translate(str.maketrans('', '', '-'))
    if is_number(number):
        if len(number) >= 7:
            return PHONE_NUMBER
        return OTHER_NUMBER
    return None # TODO maybe add other signs - dollar, euro, etc


def pre_process_data(data):
    data['text'] = data['text'].apply(pre_process_msg)
    data['text'] = data['text'].apply(clean_text)


if __name__ == "__main__":
    # General variables
    nltk_stop_words = set(nltk.corpus.stopwords.words('english'))
    signs_list = [',', '/', '.', '"', "'", '?', '\\', ':', '(', ')', '*', '-', '=', '+', '&', '^', '$', '%', '#', '@',
                  '!', '`', '~', "'s"]
    stemmer = nltk.SnowballStemmer("english")
    # nltk_words = nltk.corpus.words.words()
    english_dict = enchant.Dict("en_US")
    slang_dict = create_noslang_dict()

    # Actual code
    data = read_data()
    data = pre_process_data(data)

    pass # TODO do lesk just after we remove un english words
