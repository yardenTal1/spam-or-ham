import nltk
import numpy as np
import pandas as pd
import string
from nltk.wsd import lesk
import enchant
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import *
from matplotlib.font_manager import FontProperties
import random

PHONE_NUMBER = 'phonenumber'
OTHER_NUMBER = 'othernumber'


def read_data():
    """
    read the sms collection dataset, and convert to dataframe
    :return: sms collection dataset, represented as dataframe
    """
    messages = pd.read_csv("./spam.csv", encoding='latin-1')
    # Drop the extra columns and rename columns
    messages = messages.drop(labels=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    messages.columns = ["category", "text"]
    return messages


def create_noslang_dict():
    """
    create an slang to english dictionary, based on noSlang dictionary
    :return: the noSlang dictionary
    """
    noslang = {}
    with open('./Noslang.txt') as noslangfile:
        for line in noslangfile:
            slang, english = line.strip().split(' :', 1)
            noslang[slang] = english
    return noslang


def first_msg_translation(message):
    """
    run the first process of the given message
    :param message: the message to translate
    :return: the message after translation
    """
    # remove punctuation
    message = message.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # translate slang, and lower words
    words = [slang_dict[word.lower()].lower() if word.lower() in slang_dict else word.lower() for word in
             message.split()]
    # translate numbers to phonenumber and othernumber
    words = [handle_nembers(word) if has_number(word) and handle_nembers(word) is not None else word for word in words]

    message = " ".join(words)
    return message


def second_msg_translation(message, feature_extraction):
    """
    run a second process of the given message
    :param message: the message to translate
    :param feature_extraction: do we want to aggregate features
    :return: the message after translation
    """
    # remove punctuation
    words = []
    sentense = message.split()
    for word in sentense:
        if word in [PHONE_NUMBER, OTHER_NUMBER, 'free']:
            words.append(word)
            continue
        else:
            if not english_dict.check(word):
                suggested_words = english_dict.suggest(word)
                if len(suggested_words) > 0 and suggested_words[0].lower() != word:
                    all_optional_sinsets = []
                    for optional_word in suggested_words:
                        optional_word = optional_word.lower()
                        all_optional_sinsets += wordnet.synsets(optional_word)
                    all_optional_sinsets = set(all_optional_sinsets)
                    new_word = lesk(sentense, word, synsets=all_optional_sinsets)
                    if new_word is not None:
                        word = get_only_word_from_syn(new_word)
                    else:
                        word = suggested_words[0].lower()
            # remove stop words
            if word in nltk_stop_words or word in signs_list:
                continue
            # find a sense that describe the current word
        words.append(word)
    new_words = []
    for word in words:
        if word in [PHONE_NUMBER, OTHER_NUMBER, 'free']:
            new_words.append(word)
        else:
            word = find_sense(word, sentense, feature_extraction)
            new_words.append(stemmer.stem(word))
    words = new_words
    if len(words) == 0:
        # if the message contain only stop words, return as is
        return message
    # join words to one string
    return " ".join(words)


def find_sense(word, sentense, feature_extraction):
    """
    find appropriate sense of a word
    :param word: a word to find her sense
    :param sentense: the context of the word
    :param feature_extraction: do we want to aggregate feature by sense of not
    :return: the appropriate sense of the word
    """
    new_word = lesk(sentense, word, pos='n')
    if new_word is not None:
        if feature_extraction:
            all_hypernyms = new_word.hypernyms()
            if all_hypernyms is not None and len(all_hypernyms) > 0:
                new_word = lesk(sentense, word, synsets=all_hypernyms, pos='n')
                word = get_only_word_from_syn(new_word)
        else:
            return get_only_word_from_syn(new_word)
    return word


def get_only_word_from_syn(syn):
    """
    get the word from a given syn
    :param syn: text represent a syn
    :return: the word part of the syn
    """
    return syn._name.split('.')[0]


def compare_our_result(nb, p_nb, wh):
    """
    compare results of method stages
    :param nb: naive bayes stage results
    :param p_nb: pre processing and naive bayes results
    :param wh: whole process results
    :return: None, build graphs
    """
    # convert to percet=ntage
    nb = [x * 100 for x in nb]
    p_nb = [x * 100 for x in p_nb]
    wh = [x * 100 for x in wh]

    plt.clf()

    # set width of bar
    bar_width = 0.20

    # Set position of bar on X axis
    r1 = np.arange(len(nb))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Make the plot
    plt.bar(r1, nb, color='orange', width=bar_width, edgecolor='white', label='NB')
    plt.bar(r2, p_nb, color='blue', width=bar_width, edgecolor='white', label='PreProcessing + NB')
    plt.bar(r3, wh, color='gray', width=bar_width, edgecolor='white', label='PreProcessing + FeatureExtraction + NB')

    plt.ylabel('Precision percentage', fontweight='bold')
    # Add xticks on the middle of the group bars
    plt.xticks([r + bar_width for r in range(len(nb))], ['accuracy', 'precision', 'recall'])

    plt.title("Compare stages result")

    # limit the graph
    plt.ylim(bottom=80, top=104.9)

    # Create legend & Save graphic
    font_p = FontProperties()
    font_p.set_size('small')
    plt.legend(loc='upper left', prop=font_p)

    plt.savefig("Compare stages result.png")


def compare_result_to_papers(wh):
    """
    compare results of wh to the three papers
    :param wh: data results to compare with
    :return: None, build graphs
    """
    plt.clf()
    # calc only accuracy, and convert to percentage
    # full_names = ['our method', 'dea_nb_fp', 'jilian_mtm', 'tiago_dectw']
    names = ['Our Method', 'Paper 1', 'Paper 2', 'Paper 3']
    values = [wh[0] * 100, 98.506, 97.0, 94.2]

    # this is for plotting purpose
    index = np.arange(len(names))
    plt.bar(index, values)
    plt.xlabel('Method', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.xticks(index, names)
    plt.title('Compare Methods')
    # limit the graph
    plt.ylim(bottom=90, top=100)
    plt.savefig('Compare Methods.png')


def handle_nembers(number):
    """
    convert number to phone number or other nomber
    :param number: the number to handle
    :return: phone_number, other_number or None if the text does not represent a number
    """
    number = number.translate(str.maketrans('', '', ','))
    number = number.translate(str.maketrans('', '', '-'))
    if is_number(number):
        if len(number) >= 7:
            return PHONE_NUMBER
        return OTHER_NUMBER
    return None


def pre_process_data(data_df, aggregate_features):
    """
    run preprocessing stage
    :param data_df: the data to preprocess
    :param aggregate_features: a boolean that say if we want to aggregate features
    :return: data after pre process
    """
    print('translate slang, remove punctuation, handle numbers')
    data_df['text'] = data_df['text'].apply(first_msg_translation)
    print('translate word to english, remove stopwords, stemm')
    second_msg_func = lambda x: second_msg_translation(x, aggregate_features)
    data_df['text'] = data_df['text'].apply(second_msg_func)
    return data_df


def prepare_data_for_classify(data_df, random_state=None):
    """
    prepare the data to classifying format
    :param data_df: the data to classify
    :param random_state: a seed to split the data by (keep empty if you want a random seed)
    :return: x_train, x_test, y_train, y_test
    """
    if random_state is None:
        random_state = random.randint(0, 1000)
    # assigned label 1 if spam and 0 if ham
    data_df['label'] = data_df['category'].apply(lambda x: 0 if x == 'ham' else 1)
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data_df['text'], data_df['label'], random_state=random_state)
    return x_train, x_test, y_train, y_test


def bag_of_words(x_train, x_test):
    """
    convert data to counted vector, that count how many times each word appears
    :param x_train: training data
    :param x_test: test data
    :return: the converted counted vectors and the mapping - training, test, cv
    """
    cv = CountVectorizer()
    x_train_cv = cv.fit_transform(x_train)
    x_test_cv = cv.transform(x_test)
    return x_train_cv, x_test_cv, cv


def investigate_data(x_cv, cv):
    """
    investigate data
    :param x_cv: data represented as counted vector
    :param cv: the counted vector that mapped words to vector
    :return: None
    """
    word_freq_df = pd.DataFrame(x_cv.toarray(), columns=cv.get_feature_names())
    top_words_df = pd.DataFrame(word_freq_df.sum()).sort_values(0, ascending=False)


def multinomial_naive_bayes_classifier(x_train_cv, y_train, x_test_cv):
    """
    classify data by naive bayes classifier
    :param x_train_cv: training data represented as counted vector
    :param y_train: true labels of training data
    :param x_test_cv: test data represented as counted vector
    :return: predicted labels
    """
    naive_bayes = MultinomialNB()
    naive_bayes.fit(x_train_cv, y_train)
    predictions = naive_bayes.predict(x_test_cv)
    return predictions


def print_results(y_test, predictions):
    """
    print prediction results
    :param y_test: true labels
    :param predictions: model predictions
    :return: accuracy, precision, recall
    """
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    print('Accuracy score: ', accuracy)
    print('Precision score: ', precision)
    print('Recall score: ', recall)
    return accuracy, precision, recall


def investigate_score(y_test, predictions, title):
    """
    investigate score
    :param y_test: the messages from testing data
    :param predictions: model predictions
    :param title: name of the run we do
    :return: None, create figure that represent the result
    """
    plt.clf()
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False, xticklabels=['ham', 'spam'],
                yticklabels=['ham', 'spam'], fmt='g')
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.title('investigate Method score')
    plt.savefig(title + ' investigate score.png')


def investigate_misses(x_test, y_test, predictions):
    """
    investigate misses
    :param x_test: the messages from testing data
    :param y_test: true test labels
    :param predictions: model predictions
    :return: None
    """
    testing_predictions = []
    for i in range(len(x_test)):
        if predictions[i] == 1:
            testing_predictions.append('spam')
        else:
            testing_predictions.append('ham')
    check_df = pd.DataFrame({'actual_label': list(y_test), 'prediction': testing_predictions, 'text': list(x_test)})
    check_df.replace(to_replace=0, value='ham', inplace=True)
    check_df.replace(to_replace=1, value='spam', inplace=True)


def run_whole_stages(data_df, aggregate_features, title, random_state):
    """
    run whole stage of data prediction, include pre processing
    :param data_df: the data to run on
    :param aggregate_features: boolean that says if we want to use feature extraction
    :param title: the name of the run we do
    :param random_state: a state represent a random seed to split the data
    :return: predicted result
    """
    print('----------pre procsss data----------')
    data_df = pre_process_data(data_df, aggregate_features)
    return run_prediction_stage(data_df, title, random_state)


def run_prediction_stage(data_df, title, random_state):
    """
    run prediction stage
    :param data_df: the data to run predict on
    :param title: the name of the run we do
    :param random_state: a state represent a random seed to split the data
    :return: predicted result
    """
    x_train, x_test, y_train, y_test = prepare_data_for_classify(data_df, random_state)
    x_train_cv, x_test_cv, cv = bag_of_words(x_train, x_test)

    # investigate data
    investigate_data(x_train_cv, cv)
    investigate_data(x_test_cv, cv)

    predictions = run_nb_stage(x_train_cv, y_train, x_test_cv)

    # investigate results
    print('--------------results---------------')
    accuracy, precision, recall = print_results(y_test, predictions)
    investigate_score(y_test, predictions, title)
    investigate_misses(x_test, y_test, predictions)

    return accuracy, precision, recall


def run_nb_stage(x_train_cv, y_train, x_test_cv):
    """
    run naive bayes algorithm
    :param x_train_cv: training data, represented as counted vector
    :param y_train: labels of the training data
    :param x_test_cv: test data, to predict
    :return: preticted labels to the test data
    """
    predictions = multinomial_naive_bayes_classifier(x_train_cv, y_train, x_test_cv)
    return predictions


def has_number(in_string):
    """
    check if a string contains number
    :param in_string: the string to check
    :return: True if contain number, False otherwise
    """
    return any(char.isdigit() for char in in_string)


def is_number(in_string):
    """
    check if the string is a number
    :param in_string: the string to check
    :return: True if number, False otherwise
    """
    return all(char.isdigit() for char in in_string)


def compare_method_stages():
    """
    compare method stages
    :return: None, build graphs
    """
    print('------------------read data-------------------')
    data_df = read_data()
    print('-----------------run only NB------------------')
    nb_results = run_prediction_stage(data_df.copy(), title='Naive Bayes', random_state=234)
    print('------------run preprocess and NB-------------')
    p_nb_results = run_whole_stages(data_df.copy(), aggregate_features=False, title='PreProcess + Naive Bayes',
                                    random_state=123)
    print('---run preprocess feature extraction and NB---')
    cur_wp_results = run_whole_stages(data_df.copy(), aggregate_features=True,
                                      title='PreProcess + Feature Extraction + Naive Bayes', random_state=28)

    compare_our_result(nb_results, p_nb_results, cur_wp_results)
    compare_result_to_papers(cur_wp_results)


if __name__ == "__main__":
    """
    run the whole process
    """
    # General variables
    nltk_stop_words = set(nltk.corpus.stopwords.words('english'))
    stemmer = nltk.SnowballStemmer("english")
    english_dict = enchant.Dict("en_US")
    slang_dict = create_noslang_dict()
    wordnet = LazyCorpusLoader(
        'wordnet',
        WordNetCorpusReader,
        LazyCorpusLoader('omw', CorpusReader, r'.*/wn-data-.*\.tab', encoding='utf8'),
    )
    signs_list = [',', '/', '.', '"', "'", '?', '\\', ':', '(', ')', '*', '-', '=', '+', '&', '^', '$', '%',
                  '#', '@', '!', '`', '~', "'s"]

    # Actual code
    print('------------------read data-------------------')
    data = read_data()
    print('---run preprocess feature extraction and NB---')
    wp_results = run_whole_stages(data.copy(), aggregate_features=True,
                                  title='PreProcess + Feature Extraction + Naive Bayes', random_state=28)
    print('------compare results to papers-----')
    compare_result_to_papers(wp_results)
