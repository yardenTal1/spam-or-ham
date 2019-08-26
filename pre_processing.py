import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import words
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
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans
from gensim.models import Word2Vec


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
    message = message.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
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
            # if not english_dict.check(word):
            #     # print(word)
            #     # continue # TODO return that
            #     suggested_words = english_dict.suggest(word)
            #     if len(suggested_words) > 0:
            #         # TODO check context
            #         word = suggested_words[0]
            #     else:
            #         # TODO if there is no suggestion- remove the word
            #         # continue
            #         pass
            # remove stop words
            if word in nltk_stop_words or word in signs_list:
                continue

            # find a sense that describe the current word
            word = find_sense(word, sentense)

        words.append(stemmer.stem(word))

    if len(words) == 0:
        print('----------\nlen is 0')
        print(message)
        return message
    # join words to one string
    return " ".join(words)


def find_sense(word, sentense):
    new_word = lesk(sentense, word, 'n')
    if new_word is not None:
        return new_word._name.split('.')[0] # TODO remove later! hyper does not work well
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

    return data


def prepare_data_for_classify(data):
    # assigned label 1 if spam and 0 if ham
    data['label'] = data['category'].apply(lambda x: 0 if x =='ham' else 1)
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data['text'], data['label'], random_state = 1)
    return x_train, x_test, y_train, y_test


def bag_of_words(x_train, x_test):
    # cv = CountVectorizer(strip_accents='ascii', token_pattern = u'(?ui)\\b\\w * [a - z] +\\w *\\b', lowercase = True, stop_words ='english')
    cv = CountVectorizer()
    x_train_cv = cv.fit_transform(x_train)
    x_test_cv = cv.transform(x_test)
    return x_train_cv, x_test_cv, cv


def investigate_data(x_cv, cv):
    word_freq_df = pd.DataFrame(x_cv.toarray(), columns=cv.get_feature_names())
    top_words_df = pd.DataFrame(word_freq_df.sum()).sort_values(0, ascending=False)


def multinomial_naive_bayes_classifier(x_train_cv, x_test_cv):
    naive_bayes = MultinomialNB()
    naive_bayes.fit(x_train_cv, y_train)
    predictions = naive_bayes.predict(x_test_cv)
    return predictions


def print_results(y_test, predictions):
    print('Accuracy score: ', accuracy_score(y_test, predictions))
    print('Precision score: ', precision_score(y_test, predictions))
    print('Recall score: ', recall_score(y_test, predictions))


def investigate_score(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar = False, xticklabels = ['ham', 'spam'], yticklabels = ['ham', 'spam'], fmt='g')
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()


def investigate_misses(x_test, y_test, predictions):
    testing_predictions = []
    for i in range(len(x_test)):
        if predictions[i] == 1:
            testing_predictions.append('spam')
        else:
            testing_predictions.append('ham')
    check_df = pd.DataFrame({'actual_label': list(y_test), 'prediction': testing_predictions, 'text':list(x_test)})
    check_df.replace(to_replace=0, value='ham', inplace = True)
    check_df.replace(to_replace=1, value='spam', inplace = True)
    print('finish misses')


def wordEmbbiding(data_text, load=False):
    if load:
        model1 = Word2Vec.load("word2vec.model")
    else:
        data_text = " ".join(data_text)
        data = []

        # tokenize the into words
        for word in nltk.word_tokenize(data_text):
            data.append(word)

        # Create CBOW model
        model1 = Word2Vec(data, min_count=1,size=100, window=5)
        model1.save("word2vec.model")
    print('check')
    return model1


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
    word2vec_model = wordEmbbiding(data['text'], load=True)
    

    x_train, x_test, y_train, y_test = prepare_data_for_classify(data)
    x_train_cv, x_test_cv, cv = bag_of_words(x_train, x_test)

    # investigate data
    investigate_data(x_train_cv, cv)
    investigate_data(x_test_cv, cv)

    predictions = multinomial_naive_bayes_classifier(x_train_cv, x_test_cv)

    # investigate results
    print_results(y_test, predictions)
    investigate_score(y_test, predictions)
    investigate_misses(x_test, y_test, predictions)

    pass # TODO do lesk just after we remove un english words
