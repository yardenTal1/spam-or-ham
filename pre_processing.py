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
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import *


PHONE_NUMBER = 'phonenumber'
OTHER_NUMBER = 'othernumber'
NUM_CLUSTERS = 150


def read_data():
    messages = pd.read_csv("./spam.csv", encoding='latin-1')
    # Drop the extra columns and rename columns
    messages = messages.drop(labels=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    messages.columns = ["category", "text"]
    # messages = messages.iloc[0:100] # TODO remove after testing
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


def first_msg_translation(message):
    # remove punctuation
    message = message.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    # translate slang, and lower words
    words = [slang_dict[word.lower()].lower() if word.lower() in slang_dict else word.lower() for word in message.split()]
    # translate numbers to phonenumber and othernumber
    words = [handle_nembers(word) if has_number(word) and handle_nembers(word) is not None else word for word in words]
    
    message = " ".join(words)
    return message


def second_msg_translation(message, feature_extraction):
    # remove punctuation
    words = []
    sentense = message.split()
    for word in sentense:
        if word in [PHONE_NUMBER, OTHER_NUMBER]:
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
            word = find_sense(word, sentense, feature_extraction)
        words.append(stemmer.stem(word)) # TODO check if and where put the stemming
    if len(words) == 0:
        # if the message contain only stop words, return as is
        return message
    # join words to one string
    return " ".join(words)


def find_sense(word, sentense, feature_extraction):
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


def get_only_word_from_syn(word):
    return word._name.split('.')[0]


def plot_our_bar_graph(nb, p_nb, wh):
    # convert to percet=ntage
    nb = [x*100 for x in nb]
    p_nb = [x*100 for x in p_nb]
    wh = [x*100 for x in wh]

    plt.clf()

    # set width of bar
    barWidth = 0.20

    # Set position of bar on X axis
    r1 = np.arange(len(nb))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, nb, color='red', width=barWidth, edgecolor='white', label='NB')
    plt.bar(r2, p_nb, color='blue', width=barWidth, edgecolor='white', label='PreProcessing + NB')
    plt.bar(r3, wh, color='green', width=barWidth, edgecolor='white', label='PreProcessing + FeatureExtraction + NB')

    # Add xticks on the middle of the group bars
    plt.xlabel('Classifier', fontweight='bold')
    plt.ylabel('Score', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(nb))], ['accuracy', 'precision', 'recall'])

    # limit the graph between 80 to 100
    plt.ylim(bottom=80, top=100)

    # Create legend & Show graphic
    plt.legend()
    plt.show()
    plt.savefig("our result.png")


def plot_estimated_bar_graph(wh):
    plt.clf()
    # calc only accuracy, and convert to percentage
    names = ['our methed', 'jialin_mtm', 'jialin_svm', 'tiago_dectw', 'tiago_bnb', 'dea_nb', 'dea_nb_fp']
    values = [wh[0]*100, 97.0, 96.0, 94.2, 91.2, 98.481, 98.506]

    # this is for plotting purpose
    index = np.arange(len(names))
    plt.bar(index, values)
    plt.xlabel('Classifier', fontweight='bold')
    plt.ylabel('Score', fontweight='bold')
    plt.xticks(index, names, rotation=30)
    plt.title('Estimated result')
    # limit the graph between 90 to 100
    plt.ylim(bottom=90, top=100)
    plt.show()
    plt.savefig('estimated_result.png')


def handle_nembers(number):
    number = number.translate(str.maketrans('', '', ','))
    number = number.translate(str.maketrans('', '', '-'))
    if is_number(number):
        if len(number) >= 7:
            return PHONE_NUMBER
        return OTHER_NUMBER
    return None


def pre_process_data(data, feature_extraction):
    print('translate slang, remove punctuation, handle numbers')
    data['text'] = data['text'].apply(first_msg_translation)
    print('translate word to english, remove stopwords, stemm')
    second_msg_func = lambda x: second_msg_translation(x, feature_extraction)
    data['text'] = data['text'].apply(second_msg_func)
    return data


def prepare_data_for_classify(data, random_state):
    # assigned label 1 if spam and 0 if ham
    data['label'] = data['category'].apply(lambda x: 0 if x =='ham' else 1)
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data['text'], data['label'], random_state = random_state)
    return x_train, x_test, y_train, y_test


def bag_of_words(x_train, x_test):
    cv = CountVectorizer()
    x_train_cv = cv.fit_transform(x_train)
    x_test_cv = cv.transform(x_test)
    return x_train_cv, x_test_cv, cv


def investigate_data(x_cv, cv):
    word_freq_df = pd.DataFrame(x_cv.toarray(), columns=cv.get_feature_names())
    top_words_df = pd.DataFrame(word_freq_df.sum()).sort_values(0, ascending=False)


def multinomial_naive_bayes_classifier(x_train_cv, y_train, x_test_cv):
    naive_bayes = MultinomialNB()
    naive_bayes.fit(x_train_cv, y_train)
    predictions = naive_bayes.predict(x_test_cv)
    return predictions


def print_results(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    print('Accuracy score: ', accuracy)
    print('Precision score: ', precision)
    print('Recall score: ', recall)
    return accuracy, precision, recall


def investigate_score(y_test, predictions, title):
    plt.clf()
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar = False, xticklabels = ['ham', 'spam'], yticklabels = ['ham', 'spam'], fmt='g')
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.savefig(title + ' investigate score.png')


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


def wordEmbbiding(data_text, load=False):
    if load:
        model = Word2Vec.load("word2vec.model")
    else:
        data_text = list(data_text)
        data = []

        # tokenize the into words
        for sent in data_text:
            for word in nltk.word_tokenize(sent):
                data.append([word])

        # Create CBOW model
        model = Word2Vec(data, min_count=1,size=100, window=5)
        model.save("word2vec.model")
    return model


def clustering_with_k_means(model):
    X = model[model.wv.vocab]
    k_clusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
    assigned_clusters = k_clusterer.cluster(X, assign_clusters=True)
    return assigned_clusters, k_clusterer


def print_clustering(model, assigned_clusters, k_clusterer):
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        print(word + ":" + str(assigned_clusters[i]))


def print_means(k_clusterer):
    print('Means:', k_clusterer.means())


def change_msg_by_cluster(message, cluster_words, k_clusterer, model):
    # translate slang, and lower words
    words = message.split()
    new_words = []
    for word in words:
        word_class = k_clusterer.classify(model.wv[word])
        new_words.append(cluster_words[word_class])

    message = " ".join(new_words)
    return message


def change_data_by_cluster(data_text, k_clusterer, model, assigned_clusters):
    cluster_words = [-1]*NUM_CLUSTERS

    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        cur_class = assigned_clusters[i]
        if cluster_words[cur_class] == -1:
            cluster_words[cur_class] = word

    apply_func = lambda x: change_msg_by_cluster(x, cluster_words, k_clusterer, model)
    return data_text.apply(apply_func)


def run_whole_stages(data, feature_extraction, title, random_state):
    print('----------pre procsss data----------')
    data = pre_process_data(data, feature_extraction)

    # load=False
    # print('----------build word2vec model load=%s----------' % str(load))
    # word2vec_model = wordEmbbiding(data['text'], load=load)
    # print('----------assign clusters----------')
    # assigned_clusters, k_clusterer = clustering_with_k_means(word2vec_model)
    # data['text'] = change_data_by_cluster(data['text'], k_clusterer, word2vec_model, assigned_clusters)

    return run_prediction_stage(data, title, random_state)


def run_prediction_stage(data, title, random_state):
    x_train, x_test, y_train, y_test = prepare_data_for_classify(data, random_state)
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
    predictions = multinomial_naive_bayes_classifier(x_train_cv, y_train, x_test_cv)
    return predictions


if __name__ == "__main__":
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
    print('-----------------run only NB------------------')
    nb = run_prediction_stage(data.copy(), title='Naive Bayes', random_state=234)
    print('------------run preprocess and NB-------------')
    p_nb = run_whole_stages(data.copy(),
                                                                  feature_extraction=False, title='PreProcess + Naive Bayes', random_state=123)
    print('---run preprocess feature extraction and NB---')
    wh = run_whole_stages(data.copy(), feature_extraction=True,
                                                            title='PreProcess + Feature Extraction + Naive Bayes', random_state=28)
    print('finish')

    plot_our_bar_graph(nb, p_nb, wh)
    plot_estimated_bar_graph(wh)
