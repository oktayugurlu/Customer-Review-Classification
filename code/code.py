import pandas as pd
from math import log
import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt                              #For visualization
import seaborn as sns; sns.set()

from itertools import chain

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

##### Data reading and dataframe convertion #######
data = pd.read_csv("all_sentiment_shuffled.txt", delimiter="\n", header=None)
data["id"] = data.iloc[:,0].str.split(" ").map(lambda x: x[2])
data["category"] = data.iloc[:,0].str.split(" ").map(lambda x: x[0])
data["tokens"] = data.iloc[:,0].str.split(" ").map(lambda x: " ".join(x[3:]))
data["CLASS"] = data.iloc[:,0].str.split(" ").map(lambda x: x[1])
del data[0]

##### Normalization of CLASS column #######
class_id_dict = {}
id = 0
for class_name in data["CLASS"].unique():
    class_id_dict[class_name] = id
    id +=1

data["CLASS"] = data["CLASS"].map(lambda x: class_id_dict[x]) # The negative is 0 and the positive is 1

###### APPLYIG COUNT VECTORIZING TO THE DATA ######
vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words=ENGLISH_STOP_WORDS)
vector = vectorizer.fit_transform(data["tokens"])
vector_df = pd.DataFrame(vector.toarray())
words_list = list(vectorizer.vocabulary_.keys())
words_list.sort()
vector_df["CLASS"] = data["CLASS"]
words_list.append("CLASS")
vector_df.columns = words_list





#I have split the data as train and test by using this method

def train_test_split_df(data, pure_data, test_size=0.20):
    test_start_index = data.shape[0] - int(data.shape[0] * test_size)
    return (data.iloc[0:test_start_index,:], pure_data.iloc[test_start_index:,:].reset_index())


class NaiveBayesClassifier:
    
    def __init__(self):
        self.__feature_count_dictionary_for_classes = {}
        self.__classes_word_numbers_dictionary = {}
        self.__dataset = pd.DataFrame()
        self.__prediction_list = []
        self.__input_df = pd.DataFrame()
        self.__class_word_score_dict = {}
        self.__class_prediction_probability = {} # In this directory, each class includes probability of words used in prediction. We will use this in modul analysis.
        self.__is_tfidf = False
        self.__is_bigram = False
        self.__stop_words = None
        
    ###################### Getter functions of some variables ######################
    def get_word_probability(self):
        return self.__class_prediction_probability
    def get_word_probability_differences(self, is_words_in_true_predicted=True): # This method returns the probability difference of words in different classes placed in true predicted documents
                                                # We need to filter unseen words in true predicted rows of input.
        word_probability_difference_for_class = {}
        for class_name in self.__dataset["CLASS"].unique():
    
            word_probability_difference_for_class[class_name] = {}
            vectorizer1 = self.__initialize_count_vectorizer(self.__is_bigram, stop_words=self.__stop_words, is_tfidf=self.__is_tfidf)
            if is_words_in_true_predicted:
                vectorizer1.fit(self.__input_df[( self.__input_df.loc[:,'CLASS']- pd.DataFrame(self.__prediction_list)[0] == 0) & (self.__input_df['CLASS']==class_name)]["tokens"])
            else:
                vectorizer1.fit(self.__input_df[( self.__input_df.loc[:,'CLASS']- pd.DataFrame(self.__prediction_list)[0] != 0) & (self.__input_df['CLASS']==class_name)]["tokens"])

            tokens_occured_in_true_predicted_for_a_class = list(vectorizer1.vocabulary_.keys())
            
            for word in list(self.__class_prediction_probability[class_name].keys()):
                if word in tokens_occured_in_true_predicted_for_a_class:
                    probability_difference_with_other_words = []
                    for other_class_name in np.delete(self.__dataset["CLASS"].unique(),  self.__dataset["CLASS"].unique().tolist().index(class_name)):
                        probability_difference_with_other_words.append(self.__class_prediction_probability[other_class_name][word]-self.__class_prediction_probability[class_name][word])

                    word_probability_difference_for_class[class_name][word] = max(probability_difference_with_other_words)

        return word_probability_difference_for_class
    
    def get_prediction_list(self):
        return self.__prediction_list
        
    def get_input_df(self):
        return self.__input_df
    
    def get_feature_df(self):
        return self.__feature_count_dictionary_for_classes
    
    def accuracy(self):
        actual_labels_df = self.__input_df["CLASS"].copy()
        predicted_labels_df = pd.DataFrame(self.__prediction_list)
        # Because the actual labels index is not start from 0, I resett its index
        actual_labels_df = actual_labels_df.reset_index(drop=True).to_frame()
        actual_labels_df.columns = [0]
        difference = predicted_labels_df.sub(actual_labels_df)
        return difference[difference.iloc[:,0]==0].shape[0] / actual_labels_df.shape[0]

    
    ###################### Naive Bayes with Laplace Smoothing ######################

    def train(self, dataset): #The classes should be placed end of dataframe in trained data
        for class_name in dataset["CLASS"].unique():
            self.__class_prediction_probability[class_name] = {} 
            self.__feature_count_dictionary_for_classes[class_name] = {}
            class_df = dataset[dataset["CLASS"]==class_name]
            for word in dataset.iloc[:,:-1].columns.values.tolist():
                self.__feature_count_dictionary_for_classes[class_name][word] = np.sum(class_df[word]) +1
        self.__dataset = dataset

    
    def predict_with_laplace_smoothing(self, input_df, is_bigram=False, stop_words=None, is_tfidf=False):

        self.__is_tfidf = is_tfidf
        self.__is_bigram = is_bigram
        self.__stop_words = stop_words
        
        self.__input_df = input_df.copy()
        total_words_in_classes, class_probabilities = self.__calculate_probailities_of_classes_and_total_words_in_each_classes()
        self.__do_laplace_for_unseen_data_in_trian(is_bigram, total_words_in_classes, stop_words=stop_words, is_tfidf=is_tfidf)
                
        for row in input_df.index.tolist():
            #If the bigram is used, max feature is set as 50576 because of memory issue
            vectorizer1 = self.__initialize_count_vectorizer(is_bigram, stop_words=stop_words, is_tfidf=is_tfidf)
            
            #Fit count vectorizer for each row in test data
            vector_train = vectorizer1.fit_transform([input_df["tokens"][row]])
            splitted_tokens = list(vectorizer1.vocabulary_.keys())
            probability_dict = {}
            
            #Calculate prior for each class
            for class_name in list(self.__feature_count_dictionary_for_classes.keys()):
                probability_joint = class_probabilities[class_name]
                for token in splitted_tokens:
                    probability_of_word_given_class = self.__feature_count_dictionary_for_classes[class_name][token] / total_words_in_classes[class_name]
                    probability_joint += np.log(probability_of_word_given_class)
                    self.__class_prediction_probability[class_name][token] = probability_of_word_given_class
                probability_dict[class_name] = probability_joint
                
            #Prediction
            predicted_label = max(probability_dict.items(), key=operator.itemgetter(1))[0]
            self.__prediction_list.append(predicted_label)       
    
    ###################### Naive Bayes without Laplace Smoothing ######################
    def train_without_laplace(self, dataset): #The classes should be placed end of dataframe in trained data
        for class_name in dataset["CLASS"].unique(): 
            self.__feature_count_dictionary_for_classes[class_name] = {}
            class_df = dataset[dataset["CLASS"]==class_name]
            for word in dataset.iloc[:,:-1].columns.values.tolist():
                self.__feature_count_dictionary_for_classes[class_name][word] = np.sum(class_df[word])
        self.__dataset = dataset

    
    def predict_without_laplace_smoothing(self, input_df, is_bigram=False, stop_words=None, is_tfidf=False):

        self.__input_df = input_df.copy()
        total_words_in_classes, class_probabilities = self.__calculate_probailities_of_classes_and_total_words_in_each_classes()
                
        for row in input_df.index.tolist():
            #If the bigram is used, max feature is set as 50576 because of memory issue
            vectorizer1 = self.__initialize_count_vectorizer(is_bigram, stop_words=stop_words, is_tfidf=is_tfidf)
            
            #Fit count vectorizer for each row in test data
            vector_train = vectorizer1.fit_transform([input_df["tokens"][row]])
            splitted_tokens = list(vectorizer1.vocabulary_.keys())
            probability_dict = {}
            
            #Calculate prior for each class
            for class_name in list(self.__feature_count_dictionary_for_classes.keys()):
                probability_joint = class_probabilities[class_name]
                for token in splitted_tokens:
                    if token not in self.__feature_count_dictionary_for_classes[class_name]:
                        probability_joint += 0
                    else:
                        probability_joint += np.log(self.__feature_count_dictionary_for_classes[class_name][token] / total_words_in_classes[class_name])
                probability_dict[class_name] = probability_joint
                
            #Prediction
            predicted_label = max(probability_dict.items(), key=operator.itemgetter(1))[0]
            self.__prediction_list.append(predicted_label)       
            
    
    ###################### Private methods ######################
    def __do_laplace_for_unseen_data_in_trian(self, is_bigram, total_words_in_classes, stop_words=None, is_tfidf=False):
        key_list = list(self.__feature_count_dictionary_for_classes.keys())
        if(is_bigram):
            for class_name in key_list:
                vectorizer = None
                if(is_tfidf):
                    vectorizer = TfidfVectorizer(ngram_range=(2,2), stop_words=stop_words)
                else:
                    vectorizer = CountVectorizer(ngram_range=(2,2), stop_words=stop_words)

                for token in list(vectorizer.fit(self.__input_df["tokens"]).vocabulary_.keys()):
                    if token not in self.__feature_count_dictionary_for_classes[class_name]:
                        self.__feature_count_dictionary_for_classes[class_name][token] = 1
                        total_words_in_classes[class_name] += 1
        else:
            for class_name in key_list:
                for token in list(CountVectorizer(ngram_range=(1,1), stop_words=stop_words).fit(self.__input_df["tokens"]).vocabulary_.keys()):
                    if token not in self.__feature_count_dictionary_for_classes[class_name]:
                        self.__feature_count_dictionary_for_classes[class_name][token] = 1
                        total_words_in_classes[class_name] += 1
                        
    def __calculate_probailities_of_classes_and_total_words_in_each_classes(self):
        class_probabilities = {}
        total_words_in_classes = {}
        for class_name in list(self.__feature_count_dictionary_for_classes.keys()):
            class_probabilities[class_name] = np.log(self.__dataset[ self.__dataset["CLASS"] == class_name].shape[0] / self.__dataset.shape[0])
            total_words_in_classes[class_name] = sum(self.__feature_count_dictionary_for_classes[class_name].values())        
        return (total_words_in_classes, class_probabilities)
    
    def __initialize_count_vectorizer(self, is_bigram, stop_words=None, is_tfidf=False):
        vectorizer1 = None
        if(is_bigram):
            if(is_tfidf):
                vectorizer1 = TfidfVectorizer(max_features=50576, ngram_range=(2,2), stop_words=stop_words)
            else:
                vectorizer1 = CountVectorizer(max_features=50576, ngram_range=(2,2), stop_words=stop_words )   
        else:
            if(is_tfidf):
                vectorizer1 = TfidfVectorizer(ngram_range=(1,1), stop_words=stop_words )
            else:
                vectorizer1 = CountVectorizer(ngram_range=(1,1), stop_words=stop_words )
            
        return vectorizer1


### UNIGRAM DATA PREPARING #####

vectorizer1 = CountVectorizer()
vector_train = vectorizer1.fit_transform(data["tokens"])
df = pd.DataFrame(vector_train.toarray())
listt = list(vectorizer1.vocabulary_.keys())
listt.sort()
df.columns = listt
df["CLASS"] = data["CLASS"]

train_df, test_df = train_test_split_df(df, data, test_size=0.20)

model = NaiveBayesClassifier()
model.train_without_laplace(train_df)
model.predict_without_laplace_smoothing(test_df, is_bigram=False, stop_words=None)
print("Accuracy with UNIGRAM without laplace: ",model.accuracy())

### BIGRAM DATA PREPARING #####
test_size= 0.20
test_start_index = data.shape[0] - int(data.shape[0] * test_size)

vectorizer_bigram = CountVectorizer(max_features=50576, ngram_range = (2, 2))
bigram_vector_train = vectorizer_bigram.fit_transform(data.iloc[:test_start_index,-2])

bigram_df = pd.DataFrame(bigram_vector_train.toarray())
listt = list(vectorizer_bigram.vocabulary_.keys())
listt.sort()
bigram_df.columns = listt
bigram_df["CLASS"] = data["CLASS"]

listt = list(vectorizer_bigram.vocabulary_.keys())
listt.sort()

bigram_train_df = bigram_df
bigram_test_df = data.iloc[test_start_index:,:]

#I have drop this index because of the comment only includes 1 word
bigram_test_df = bigram_test_df.reset_index()
bigram_test_df = bigram_test_df.drop([1939])
bigram_test_df = bigram_test_df.reset_index()


model = NaiveBayesClassifier()
model.train_without_laplace(bigram_train_df)
model.predict_without_laplace_smoothing(bigram_test_df, is_bigram=True, stop_words=None)
print("Accuracy with BIGRAM without laplace",model.accuracy())

### UNIGRAM DATA PREPARING #####

vectorizer1 = CountVectorizer()
vector_train = vectorizer1.fit_transform(data["tokens"])
df = pd.DataFrame(vector_train.toarray())
listt = list(vectorizer1.vocabulary_.keys())
listt.sort()
df.columns = listt
df["CLASS"] = data["CLASS"]

train_df, test_df = train_test_split_df(df, data, test_size=0.20)


model_unigram = NaiveBayesClassifier()
model_unigram.train(train_df)
model_unigram.predict_with_laplace_smoothing(test_df, is_bigram=False, stop_words=None)
acc_unigram = model_unigram.accuracy()
print("Accuracy with UNIGRAM: ",acc_unigram)


### UNIGRAM DATA PREPARING #####

vectorizer1 = TfidfVectorizer()
vector_train = vectorizer1.fit_transform(data["tokens"])
df = pd.DataFrame(vector_train.toarray())
listt = list(vectorizer1.vocabulary_.keys())
listt.sort()
df.columns = listt
df["CLASS"] = data["CLASS"]

train_df, test_df = train_test_split_df(df, data, test_size=0.20)


model_unigram_tfidf = NaiveBayesClassifier()
model_unigram_tfidf.train(train_df)
model_unigram_tfidf.predict_with_laplace_smoothing(test_df, is_bigram=False, stop_words= None, is_tfidf=True)
acc_unigram_tfidf = model_unigram_tfidf.accuracy()
print("Accuracy with UNIGRAM and TF-IDF: ",acc_unigram_tfidf)


### UNIGRAM DATA PREPARING #####

vectorizer1 = CountVectorizer(stop_words= ENGLISH_STOP_WORDS)
vector_train = vectorizer1.fit_transform(data["tokens"])
df = pd.DataFrame(vector_train.toarray())
listt = list(vectorizer1.vocabulary_.keys())
listt.sort()
df.columns = listt
df["CLASS"] = data["CLASS"]

train_df, test_df = train_test_split_df(df, data, test_size=0.20)

model_unigram_stop_words = NaiveBayesClassifier()
model_unigram_stop_words.train(train_df)
model_unigram_stop_words.predict_with_laplace_smoothing(test_df, is_bigram=False, stop_words= ENGLISH_STOP_WORDS)
acc_unigram_stop_words = model_unigram_stop_words.accuracy()
print("Accuracy with UNIGRAM and without STOP WORDS: ",acc_unigram_stop_words)

### UNIGRAM DATA PREPARING #####

vectorizer1 = TfidfVectorizer(stop_words= ENGLISH_STOP_WORDS)
vector_train = vectorizer1.fit_transform(data["tokens"])
df = pd.DataFrame(vector_train.toarray())
listt = list(vectorizer1.vocabulary_.keys())
listt.sort()
df.columns = listt
df["CLASS"] = data["CLASS"]

train_df, test_df = train_test_split_df(df, data, test_size=0.20)
model_unigram_stop_words_tfidf = NaiveBayesClassifier()
model_unigram_stop_words_tfidf.train(train_df)
model_unigram_stop_words_tfidf.predict_with_laplace_smoothing(test_df, is_bigram=False, stop_words= ENGLISH_STOP_WORDS, is_tfidf=True)
acc_unigram_stop_words_tfidf = model_unigram_stop_words_tfidf.accuracy()
print("Accuracy with UNIGRAM, TF-IDF and without STOP WORDS: ", acc_unigram_stop_words_tfidf)


### BIGRAM DATA PREPARING #####
test_size= 0.20
test_start_index = data.shape[0] - int(data.shape[0] * test_size)

vectorizer_bigram = CountVectorizer(max_features=50576, ngram_range = (2, 2))
bigram_vector_train = vectorizer_bigram.fit_transform(data.iloc[:test_start_index,-2])

bigram_df = pd.DataFrame(bigram_vector_train.toarray())
listt = list(vectorizer_bigram.vocabulary_.keys())
listt.sort()
bigram_df.columns = listt
bigram_df["CLASS"] = data["CLASS"]

listt = list(vectorizer_bigram.vocabulary_.keys())
listt.sort()

bigram_train_df = bigram_df
bigram_test_df = data.iloc[test_start_index:,:]
bigram_test_df = bigram_test_df.reset_index()

#I have drop this index because of the comment only includes 1 word
bigram_test_df = bigram_test_df.drop([1939])
bigram_test_df = bigram_test_df.reset_index()

model_bigram = NaiveBayesClassifier()
model_bigram.train(bigram_train_df)
model_bigram.predict_with_laplace_smoothing(bigram_test_df, is_bigram=True, stop_words=None)
acc_bigram = model_bigram.accuracy()
print("Accuracy with BIGRAM", acc_bigram)

### BIGRAM DATA PREPARING #####
test_size= 0.20
test_start_index = data.shape[0] - int(data.shape[0] * test_size)

vectorizer_bigram = TfidfVectorizer(max_features=50576, ngram_range = (2, 2))
bigram_vector_train = vectorizer_bigram.fit_transform(data.iloc[:test_start_index,-2])

bigram_df = pd.DataFrame(bigram_vector_train.toarray())
listt = list(vectorizer_bigram.vocabulary_.keys())
listt.sort()
bigram_df.columns = listt
bigram_df["CLASS"] = data["CLASS"]

listt = list(vectorizer_bigram.vocabulary_.keys())
listt.sort()

bigram_train_df = bigram_df
bigram_test_df = data.iloc[test_start_index:,:]
bigram_test_df = bigram_test_df.reset_index()

#I have drop this index because of the comment only includes 1 word
bigram_test_df = bigram_test_df.drop([1939])
bigram_test_df = bigram_test_df.reset_index()

model_bigram_tfidf = NaiveBayesClassifier()
model_bigram_tfidf.train(bigram_train_df)
model_bigram_tfidf.predict_with_laplace_smoothing(bigram_test_df, is_bigram=True, stop_words=None, is_tfidf=True )
acc_bigram_tfidf = model_bigram_tfidf.accuracy()
print("Accuracy with BIGRAM and TF-IDF", acc_bigram_tfidf)

### BIGRAM DATA PREPARING #####
test_size= 0.20
test_start_index = data.shape[0] - int(data.shape[0] * test_size)

vectorizer_bigram = CountVectorizer(max_features=50576, ngram_range = (2, 2), stop_words= ENGLISH_STOP_WORDS)
bigram_vector_train = vectorizer_bigram.fit_transform(data.iloc[:test_start_index,-2])

bigram_df = pd.DataFrame(bigram_vector_train.toarray())
listt = list(vectorizer_bigram.vocabulary_.keys())
listt.sort()
bigram_df.columns = listt
bigram_df["CLASS"] = data["CLASS"]

listt = list(vectorizer_bigram.vocabulary_.keys())
listt.sort()

bigram_train_df = bigram_df
bigram_test_df = data.iloc[test_start_index:,:]
bigram_test_df = bigram_test_df.reset_index()

#I have drop this index because of the comment only includes 1 word
bigram_test_df = bigram_test_df.drop([1939])
bigram_test_df = bigram_test_df.reset_index()

model_bigram_stop_words = NaiveBayesClassifier()
model_bigram_stop_words.train(bigram_train_df)
model_bigram_stop_words.predict_with_laplace_smoothing(bigram_test_df, is_bigram=True, stop_words=ENGLISH_STOP_WORDS)
acc_bigram_stop_words = model_bigram_stop_words.accuracy()
print("Accuracy with BIGRAM and without STOP WORDS", acc_bigram_stop_words)

### BIGRAM DATA PREPARING #####
test_size= 0.20
test_start_index = data.shape[0] - int(data.shape[0] * test_size)

vectorizer_bigram = TfidfVectorizer(max_features=50576, ngram_range = (2, 2), stop_words= ENGLISH_STOP_WORDS)
bigram_vector_train = vectorizer_bigram.fit_transform(data.iloc[:test_start_index,-2])

bigram_df = pd.DataFrame(bigram_vector_train.toarray())
listt = list(vectorizer_bigram.vocabulary_.keys())
listt.sort()
bigram_df.columns = listt
bigram_df["CLASS"] = data["CLASS"]

listt = list(vectorizer_bigram.vocabulary_.keys())
listt.sort()

bigram_train_df = bigram_df
bigram_test_df = data.iloc[test_start_index:,:]
bigram_test_df = bigram_test_df.reset_index()

#I have drop this index because of the comment only includes 1 word
bigram_test_df = bigram_test_df.drop([1939])
bigram_test_df = bigram_test_df.reset_index()

model_bigram_stop_words_tf_idf = NaiveBayesClassifier()
model_bigram_stop_words_tf_idf.train(bigram_train_df)
model_bigram_stop_words_tf_idf.predict_with_laplace_smoothing(bigram_test_df, is_bigram=True, stop_words=ENGLISH_STOP_WORDS, is_tfidf=True)
acc_bigram_stop_words_tf_idf = model_bigram_stop_words_tf_idf.accuracy()
print("Accuracy with BIGRAM, TF-IDF and without STOP WORDS: ", acc_bigram_stop_words_tf_idf)


########### MODULE ANALYSIS #############
probability_true_predicted = model_unigram_tfidf.get_word_probability_differences()

print("1st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[0])
print("2st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[1])
print("3st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[2])
print("4st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[3])
print("5st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[4])
print("6st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[5])
print("7st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[6])
print("8st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[7])
print("9st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[8])
print("10st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[9])

print("1st most affect of absence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=True)[0])
print("2st most affect of absence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=True)[1])
print("3st most affect of absence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=True)[2])
print("4st most affect of absence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=True)[3])
print("5st most affect of absence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=True)[4])
print("6st most affect of absence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=True)[5])
print("7st most affect of absence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=True)[6])
print("8st most affect of absence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=True)[7])
print("9st most affect of absence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=True)[8])
print("10st most affect of absence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=True)[9])

print("1st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[0])
print("2st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[1])
print("3st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[2])
print("4st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[3])
print("5st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[4])
print("6st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[5])
print("7st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[6])
print("8st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[7])
print("9st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[8])
print("10st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[9])

print("1st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=True)[0])
print("2st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=True)[1])
print("3st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=True)[2])
print("4st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=True)[3])
print("5st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=True)[4])
print("6st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=True)[5])
print("7st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=True)[6])
print("8st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=True)[7])
print("9st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=True)[8])
print("10st most affect of presence to predict negative: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=True)[9])



############## STOP WORDS #############
probability_true_predicted = model_unigram_stop_words_tfidf.get_word_probability_differences()

print("1st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[0])
print("2st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[1])
print("3st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[2])
print("4st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[3])
print("5st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[4])
print("6st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[5])
print("7st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[6])
print("8st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[7])
print("9st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[8])
print("10st most affect of presence to predict positive: ",sorted(probability_true_predicted[1], key=probability_true_predicted[1].get, reverse=False)[9])


print("1st most affect of presence to predict positive: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[0])
print("2st most affect of presence to predict positive: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[1])
print("3st most affect of presence to predict positive: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[2])
print("4st most affect of presence to predict positive: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[3])
print("5st most affect of presence to predict positive: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[4])
print("6st most affect of presence to predict positive: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[5])
print("7st most affect of presence to predict positive: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[6])
print("8st most affect of presence to predict positive: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[7])
print("9st most affect of presence to predict positive: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[8])
print("10st most affect of presence to predict positive: ",sorted(probability_true_predicted[0], key=probability_true_predicted[0].get, reverse=False)[9])



