from random_vector import RandomVec
import numpy as np
import random
import pickle as pkl
import os
from multiprocessing import Process, cpu_count
import codecs
from pprint import pprint
import sys

WORD_DIM = 100
MAX_SENTENCE_LENGTH = 30
random_vector = RandomVec(WORD_DIM)

class WordEmbeddings(object):
    """docstring for WordEmbeddings."""
    def __init__(self, training_file, output_folder="test"):
        super(WordEmbeddings, self).__init__()
        self.analyse_training_file(training_file)
        self.output_folder = output_folder

        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        self.write_config_settings()
        self.processes = []


    def write_config_settings(self):
        with open("{}/config.csv".format(self.output_folder), "w") as output:
            output.write("NUM_FEAT,MAX_CAT,MAX_SENT_LENGTH, WORD_DIM\n")
            output.write("{},{},{},{}\n".format(self.num_features,
                                             self.max_categories,
                                             self.max_length,
                                             self.WORD_DIM))

    def wait(self):
        if not self.processes:
            return
        else:
            for p in self.processes:
                p.join()

    def analyse_training_file(self, training_file):
        """
        Analyse the amount of features in the training dataset. The data should
        look like the following:

        Mikkel feat1 feat2 ... B-PERS
        Vilstrup feat1 feat2 ... I-PERS
        is feat1 feat2 ... O
        the feat1 feat2 ... O
        author feat1 feat2 ... O
        ...

        where feat1 feat2 ... are features related to the word on the left
        It is assumed that the features never change position, and that each feature
        is seperated by a space (" ").
        """
        lines = codecs.open(training_file, encoding="utf-8").readlines()
        lines = [l for l in lines if "-DOCSTART-" not in l]

        num_features = 0
        for line in lines:
            num_features = max(num_features, len(line.split()) - 1) # find the maximum amount of features

        features = [{} for i in range(num_features)] # create a dictionary for each feature

        max_length = 0
        current_length = 0
        # Iterate through all the lines to get the categories of each feature
        for line in lines:
            if line in ['\n', '\r\n']:
                # this is the end of a sentence.
                max_length = min(MAX_SENTENCE_LENGTH, max(max_length, current_length))
                current_length = 0
                continue
            else:
                current_length +=1
                words = line.split()[1:] # discard the word on the left

                for index, word in enumerate(words):
                    if word not in features[index]:
                        features[index][word] = True

        max_categories = 0
        for keys in [f.keys() for f in features]:
            max_categories = max(max_categories, len(keys))

        self.num_features = num_features
        self.max_categories = max_categories
        self.max_length = max_length
        self.features = [f.keys() for f in features]
        pprint(self.features)
        self.WORD_DIM = WORD_DIM + 1  # We add the feature whether it is capital or not
        for feature in self.features[:-1]:
            self.WORD_DIM += len(feature)


    def get_feature_vector(self, category, feature_index):
        onehot = np.zeros(len(self.features[feature_index]))
        # Assign one element in the vector to one, corresponding to the index
        # of the category in features
        onehot[self.features[feature_index].index(category)] = 1
        return onehot

    def get_word_vector(self, word):
        # TODO: This could be improved by instantiating with a word2vec model
        return random_vector.getVec(word)

    def is_capital(self, word):
    	if ord(word[0]) >= 'A' and ord(word[0]) <= 'Z':
    		return np.array([1])
    	else:
    		return np.array([0])

    def create_vectors_async(self, file_name, output_name, feature_output_name):
        p = Process(target=self.create_vectors, args=(file_name, output_name, feature_output_name))
        p.start()
        self.processes.append(p)

    def create_vectors(self, file_name, output_name, feature_output_name):
        words = []
        features = []
        sentences = []
        sentence_features = []


        max_sentence_length = self.max_length
        current_sentence_length = 0

        lines = codecs.open(file_name, encoding="utf-8").readlines()
        lines = [l for l in lines if "-DOCSTART-" not in l]
        
        for line in lines:
            if line in ['\n', '\r\n']:
                # end of line. Make sure all sentences are of equal length
                for _ in range(max_sentence_length - current_sentence_length):
                    words.append(np.zeros(self.WORD_DIM))
                    features.append(np.zeros(len(self.features[-1])))

                # Add current sentence words to sentences and refresh the lists
                sentences.append(words)
                sentence_features.append(features)
                words = []
                features = []
                current_sentence_length = 0
            else:
                # Make sure all lines have the right amount of features
                assert(len(line.split()) == self.num_features + 1)

                # make sure no sentence is longer than max_sentence_length
                if current_sentence_length == max_sentence_length:
                    sentences.append(words)
                    sentence_features.append(features)
                    words = []
                    features = []
                    current_sentence_length = 0



                # get the vector of the word in first position of each line
                word_and_features = line.split()
                temp = []
                temp = np.append(temp, self.get_word_vector(word_and_features[0]))

                # get the feature vector for each feature of the word
                for index, feature in enumerate(word_and_features[1:-1]):
                    temp = np.append(temp, self.get_feature_vector(feature, index))
                """
                Below are some additional features
                """

                temp = np.append(temp, self.is_capital(word_and_features[0]))
                words.append(temp)

                # Add the tag to the tag list
                features.append(self.get_feature_vector(word_and_features[-1], len(self.features)-1))


                current_sentence_length += 1


        # Check there are features for each sentence
        assert(len(sentences) == len(sentence_features))

        print("Storing vectors in folder: {}".format(self.output_folder))
        pkl.dump(sentences,
                 open("{}/{}".format(self.output_folder, output_name),'wb'))

    	pkl.dump(sentence_features,
                 open("{}/{}".format(self.output_folder, feature_output_name),
                      'wb'))

if __name__ == "__main__":
    embeddings = WordEmbeddings("data/ned.train", "data/ned")

    # create embeddings for the three different files
    embeddings.create_vectors_async("data/ned.train", "train_wvec", "train_features")
    embeddings.create_vectors_async("data/ned.testa", "testa_wvec", "testa_features")
    embeddings.create_vectors_async("data/ned.testb", "testb_wvec", "testb_features")

    embeddings.wait()
    print("done")
