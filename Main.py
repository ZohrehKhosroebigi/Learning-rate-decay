# TENSOR 2.0.0
import numpy as np
import csv

from Loadrawdata_show_2 import Loadrawdata_show_train
from Word_tags_to_idx_3 import WordTagsToIdx_train
from Preprocessing_Sentence_to_idx_4 import SentenceToIdx_train

##############################################


"""filename_train='Raw_Data/Test_set.txt'
max_len_input=300
loadrawdata_train=Loadrawdata_show_train()
wrd_tag_vec_train=WordTagsToIdx_train()
wrd_tag_vec_train.word_tags_to_idx(loadrawdata_train.load(filename_train),max_len_input)
sent_idx_train=SentenceToIdx_train()
sent_idx_train.sentence_to_idx(wrd_tag_vec_train,max_len_input)"""
# read from CSV
import csv

reader = csv.reader(open("Raw_Data/X_train.CSV", "r"), delimiter=",")
x = list(reader)
result = np.array(x).astype("float")
X_train = result.reshape(2, 300)

reader = csv.reader(open("Raw_Data/Y_train.CSV", "r"), delimiter=",")
x = list(reader)
result = np.array(x).astype("float")
Y_train = result.reshape(2, 300, 3)
n_X_train = X_train.shape[0]
NUM_EPOCHS = 75
batch_size= 32


def generator(features, labels, batch_size):
     # Create empty arrays to contain batch of features and labels#
     batch_features = np.zeros((batch_size, 2,300))
     batch_labels = np.zeros((batch_size,2, 300, 3))
     while True:
           for i in range(batch_size):
             # choose random index in features
             batch_features[i,] = features[i,]
             batch_labels[i,] = labels[i,]
           yield batch_features, batch_labels


model.fit_generator(generator(X_train, Y_train, batch_size), samples_per_epoch=50, nb_epoch=10)