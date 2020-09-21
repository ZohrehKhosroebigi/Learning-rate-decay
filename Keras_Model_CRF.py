# load and evaluate a saved model
from keras.models import load_model
from keras_contrib import *
from keras_contrib.layers.crf import CRF,crf_loss
from keras_contrib.metrics import crf_accuracy

from keras_contrib.layers import CRF
from keras_contrib.losses import  crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
import numpy as np

from Loadrawdata_show_2 import Loadrawdata_show_train
from Word_tags_to_idx_3 import WordTagsToIdx_train
from Preprocessing_Sentence_to_idx_4 import SentenceToIdx_train
keras_model= load_model('models_keras/mykerasmodel.h5',custom_objects={'CRF':CRF,
                                                  'crf_loss':crf_loss,
                                                  'crf_viterbi_accuracy':crf_viterbi_accuracy})
#################################################################
keras_model.summary()
# load dataset
filename_train='Raw_Data/Test_set.txt'
max_len_input=300

loadrawdata_train=Loadrawdata_show_train()
wrd_tag_vec_train=WordTagsToIdx_train()
wrd_tag_vec_train.word_tags_to_idx(loadrawdata_train.load(filename_train),max_len_input)
sent_idx_train=SentenceToIdx_train()
sent_idx_train.sentence_to_idx(wrd_tag_vec_train,max_len_input)

preds = keras_model.predict(sent_idx_train.X_train, verbose=1)
# show the inputs and predicted outputs

ynew = keras_model.predict(sent_idx_train.X_train)
# show the inputs and predicted outputs
print("X=%s, Predicted=%s" % (sent_idx_train.X_train[0], ynew[0]))


idx_to_tag={1: 'O', 2: 'B-ORG', 3: 'CB-LOC', 4: 'B-PER', 5: 'I-LOC', 6: 'I-ORG', 7: 'I-PER', 8: 'B-loc', 9: 'I-loc', 10: 'B-pers', 11: 'B-org', 12: 'I-pers', 13: 'B-event', 14: 'I-event', 15: 'B-fac', 16: 'I-fac', 17: 'I-org', 18: 'B-pro', 19: 'I-pro', 20: 'B-DAT', 21: 'I-DAT', 22: 'B-EVE', 23: 'I-EVE', 0: 'PAD'}


print("{:15} {:5}    ".format("Word", "Pred"))
for  words,predictions in zip(sent_idx_train.X_train,preds):
        #print("prediction....."+str((preds)))
        for word, prediction in zip(words,predictions):
                print("_" * 20)
                predict_val=np.max(prediction)
                predict_position=np.argmax(prediction)
                predict_tag=idx_to_tag[predict_position]
                word=wrd_tag_vec_train.idx2word[word]
                print("{:15}:{:5}     ".format(word, predict_tag))


