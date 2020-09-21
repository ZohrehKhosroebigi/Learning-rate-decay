# Tensor version is 1.13.1
#kears 2.2.4
#8954a53000bd8db0944cfc991114bc1b6cf2437e
from sklearn.model_selection import KFold
from Create_LSTM_model_5 import CreatingModel
from Train_model_7 import Trainmodel
from Compile_model_6 import Compilemodel
from writing import Writelogs
from Loadrawdata_show_22_test import Loadrawdata_show_test
from Word_tags_to_idx_33_test import WordTagsToIdx_test
from Preprocessing_Sentence_to_idx_44_test import SentenceToIdx_test
from Loadrawdata_show_2 import Loadrawdata_show_train
from Word_tags_to_idx_3 import WordTagsToIdx_train
from Preprocessing_Sentence_to_idx_4 import SentenceToIdx_train
from Evaluate_model_8 import Evaluate_model
from configlearning import ConfigLearningDecay
import numpy as np
from sklearn.model_selection import StratifiedKFold
filename_train='Raw_Data/Test_set.txt'
#filename_test='Raw_Data/Test_set.txt'
#Hyper paramet
n_fold=2
max_len_input=30
embedding_vector=10
epoch=10
#schedule="step"
#schedule="linear"
schedule="poly"
#schedule="None"
bachsize=64
seed = 7
np.random.seed(seed)
unit1=1
unit2=2
#preparing TRAIN TEST
loadrawdata_train=Loadrawdata_show_train()
wrd_tag_vec_train=WordTagsToIdx_train()
wrd_tag_vec_train.word_tags_to_idx(loadrawdata_train.load(filename_train),max_len_input)
sent_idx_train=SentenceToIdx_train()
sent_idx_train.sentence_to_idx(wrd_tag_vec_train,max_len_input)

#preparing TEST SET
"""loadrawdata_test=Loadrawdata_show_test()
wrd_tag_vec_test=WordTagsToIdx_test()
wrd_tag_vec_test.word_tags_to_idx(loadrawdata_test.load(filename_test),max_len_input)
sent_idx_test=SentenceToIdx_test()
sent_idx_test.sentence_to_idx(wrd_tag_vec_test,max_len_input)"""
######################################
kfold = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
kfold.get_n_splits(sent_idx_train.X_train)
a=kfold
cvscores = []
namepic=1
#Config Learning Decay

configlrdecay=ConfigLearningDecay()
configlrdecay.configdecay(epoch,schedule)

#Neural Network
LSTM_implement=CreatingModel()
LSTM_implement.creating_model(wrd_tag_vec_train,embedding_vector,unit1,unit2)
compile_model=Compilemodel()
compile_model.compilemodel(LSTM_implement,configlrdecay.decay)
train_model=Trainmodel()
train_model.trainmodel(sent_idx_train.X_train, sent_idx_train.Y_train, sent_idx_train.X_validation, sent_idx_train.Y_validation, LSTM_implement.model_, epoch, bachsize , configlrdecay.callbacks,configlrdecay.schedule)


"""evaluation=Evaluate_model()
evaluation.evaluatemodel(sent_idx_test.X_test,sent_idx_test.Y_test, LSTM_implement.model_,wrd_tag_vec_train)"""




