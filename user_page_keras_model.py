# load and evaluate a saved model
from keras.models import load_model
import numpy as np
from Loadrawdata_show_22_test import Loadrawdata_show_test
from Word_tags_to_idx_33_test import WordTagsToIdx_test
from Preprocessing_Sentence_to_idx_44_test import SentenceToIdx_test
import ast
keras_model= load_model('models_keras/model.h5')
keras_model.summary()
# load dataset
filename='Raw_Data/Test_set.txt'
max_len_input=300
# if u want to read tags from the file note: there is a mistake in file. it needs a singe cote
"""fidx_to_tag=open("Raw_Data/idx_to_word.txt","r",encoding="utf8")
idx_to_tag=fidx_to_tag.read()
idx_to_tag=ast.literal_eval(idx_to_tag)
"""
idx_to_tag={1: 'O', 2: 'B-ORG', 3: 'CB-LOC', 4: 'B-PER', 5: 'I-LOC', 6: 'I-ORG', 7: 'I-PER', 8: 'B-loc', 9: 'I-loc', 10: 'B-pers', 11: 'B-org', 12: 'I-pers', 13: 'B-event', 14: 'I-event', 15: 'B-fac', 16: 'I-fac', 17: 'I-org', 18: 'B-pro', 19: 'I-pro', 20: 'B-DAT', 21: 'I-DAT', 22: 'B-EVE', 23: 'I-EVE', 0: 'PAD'}

# prepating data to model
loadrawdata=Loadrawdata_show_test_()
wrd_tag_vec=WordTagsToIdx_test()
wrd_tag_vec.word_tags_to_idx(loadrawdata.load(filename),max_len_input)
sent_idx=SentenceToIdx_test()
sent_idx.sentence_to_idx(wrd_tag_vec,max_len_input)
predicts = keras_model.predict(sent_idx.X_test, verbose=1)
#print("preds....."+str((preds)))
print("{:15} {:5}    ".format("Word", "Pred"))
for  words,predictions in zip(sent_idx.X_test,predicts):
        #print("prediction....."+str((preds)))
        for word, prediction in zip(words,predictions):
                print("_" * 20)
                predict_val=np.max(prediction)
                predict_position=np.argmax(prediction)
                predict_tag=idx_to_tag[predict_position]
                word=wrd_tag_vec.idx2word[word]
                print("{:15}:{:5}     ".format(word, predict_tag))