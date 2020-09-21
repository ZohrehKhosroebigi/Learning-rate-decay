import tensorflow
from numpy import savetxt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
class SentenceToIdx_train():
    def sentence_to_idx(self, myobj, max_vector_sentence):

        fx=open("Raw_Data/x.txt","w",encoding="utf8")
        fy = open("Raw_Data/y.txt", "w", encoding="utf8")
        self.n_tags=myobj.n_tags
        self.max_len_vector=max_vector_sentence
        # Convert each sentence from list of Token to list of word_index
        my_sent=myobj.sentence

        n_lrg_sent=0
        try:
            X = [[myobj.word2idx[w[0]] for w in s] for s in myobj.sentence]
        except Exception as err:
            print("erorr is ----"+str(err))
        fx.write(str(X))

        # Padding each sentence to have the same lenght
        X = pad_sequences(maxlen=self.max_len_vector, sequences=X, padding="post", value=myobj.word2idx["PAD"])

        # Convert Tag/Label to tag_index
        Y = [[myobj.tag2idx[w[1]] for w in s] for s in myobj.sentence]
        fy.write(str(Y))

        # Padding each sentence to have the same lenght
        Y = pad_sequences(maxlen=self.max_len_vector, sequences=Y, padding="post", value=myobj.tag2idx["PAD"])

        # One-Hot encode
        Y = [to_categorical(i, num_classes=myobj.n_tags + 1) for i in Y]  # n_tags+1(PAD)
        seed = 7
        np.random.seed(seed)
        self.X_train, self.X_validation, self.Y_train, self.Y_validation=X,X,Y,Y


        #self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(X, Y, test_size=0.1,random_state=seed)
        self.Y_train = np.array(self.Y_train)
        self.Y_validation = np.array(self.Y_validation)
        print("self.X_train shape---"+str(self.X_train.shape))
        print(" larger than 300--"+str(n_lrg_sent))
        freport = open("logs/report.txt", "a", encoding="utf8")
        freport.write("self.X_train.shape---------"+str(self.X_train.shape) + "\n")
        freport.write("self.Y_train.shape---------"+str((self.Y_train.shape)) + "\n")
        # self.X_train.shape---------(6307, 300)
        # self.Y_train.shape---------(6307, 300, 14)
        #falt_X_train=self.X_train.reshape(1,600)
        #falt_Y_train = self.Y_train.reshape(1,1800)
        print(self.Y_train)
        #savetxt('Raw_Data/X_train.CSV', falt_X_train, delimiter=',')
        #savetxt('Raw_Data/Y_train.CSV', falt_Y_train, delimiter=',')

        return self.X_train,  self.Y_train,self.X_validation,self.Y_validation,self.n_tags, self.max_len_vector

    def __str__(self):
        return f'{self.X_train.shape}{self.Y_train.shape}{self.X_validation}{self.Y_validation}{self.n_tags}{self.max_len_vector}'

    def __repr__(self):
        return f'{self.X_train}{self.Y_train}{self.X_validation}{self.Y_validation}{self.n_tags}{self.max_len_vector}'



