from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
import keras as k
from keras_contrib.layers import CRF

class CreatingModel():
    def creating_model(self,myobj,EMBEDDING,unit1,unit2):
         # Dimension of word embedding vector
        # Model definition
         #max_len_vector is max len sentence
        input = Input(shape=(myobj.max_len_vector,))

        X = Embedding(input_dim=myobj.n_words+2, output_dim=EMBEDDING, # n_words + 2 (PAD & UNK)
                          input_length=myobj.max_len_vector, mask_zero=True)(input)  # default: 20-dim embedding
        freport = open("logs/report.txt", "a", encoding="utf8")
        freport.write("embed.shape------" + str(X.shape) + "\n")
        print ("shape of embedding:   "+str(X.shape))
        X = Bidirectional(LSTM(units=unit1, return_sequences=True,dropout=0.3,
                                   recurrent_dropout=0.3))(X)
        freport.write("layer1.shape-----"+str((X.shape)) + "\n")

        print("shape of layer1:   " + str(X.shape))
        """X = Bidirectional(LSTM(units=unit2,return_sequences=True,dropout=0.3,
                               recurrent_dropout=0.3))(X)
        print("shape of layer2:   " + str(X.shape))
        freport.write("layer2.shape-----" + str((X.shape)) + "\n")"""
        # variational biLSTM
        #x = add([layer1,layer2])

        #X=out
        X= TimeDistributed(Dense(myobj.n_tags+1, activation="softmax"))(X)  # a dense layer as suggested by neuralNer
        self.crf=CRF(myobj.n_tags+1)

        out = self.crf(X)  # output
        self.model_ = Model(input, out)
        return self.model_, self.crf

        """self.model_ = Model(input, X)
        return self.model_
"""
