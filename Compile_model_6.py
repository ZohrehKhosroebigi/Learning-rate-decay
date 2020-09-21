from keras_contrib.layers import crf
from keras.optimizers import SGD


class Compilemodel():
    def compilemodel(self,myobj, decay):
        opt=SGD(lr=0.1,decay=decay)
        myobj.model_.compile(optimizer=opt, loss=myobj.crf.loss_function, metrics=[myobj.crf.accuracy])

        myobj.model_.summary()
        freport = open("logs/report.txt", "a", encoding="utf8")
        freport.write("model_.summary---------" + str(myobj.model_.summary()) + "\n")

        return myobj.model_
