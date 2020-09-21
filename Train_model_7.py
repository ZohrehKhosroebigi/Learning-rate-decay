import matplotlib.pyplot as plt
import matplotlib
from Save_model_json_2 import SaveModelJson
from Save_model_keras import SaveModelKeras
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
from keras.models import Sequential

from Data_generator import DataGenerator
class Trainmodel():

    def trainmodel(self, x_train, y_train, x_validation, y_validation, model_, epoch, batchsize, callbacks, schedule):
        if not os.path.exists("models_keras"):
            os.mkdir("models_keras")
        #model_.fit(x=x_train, y=y_train, epochs=epoch, batchsize=batchsize)
        print(x_train)
        print(len(y_train))
        filepath="models_keras/weights.best.hdf5"
        """checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True,mode='max')
        callbacks_list = [checkpoint]"""
        history =model_.fit(x_train, np.array(y_train), validation_data=(x_validation, np.array(y_validation)),batch_size=batchsize, epochs=epoch, callbacks=callbacks, shuffle=True, verbose=1)

        #history =model_.fit(x_train, np.array(y_train),validation_data=(x_validation, np.array(y_validation)), batchsize=batchsize, epochs=epoch, verbose=2, callbacks=callbacks_list, shuffle=True)
        fresult=open("logs/result.txt", "w", encoding="utf8")
        fresult.write("history.history.keys:   "+str(history.history.keys()) + "\n")
        fresult.write("history.history.values:   " + str(history.history.values()) + "\n")
        print(history.history.keys())
        print(history.history.values())
############################################
#dict_keys(['val_loss', 'val_crf_viterbi_accuracy', 'loss', 'crf_viterbi_accuracy'])
        # Plot the graph
        plt.style.use('ggplot')

        def plot_history(history):
            # plot the training loss and accuracy
            N = np.arange(0, epoch)
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, history.history["loss"], label="train_loss")
            plt.plot(N, history.history["val_loss"], label="val_loss")
            plt.plot(N, history.history["crf_viterbi_accuracy"], label="train_acc")
            plt.plot(N, history.history["val_crf_viterbi_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy on CIFAR-10")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig("training_loss_accu.png")
            # if the learning rate schedule is not empty, then save the learning
            # rate plot
            if schedule is not None:
                schedule.plot(N)
                plt.savefig("lr_plot.png")



        plot_history(history)

        #save keras model
        saveTokeras_model = SaveModelKeras()
        saveTokeras_model.save_model_keras(model_,"mykeras")
        #save json model
        saveTojason_model=SaveModelJson()
        saveTojason_model.save_model_json(model_,"myjson")
        return model_