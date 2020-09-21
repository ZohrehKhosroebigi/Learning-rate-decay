import matplotlib.pyplot as plt
import numpy as np

# import the necessary packages
from keras.callbacks import LearningRateScheduler

from learning_decay import StepDecay
from learning_decay import PolynomialDecay

class ConfigLearningDecay():
    def configdecay(self,epoch,schedule):


        # store the number of epochs to train for in a convenience variable,
        # then initialize the list of callbacks and learning rate scheduler
        # to be used
        epochs = epoch
        self.callbacks = []
        #schedule = None
        # check to see if step-based learning rate decay should be used
        if schedule == "step":
            print("[INFO] using 'step-based' learning rate decay...")
            self.schedule = StepDecay(initAlpha=1e-1, factor=0.25, dropEvery=15)
        # check to see if linear learning rate decay should should be used
        elif schedule == "linear":
            print("[INFO] using 'linear' learning rate decay...")
            self.schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=1e-1, power=1)
        # check to see if a polynomial learning rate decay should be used
        elif schedule == "poly":
            print("[INFO] using 'polynomial' learning rate decay...")
            self.schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=1e-1, power=5)

        else:
            self.schedule = None
        # if the learning rate schedule is not empty, add it to the list of
        # callbacks
        if self.schedule is not None:
            self.callbacks = [LearningRateScheduler(self.schedule)]

        ##############################################################
        # initialize the decay for the optimizer
        self.decay = 0.0
        # if we are using Keras' "standard" decay, then we need to set the
        # decay parameter
        if schedule == "standard":
            print("[INFO] using 'keras standard' learning rate decay...")
            self.decay = 1e-1 / epochs
        # otherwise, no learning rate schedule is being used
        elif schedule is None:
            print("[INFO] no learning rate schedule being used")

        ##########################################################################
        print(f'callbacks is    {self.callbacks}')
        return self.decay,self.callbacks,self.schedule