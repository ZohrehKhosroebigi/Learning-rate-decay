import os
import datetime
from tensorflow.keras.models import model_from_json
class SaveModelKeras():
    def save_model_keras(self,model_,name):
        if not os.path.exists("models_keras"):
            os.mkdir("models_keras")
        model_.save("models_keras/"+name+"model.h5")
        print("Saved model to models_keras")