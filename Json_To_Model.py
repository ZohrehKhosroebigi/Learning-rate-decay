from tensorflow.keras.models import model_from_json
import os
import datetime
class JsonToModel():
    @property
    def json_to_model(self):
        if not os.path.exists("models_json"):
                print("No model folder")
        # load json and create model
        json_file = open('models_json/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.my_model_ = model_from_json(loaded_model_json)
        # load weights into new model
        self.my_model_.load_weights("models_json/model.h5")
        print("Loaded model from disk")
        return self.my_model_
