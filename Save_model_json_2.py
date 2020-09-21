import os
import datetime
from keras.models import model_from_json
from keras.models import model_from_json
class SaveModelJson():
    def save_model_json(self,model_,name):
        if not os.path.exists("models_json"):
            os.mkdir("models_json")
        # serialize model to JSON
        self.model_json = model_.to_json()
        with open("models_json/"+name+"model.json", "w") as json_file:
            json_file.write(self.model_json)
        # serialize weights to HDF5
        model_.save_weights("models_json/"+name+"model.json")
        print("Saved model to models_json")

