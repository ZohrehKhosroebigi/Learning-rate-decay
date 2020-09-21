# evaluate loaded model on test data
from Json_To_Model import JsonToModel
from Evaluate_model_8 import Evaluate_model
from Compile_model_6 import Compilemodel
from Loadrawdata_show_2 import Loadrawdata_show_train
from Word_tags_to_idx_3 import WordTagsToIdx_train
model_=JsonToModel()
model_.json_to_model
model_.my_model_.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model_.compile(optimizer="rmsprop", loss=myobj.crf.loss_function, metrics=[myobj.crf.accuracy])

#Test loaded data
loadrawdata=Loadrawdata_show()
loadrawdata.load

norm_data=NoramlPic()
norm_data.norm(loadrawdata.load)

evaluate_model=Evaluate_model()
evaluate_model.evaluatemodel(norm_data.X_test,norm_data.Y_test,model_.my_model_)

