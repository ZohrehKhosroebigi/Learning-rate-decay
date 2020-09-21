from writing import *
import numpy as np
from sklearn_crfsuite.metrics import flat_classification_report
class Evaluate_model():

        def evaluatemodel(self, x_test, y_test,model_,myobj,cvscores):
           #Androw
            preds = model_.evaluate(x=x_test, y=y_test)
            mywriting = Writelogs()
            mywriting.writing("evaluate = " + str(preds[0]), "Test Accuracy = " + str(preds[1])+'\n')
            # Eval
            pred_cat = model_.predict(x_test)
            pred = np.argmax(pred_cat, axis=-1)
            y_te_true = np.argmax(y_test, -1)
            mywriting.writing("predict = " + str(preds[0]), "Test Accuracy = " + str(preds[1])+'\n')
            # Convert the index to tag
            pred_tag = [[myobj.idx2tag[i] for i in row] for row in pred]
            y_te_true_tag = [[myobj.idx2tag[i] for i in row] for row in y_te_true]
            report = flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag)
            print(report)
            mywriting.writing(report)
            ###############
            scores = model_.evaluate(x_test, y_test, verbose=0)
            print("%s: %.2f%%" % (model_.metrics_names[1], scores[1] * 100))
            cvscores.append(scores[1] * 100)
            print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
            mywriting.writing(("model_.metrics_names scores"),str((model_.metrics_names[1], scores[1] * 100)))
            mywriting.writing(("mean(cvscores)  np.std(cvscores)"),str((np.mean(cvscores), np.std(cvscores))))
            return model_