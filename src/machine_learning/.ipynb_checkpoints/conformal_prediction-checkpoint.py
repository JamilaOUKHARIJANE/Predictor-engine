import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from src.dataset_manager.datasetManager import DatasetManager

class conformal_prediction:

    def __init__(self, prediction_cal, prediction_test, y_cal, y_test, alpha):

        self.prediction_cal = prediction_cal
        self.prediction_test = prediction_test
        self.y_cal = y_cal
        self.y_test = y_test
        self.alpha = alpha
        pass
        # self.alpha=alpha

    # compute q-hat based on the Inverse probability nonconformity score
    def get_qhat(self, alpha):
        self.alpha = alpha

        """Inverse probability nonconformity score returns :math:`1 - p`, where :math:`p` is the probability
    assigned to the actual class by the underlying classification model (:py:attr:`ClassModelNC.model`)."""
        #         self.predictions_cal=predictions_cal
        #         self.y_cal=y_cal
        #         self.alpha=alpha
        
        N = self.prediction_cal.shape[0]
        scores = np.zeros(N)  # non-conformity scores
        
        for i, row in self.y_cal.iterrows():
            if (row['label']== True):
                true_class_proba = self.prediction_cal[i][1]  # predicted proba of the true class, y_cal[i]: to get the true class
            else:
                true_class_proba = self.prediction_cal[i][0]
            scores[i] = 1 - true_class_proba  # probs of the opposite class, s1, s2, .., sn

        q_yhat = np.quantile(scores, np.ceil((N + 1) * (1 - self.alpha)) / N)  # quantile of the non-conformity scores
        return q_yhat

    # function to get predict sets: Naive method
    def get_pred_set(self, q_yhat):
        self.q_yhat = q_yhat
        #         self.predictions_test=predictions_test
        #         self.y_test=y_test
        # self.model=model

        softmax_outputs = self.prediction_test  # probs for test set
        N = softmax_outputs.shape[0]  # how many rows in the test set
        pred_sets = []  # pred set

        for i in range(N):  # for each row
            aux = []
            for j in range(softmax_outputs.shape[1]):  # loop over the number of classes
                if softmax_outputs[i][j] >= 1 - self.q_yhat:  # pred_proba[class_0 | 1] >
                    aux.append(j)
            pred_sets.append(aux)

        return pred_sets
       
    def get_pred_set_size(self, alpha, pred_sets):
        self.alpha = alpha
        self.pred_sets = pred_sets

        my_dict = {}
        my_dict_alpha = {}
        import pandas as pd

        for a in self.alpha:
            my_dict = {}
            print(f"Alpha: {a}")
            for i in range(len(pd.Series(self.pred_sets[a]).value_counts().index[:].to_list())):
                # print(i)
                my_dict[str(pd.Series(self.pred_sets[a]).value_counts().index[:].to_list()[i])] = \
                pd.Series(self.pred_sets[a]).value_counts().to_list()[i]
                # break
            print(my_dict)
            my_dict_alpha[a] = my_dict
        return my_dict_alpha

    # function to evaluate coverage
    def calculate_coverage(self, pred_sets, y_true):
        s = 0

        for i in range(len(pred_sets)):
            if y_true[i] in pred_sets[i]:
                s += 1

        return s / len(y_true)

    def get_df_with_pred_sets(self,df, alpha, pred_set,y_test,predict_test):
        df = pd.DataFrame(df).copy()
        for a in alpha:
            #print(a)
            df["alpha=" + str(a)] = pred_set[a]
        df["actual"]= y_test['label']
        df["predicted"]= predict_test
        return df



        

