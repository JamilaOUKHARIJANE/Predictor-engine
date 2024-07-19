from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
from time import time
from pm4py.objects.conversion.log import converter as log_converter
import os
import settings

class latest_IndexBasedTransformer(TransformerMixin):
    
    def __init__(self, case_id_col, cat_cols, label_col,pos_label, num_cols,static_cat_cols,static_num_cols, dynamic_activity_col, support_threshold_dict, max_events=None, fillna=True, create_dummies=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.static_cat_cols = static_cat_cols
        self.dynamic_activity_col = dynamic_activity_col
        self.static_num_cols = static_num_cols
        self.max_events = max_events
        self.label_col = label_col 
        self.pos_label = pos_label
        self.fillna = fillna
        self.create_dummies = create_dummies
        self.support_threshold_dict=support_threshold_dict
        
        self.columns = None
        
        self.fit_time = 0
        self.transform_time = 0
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        start = time()
        '''dt_transformed1 = pd.get_dummies(X[self.dynamic_activity_col])
        dt_transformed1[self.case_id_col] = X[self.case_id_col]
        dt_transformed1 = dt_transformed1.groupby(self.case_id_col).max()
        #print(dt_transformed1)'''
        
        grouped = X.groupby(self.case_id_col, as_index=False)
            
        dt_transformed = pd.DataFrame(grouped.apply(lambda x: x.name), columns=[self.case_id_col])
        
        #static columns
        dt_index2=grouped.tail(1)[[self.case_id_col] + self.static_cat_cols+ self.static_num_cols + [col for col in self.cat_cols if col not in self.dynamic_activity_col] + self.num_cols]
        dt_index2.columns = [self.case_id_col] + ["%s,"%col for col in self.static_cat_cols]+ ["%s"%col for col in self.static_num_cols]+ ["%s,"%col for col in self.cat_cols if col not in self.dynamic_activity_col]+ ["%s"%col for col in self.num_cols]
        dt_transformed = pd.merge(dt_transformed, dt_index2, on=self.case_id_col, how="left")
        #dt_transformed = pd.merge(dt_transformed, dt_transformed1, on=self.case_id_col, how="left")
        #print(dt_transformed)
        
        if y is None:
            z= grouped.tail(1)[[self.case_id_col] +[self.label_col]]
            z["prefix"]=grouped.size()['size']
            #dt_transformed = pd.merge(dt_transformed, z, on=self.case_id_col, how="left")
            #print(dt_transformed)
            y = z[self.label_col].apply(lambda x: True if x == self.pos_label else False)
            #print(y)
            
    
        # one-hot-encode cat cols
        if self.create_dummies:
            all_cat_cols = ["%s,"%col for col in self.cat_cols if col not in self.dynamic_activity_col]+ ["%s,"%col for col in self.static_cat_cols]
            dt_transformed = pd.get_dummies(dt_transformed, columns=all_cat_cols).drop(self.case_id_col, axis=1)

        
        # fill missing values with 0-s
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)

        # add missing columns if necessary
        if self.columns is None:
            self.columns = dt_transformed.columns
        else:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]

        self.transform_time = time() - start

        file_XT = os.path.join(settings.results_dir, "Encoded_data.csv")
        dt_transformed.to_csv(file_XT, index=False, sep=";") 
        return dt_transformed, y
        