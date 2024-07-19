from src.machine_learning.labeling import *
from src.models.DTInput import *
from src.enums.ConstraintChecker import *
import settings
from collections import defaultdict
from src.machine_learning.latest_IndexBasedTransformer import latest_IndexBasedTransformer
      
def get_encoder(method, support_threshold_dict, case_id_col=None, static_cat_cols=None, label_col=None,pos_label=None, static_num_cols=None, dynamic_cat_cols=None, dynamic_activity_col=None, dynamic_num_cols=None, fillna=True, max_events=None):
    return latest_IndexBasedTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, dynamic_activity_col=dynamic_activity_col, static_cat_cols= static_cat_cols,static_num_cols =static_num_cols, label_col=label_col,pos_label=pos_label, num_cols=dynamic_num_cols, max_events=max_events, fillna=fillna, support_threshold_dict=support_threshold_dict) 

def freq_encode_traces(log):
    features = []
    encoded_data = []
    events=get_events(log)
    for trace in log:
        trace_result = {}
        for a in events:
            key = a
            if existence(trace, a)>0:                
                trace_result[key] = 1
            else:
                trace_result[key] = 0
        if not features:
            features = list(trace_result.keys())
        encoded_data.append(list(trace_result.values()))
    return features, encoded_data

def existence(trace, a):
    num_activations = 0
    for A in trace:
        if A["concept:name"] == a: 
            num_activations += 1

    return num_activations

def get_events(log):
    res = defaultdict(lambda: 0, {})
    for trace in log:
        for event in trace:
            event_name = event["concept:name"]
            res[event_name] += 1
    events = []
    for key in res:
        events.append(key)
    return events

