import pandas as pd
import matplotlib.pyplot as plt
import os
import settings
from src.dataset_manager.datasetManager import DatasetManager

'''dataset_names = ["bpic2011_f1", "bpic2011_f2", "bpic2011_f3", "bpic2011_f4",
                  "bpic2012_accepted", "bpic2012_cancelled", "bpic2012_declined",
                  "bpic2015_1_f2", "bpic2015_2_f2", "bpic2015_3_f2", "bpic2015_4_f2",
                  "bpic2015_5_f2", "hospital_billing_2", "hospital_billing_3", "Production",
                  "sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4"]'''
dataset_names = ["bpic2011_f3", "bpic2015_4_f2","Production","sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4"]

for dataset_name in dataset_names:
    dataset_manager = DatasetManager(dataset_name.lower())
    data = dataset_manager.read_dataset(os.path.join(os.getcwd(), settings.dataset_folder))

    # split into training and test
    train_val_ratio = 0.8
    if dataset_name == "bpic2015_4_f2":
        train_val_ratio = 0.85
    train_ratio = 0.8
    train_val_df, test_df = dataset_manager.split_data_strict(data, train_val_ratio)
    train_df, val_df = dataset_manager.split_data(train_val_df, train_ratio, split="random")

    # determine min and max (truncated) prefix lengths
    min_prefix_length = 1
    if "traffic_fines" in dataset_name:
        max_prefix_length_test, max_prefix_length_val, max_prefix_train_val = 9, 9, 9
    elif "bpic2017" in dataset_name:
        max_prefix_length_test = min(20, dataset_manager.get_pos_case_length_quantile(test_df, 0.90))
        max_prefix_length_val = min(20, dataset_manager.get_pos_case_length_quantile(val_df, 0.90))
        max_prefix_train_val =min(20, dataset_manager.get_pos_case_length_quantile(train_df, 0.90))
    else:
        max_prefix_length_test = min(40, dataset_manager.get_pos_case_length_quantile(test_df, 0.90))
        max_prefix_length_val = min(40, dataset_manager.get_pos_case_length_quantile(val_df, 0.90))
        max_prefix_train_val =min(40, dataset_manager.get_pos_case_length_quantile(train_df, 0.90))

    print(dataset_name, ":", max_prefix_train_val)
    print("-------------------------------------------------------")