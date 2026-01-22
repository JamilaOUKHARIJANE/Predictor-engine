import pm4py
from pm4py.utils import get_properties

from src.dataset_manager import dataset_confs
from pm4py.statistics.variants.log import get as variants_get
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from pm4py.objects.conversion.log import converter as log_converter



class DatasetManager:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        self.case_id_col = dataset_confs.case_id_col[self.dataset_name]
        self.activity_col = dataset_confs.activity_col[self.dataset_name]
        self.timestamp_col = dataset_confs.timestamp_col[self.dataset_name]
        self.label_col = dataset_confs.label_col[self.dataset_name]
        self.pos_label = dataset_confs.pos_label[self.dataset_name]
        self.neg_label = dataset_confs.neg_label[self.dataset_name]

        self.dynamic_activity_col = dataset_confs.dynamic_activity_col[self.dataset_name]
        self.dynamic_cat_cols = dataset_confs.dynamic_cat_cols[self.dataset_name]
        self.static_cat_cols = dataset_confs.static_cat_cols[self.dataset_name]
        self.dynamic_num_cols = dataset_confs.dynamic_num_cols[self.dataset_name]
        self.static_num_cols = dataset_confs.static_num_cols[self.dataset_name]

        self.sorting_cols = [self.timestamp_col, self.activity_col]

    def read_dataset(self, dataset_path):
        # read dataset
        dtypes = {col: "object" for col in
                  self.dynamic_cat_cols + self.static_cat_cols + [self.case_id_col, self.label_col, self.timestamp_col]}
        for col in self.dynamic_num_cols + self.static_num_cols:
            dtypes[col] = "float"
        if str(dataset_confs.filename[self.dataset_name]).endswith('.csv'):
            data = pd.read_csv(os.path.join(dataset_path, dataset_confs.filename[self.dataset_name]), sep=";",
                               dtype=dtypes)
        else:
            data = pm4py.read_xes(os.path.join(dataset_path, dataset_confs.filename[self.dataset_name]))
        data[self.timestamp_col] = pd.to_datetime(data[self.timestamp_col])
        return data

    def split_data(self, data, train_ratio, split="temporal", seed=22):
        # split into train and test using temporal split
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        if split == "temporal":
            start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
        elif split == "random":
            np.random.seed(seed)
            start_timestamps = start_timestamps.reindex(np.random.permutation(start_timestamps.index))
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio * len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True,
                                                                         kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True,
                                                                         kind='mergesort')
        return train, test

    def _to_event_log(self, df):
        params = get_properties(df,
                                case_id_key=self.case_id_col,
                                activity_key=self.activity_col,
                                timestamp_key=self.timestamp_col)
        return log_converter.apply(df,
                                   variant=log_converter.Variants.TO_EVENT_LOG,
                                   parameters=params), params

    def get_dominant_variant(self, train_log):
        elog, params = self._to_event_log(train_log)
        variants = variants_get.get_variants(elog, parameters=params)  # {variant_tuple: [trace, ...]}

        if not variants:
            raise ValueError("No variants found in train_log")

        # pick most frequent full-trace variant v*
        top3_variants = [v for v, traces in sorted(variants.items(),
                                                   key=lambda kv: len(kv[1]),
                                                   reverse=True)[:3]]
        return top3_variants

    def label_data_by_dominant_variant(self, v_star1, v_star2, v_star3, labeled_log, label_key="label"):
        elog, _ = self._to_event_log(labeled_log)

        for trace in elog:
            seq = tuple(ev[self.activity_col] for ev in trace)
            k = len(seq)

            if k <= max(len(v_star1), len(v_star2), len(v_star3)) and seq in [v_star1[:k], v_star2[:k], v_star3[:k]]:
                trace.attributes[label_key] = "noAdapt"
            else:
                trace.attributes[label_key] = "Adapt"

        df = log_converter.apply(elog, variant=log_converter.Variants.TO_DATA_FRAME)
        case_to_label = {t.attributes["concept:name"]: t.attributes.get(label_key) for t in elog}
        df[label_key] = df[self.case_id_col].map(case_to_label)
        return df

    def generate_prefix_data_new(self, data, min_length, max_length, gap=1,
                                 v_star=None, label_key="label",
                                 activity2idx=None, pad_idx=0, unknown_idx=None):
        """
        Returns prefix-level table: 1 row per prefix.
        - prefix_act_idx: tuple length=max_length with activity indices (pad with 0)
        - calls self.label_data_by_dominant_variant(v_star, labeled_log)
        - booleans -> 0/1, numeric missing before first obs -> 0 (e.g., CRP)
        """

        df = data.copy()

        if hasattr(self, "timestamp_col") and self.timestamp_col in df.columns:
            df = df.sort_values([self.case_id_col, self.timestamp_col])
        else:
            df = df.sort_values([self.case_id_col])

        df["event_nr"] = df.groupby(self.case_id_col).cumcount() + 1
        df["case_length"] = df.groupby(self.case_id_col)[self.activity_col].transform("size")
        df["case_length_capped"] = df["case_length"].clip(upper=max_length)

        # ---- boolean -> 0/1 ----
        bool_cols = [c for c in df.columns if df[c].dtype == bool]
        for c in bool_cols:
            df[c] = df[c].astype(int)

        # also handle "True"/"False" stored as strings
        for c in df.columns:
            if df[c].dtype == object:
                non_null = df[c].dropna()
                if not non_null.empty:
                    low = non_null.astype(str).str.lower()
                    if low.isin(["true", "false"]).all():
                        df[c] = df[c].astype(str).str.lower().map({"true": 1, "false": 0}).astype("float")

        # ---- forward fill ALL non-core attributes within each case, then fill remaining NaN with 0 for numeric ----
        core = {self.case_id_col, self.activity_col, "event_nr", "case_length", "case_length_capped"}
        if hasattr(self, "timestamp_col") and self.timestamp_col in df.columns:
            core.add(self.timestamp_col)

        attr_cols = [c for c in df.columns if c not in core]

        # forward fill attributes so "last-known in prefix" is available at event k
        if attr_cols:
            df[attr_cols] = df.groupby(self.case_id_col)[attr_cols].ffill()

            # fill numeric NaN with 0 (CRP before first CRP -> 0)
            for c in attr_cols:
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(0)

        # ---- activity2idx (train-only recommended) ----
        if activity2idx is None:
            acts = sorted(pd.unique(df[self.activity_col].dropna().astype(str)))
            activity2idx = {a: i + 1 for i, a in enumerate(acts)}  # reserve 0 for pad
        if unknown_idx is None:
            unknown_idx = max(activity2idx.values()) + 1 if activity2idx else 1

        df["_act_idx"] = df[self.activity_col].astype(str).map(activity2idx).fillna(unknown_idx).astype(int)

        # Generate prefixes
        prefix_event_rows = []
        for k in range(min_length, max_length + 1, gap):
            tmp = df[df["case_length"] >= k].groupby(self.case_id_col).head(k).copy()

            tmp["orig_case_id"] = tmp[self.case_id_col]
            tmp[self.case_id_col] = tmp["orig_case_id"].astype(str) + f"_{k}"
            tmp["prefix_nr"] = k
            prefix_event_rows.append(tmp)

        dt_event_prefixes = pd.concat(prefix_event_rows, axis=0, ignore_index=True)

        # labeling prefixes
        if v_star is not None:
            dt_event_prefixes = self.label_data_by_dominant_variant(
                v_star=v_star,
                labeled_log=dt_event_prefixes,
                label_key=label_key
            )
        else:
            dt_event_prefixes[label_key] = None

        # Encoding prefixes

        # fixed-length tuple of activity indices per prefix trace
        def pad_to_max(seq):
            seq = list(seq)
            if len(seq) < max_length:
                seq = seq + [pad_idx] * (max_length - len(seq))
            else:
                seq = seq[:max_length]
            return tuple(seq)

        # activity index sequence per prefix-case
        seq_per_prefix = (
            dt_event_prefixes
            .sort_values([self.case_id_col, "event_nr"])
            .groupby(self.case_id_col)["_act_idx"]
            .apply(pad_to_max)
            .rename("prefix_act_idx")
            .reset_index()
        )

        # take LAST EVENT ROW of each prefix (it contains last-known carried attrs at step k)
        last_rows = (
            dt_event_prefixes
            .sort_values([self.case_id_col, "event_nr"])
            .groupby(self.case_id_col, as_index=False)
            .tail(1)
        )

        # keep minimal id cols + attributes (drop event-specific columns you don’t want)
        keep_cols = [self.case_id_col, label_key]
        # keep all attributes except raw activity/timestamp/internal helpers (adjust as you like)
        drop_cols = {self.activity_col, "_act_idx", "event_nr", "orig_case_id", "prefix_nr", "case_length_capped", }
        if hasattr(self, "timestamp_col") and self.timestamp_col in last_rows.columns:
            drop_cols.add(self.timestamp_col)

        extra_cols = [c for c in last_rows.columns if c not in set(keep_cols) | drop_cols]
        last_rows = last_rows[keep_cols + extra_cols]

        # merge activity tuple
        dt_prefix_level = last_rows.merge(seq_per_prefix, on=self.case_id_col, how="left")

        # rename for clarity
        dt_prefix_level = dt_prefix_level.rename(columns={"case_length_capped": "case_length"})

        return dt_prefix_level, activity2idx

    def generate_prefix_data(self, data, min_length, max_length, gap=1):
        # generate prefix data (each possible prefix becomes a trace)
        data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)

        dt_prefixes = data[data['case_length'] >= min_length].groupby(self.case_id_col).head(min_length)
        dt_prefixes["prefix_nr"] = 1
        # dt_prefixes["orig_case_id"] = dt_prefixes[self.case_id_col]
        for nr_events in range(min_length + gap, max_length + 1, gap):
            tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events)
            # tmp["orig_case_id"] = tmp[self.case_id_col]
            tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s" % (x, nr_events))
            # tmp["prefix_nr"] = nr_events
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)

        dt_prefixes['case_length'] = dt_prefixes['case_length'].apply(lambda x: min(max_length, x))
        return dt_prefixes

    def get_pos_case_length_quantile(self, data, quantile=0.90, save_hist=False, ):
        if save_hist:
            hist = data.groupby(self.case_id_col).size().plot.hist(bins=20)
            hist_1 = data[data[self.label_col] == self.pos_label].groupby(self.case_id_col).size().plot.hist(bins=20)
            plt.savefig(f'lbl_hist_{self.dataset_name}.pdf')
        return int(
            np.ceil(data[data[self.label_col] == self.pos_label].groupby(self.case_id_col).size().quantile(quantile)))

    def get_indexes(self, data):
        return data.groupby(self.case_id_col).first().index

    def get_relevant_data_by_indexes(self, data, indexes):
        return data[data[self.case_id_col].isin(indexes)]

    def get_label(self, data):
        return data.groupby("case:concept:name").first()[self.label_col]

    def get_prefix_lengths(self, data):
        return data.groupby(self.case_id_col).last()["prefix_nr"]

    def get_case_ids(self, data, nr_events=1):
        case_ids = pd.Series(data.groupby(self.case_id_col).first().index)
        if nr_events > 1:
            case_ids = case_ids.apply(lambda x: "_".join(x.split("_")[:-1]))
        return case_ids

    def get_label_numeric(self, data):
        y = self.get_label(data)  # one row per case
        return [1 if label == self.pos_label else 0 for label in y]

    def get_class_ratio(self, data):
        class_freqs = data[self.label_col].value_counts()
        return class_freqs[self.pos_label] / class_freqs.sum()

    def get_stratified_split_generator(self, data, n_splits=5, shuffle=True, random_state=22):
        grouped_firsts = data.groupby(self.case_id_col, as_index=False).first()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        for train_index, test_index in skf.split(grouped_firsts, grouped_firsts[self.label_col]):
            current_train_names = grouped_firsts[self.case_id_col][train_index]
            train_chunk = data[data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col,
                                                                                             ascending=True,
                                                                                             kind='mergesort')
            test_chunk = data[~data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col,
                                                                                             ascending=True,
                                                                                             kind='mergesort')
            yield (train_chunk, test_chunk)

    def get_idx_split_generator(self, dt_for_splitting, n_splits=5, shuffle=True, random_state=22):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        for train_index, test_index in skf.split(dt_for_splitting, dt_for_splitting[self.label_col]):
            current_train_names = dt_for_splitting[self.case_id_col][train_index]
            current_test_names = dt_for_splitting[self.case_id_col][test_index]
            yield (current_train_names, current_test_names)
