import pandas as pd
from sklearn.base import TransformerMixin


class latest_IndexBasedTransformer(TransformerMixin):

    def __init__(self, case_id_col, dynamic_cat_cols, label_col, pos_label, dynamic_num_cols, static_cat_cols,
                 static_num_cols, dynamic_activity_col, support_threshold_dict, cat_maps_, max_events=None,
                 fillna=True, create_dummies=True):
        self.case_id_col = case_id_col
        self.cat_cols = dynamic_cat_cols
        self.num_cols = dynamic_num_cols
        self.static_cat_cols = static_cat_cols
        self.activity_col = dynamic_activity_col
        self.static_num_cols = static_num_cols
        self.max_events = max_events
        self.label_col = label_col
        self.pos_label = pos_label
        self.fillna = fillna
        self.create_dummies = create_dummies
        self.support_threshold_dict = support_threshold_dict
        self.cat_maps_ = cat_maps_

        self.columns = None

        self.fit_time = 0
        self.transform_time = 0

    def fit(self, X, y=None):
        return self

    def _pad_list(self, seq, max_len, pad_val=0):
        seq = list(seq)
        if len(seq) >= max_len:
            return seq[:max_len]
        return seq + [pad_val] * (max_len - len(seq))

    def convert_true_false_object_cols_to_int(self, df):
        df = df.copy()

        # candidate object columns
        obj_cols = df.select_dtypes(include=["object"]).columns

        for c in obj_cols:
            s = df[c]
            nn = s.dropna()
            if nn.empty:
                continue

            # normalize to strings and check if it's only true/false/0/1
            norm = set(nn.astype(str).str.strip().str.lower().unique())
            allowed = {"true", "false", "0", "1"}

            if norm.issubset(allowed):
                df[c] = (
                    s.replace({True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0})
                    .pipe(pd.to_numeric, errors="coerce")
                    .fillna(0)
                    .astype(int)
                )

        return df

    def transform(self, X, y=None):
        grouped = X.groupby(self.case_id_col, sort=False)

        # max_len
        if self.max_events is None:
            self.max_events = int(grouped.size().max())
        max_len = int(self.max_events)

        X["_act_idx"] = (
            X[self.activity_col].astype(str)
            .map(self.activity2idx)
            .fillna(0)
            .astype(int)
        )

        act_seq = grouped["_act_idx"].apply(lambda s: self._pad_list(s.tolist(), max_len, 0))
        act_df = pd.DataFrame(act_seq.tolist(),
                              index=act_seq.index,
                              columns=[f"act_{i}" for i in range(1, max_len + 1)]
                              ).reset_index()
        act_df = act_df.rename(columns={"index": self.case_id_col})

        attr_cols = (
                self.static_cat_cols + self.static_num_cols +
                self.cat_cols + self.num_cols
        )

        if attr_cols:
            X[attr_cols] = X.groupby(self.case_id_col, sort=False)[attr_cols].ffill()

        # take last row per case (contains last-known values)
        last = X.groupby(self.case_id_col, sort=False).tail(1)[[self.case_id_col] + attr_cols].copy()
        # index encode cat cols
        all_cat_cols = self.static_cat_cols + self.cat_cols
        for col in all_cat_cols:
            last[col] = (
                last[col].astype(str).fillna("__MISSING__")
                .map(self.cat_maps_[col])
                .fillna(0)
                .astype(int)
            )

        last.columns = (
                [self.case_id_col] +
                [f"{c}," for c in self.static_cat_cols] +
                [f"{c}" for c in self.static_num_cols] +
                [f"{c}," for c in self.cat_cols] +
                [f"{c}" for c in self.num_cols]
        )

        dt_transformed = act_df.merge(last, on=self.case_id_col, how="left")

        if y is None:
            y = (
                X.groupby(self.case_id_col, sort=False).tail(1)[self.label_col]
                .apply(lambda x: True if x == self.pos_label else False)
                .reset_index(drop=True)
            )

        dt_transformed = dt_transformed.drop(columns=[self.case_id_col], axis=1)
        dt_transformed = dt_transformed.fillna(0)
        dt_transformed = self.convert_true_false_object_cols_to_int(dt_transformed)

        return dt_transformed, y

