import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(df, encoder):
    df_x = df.drop(["exited"], axis=1)
    df_y = df["exited"]

    np_x_categorical = df_x[["corporation"]].values
    df_x_continuous = df_x.drop(["corporation"], axis=1)

    if not encoder:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        np_x_categorical = encoder.fit_transform(np_x_categorical)
    else:
        np_x_categorical = encoder.transform(np_x_categorical)
    df_x = np.concatenate([np_x_categorical, df_x_continuous], axis=1)

    return df_x, df_y, encoder