import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import data_module
def read_dataset(filepath):
    enc_tag = preprocessing.LabelEncoder()

    df = pd.read_csv(filepath)
    df.loc[:, "Conference"] = enc_tag.fit_transform(df["Conference"])
    X_train, X_val, y_train, y_val = train_test_split(df.Title.values,
                                                      df.Conference.values,
                                                      test_size=0.15,
                                                      random_state=42,
                                                      stratify=df.Conference.values)

    return X_train, X_val, y_train, y_val,enc_tag


if __name__ == '__main__':

    X_train, X_val, y_train, y_val, enc_tag = read_dataset('data.csv')
    train_dataset = data_module.ExampleDataset(text=X_train,target=y_train).setup()
    val_dataset = data_module.ExampleDataset(text=X_val, target=y_val).setup()
    


