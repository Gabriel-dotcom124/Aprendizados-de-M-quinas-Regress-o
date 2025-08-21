!wget -q "https://raw.githubusercontent.com/andre-marcos-perez/ebac-course-utils/main/dataset/nba.csv"

import pandas as pd
import numpy as np
import seaborn as sns

nba = pd.read_csv("nba.csv")

nba.head()

nba.info()

nba.describe().T

nba.drop(['rating', 'draft_year'], axis= 1).describe()

data = nba[["height", "weight"]]

data.head()

data[["height"]].head()

data['height'] = data['height'].apply(lambda height: float(height.split(sep='/')[-1].strip()))

data[['height']].describe().T

data['weight'].head()

data.loc[:, 'weight'] = pd.to_numeric(data['weight'].astype(str).str.replace(' kg.', '', regex=False).apply(lambda x: x.split('/')[-1].strip()), errors='coerce')

data['weight'].head()

data.head()

"""

# 1.   *Treino / Teste*


"""

from sklearn.model_selection import train_test_split

predictors_train, predictors_test, target_train, target_test= train_test_split(
    data.drop(['weight'], axis= 1),
    data['weight'],
    test_size = 0.25,
    random_state = 123
)

predictors_train.head()

predictors_train.shape

predictors_test.head()

predictors_test.shape

target_test.head()

target_test.shape

target_train.head()

target_train.shape