import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBClassifier

user_data = pd.read_csv('Recommendation Engine Challenge/user_data.csv')
problem_data = pd.read_csv('Recommendation Engine Challenge/problem_data.csv')
test = pd.read_csv('Recommendation Engine Challenge/test_submissions.csv')
train = pd.read_csv('Recommendation Engine Challenge/train_submissions.csv')

train = train.merge(problem_data, on='problem_id')
train = train.merge(user_data, on='user_id')
print(train.head())

test = test.merge(problem_data, on='problem_id')
test = test.merge(user_data, on='user_id')
test_ids = test.pop('ID')

#one-hot encode
train = pd.concat([train.drop('tags', axis=1), train.tags.str.get_dummies(sep=',').add_prefix('tag_')],1)
test = pd.concat([test.drop('tags', axis=1), test.tags.str.get_dummies(sep=',').add_prefix('tag_')],1)

#print(f'NaN in train % : {(train.level_type.isna().sum()/train.shape[0]*100):.2f}%')
#print(f'NaN in test % : {(test.level_type.isna().sum()/test.shape[0]*100):.2f}%')

train['level_type'] = train['level_type'].fillna('A')
test['level_type'] = test['level_type'].fillna('A')

le = LabelEncoder()
train['level_type'] = le.fit_transform(train['level_type'])
test['level_type'] = le.transform(test['level_type'])

le = LabelEncoder()
train['rank'] = le.fit_transform(train['rank'])
test['rank'] = le.transform(test['rank'])

train['user_id'] = train['user_id'].str[5:].astype(int)
test['user_id'] = test['user_id'].str[5:].astype(int)

train['problem_id'] = train['problem_id'].str[5:].astype(int)
test['problem_id'] = test['problem_id'].str[5:].astype(int)

#print(f'NaN values present in train for country column in terms of percentage value : {(train.country.isna().sum()/train.shape[0]*100):.2f}%')
#print(f'NaN values present in test for country column in terms of percentage value : {(test.country.isna().sum()/test.shape[0]*100):.2f}%')

imp = IterativeImputer()
train_country = train.pop('country')
train_cols = train.columns
train = pd.DataFrame(imp.fit_transform(train), columns = train_cols)
train = train.astype(int)
train['country'] = train_country


test_country = test.pop('country')
test_cols = test.columns
test = pd.DataFrame(imp.fit_transform(test), columns = test_cols)
test = test.astype(int)
test['country'] = test_country

le = LabelEncoder()
le_tr = le.fit(train['country'].dropna())
country_map = dict(zip(le.classes_, range(len(le.classes_))))
train['country'] = train['country'].map(country_map)
test['country'] = test['country'].map(country_map)

imp = IterativeImputer()
train_cols = train.columns
train = pd.DataFrame(imp.fit_transform(train), columns = train_cols)
train = train.astype(int)

test_cols = test.columns
test = pd.DataFrame(imp.fit_transform(test), columns = test_cols)
test = test.astype(int)

#print(train.isna().sum().any())
#print(test.isna().sum().any())

X_train = train.drop('attempts_range', axis=1)
y_train = train['attempts_range']

print(f'X_train.shape: {X_train.shape}')
print(f'y_train.shape: {y_train.shape}')
print(f'test.shape: {test.shape}')

#XGBoosted
x_trai, x_test, y_trai, y_test = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, random_state=1)
xgb = XGBClassifier(max_depth=12, eta=0.3)
xgb.fit(x_trai, y_trai)
y_predicted_test = xgb.predict(x_test)
y_predicted_train = xgb.predict(x_trai)
print(f"f1-score for Train : {f1_score(y_trai, y_predicted_train, average= 'weighted')}")
print(f"f1-score for Test : {f1_score(y_test, y_predicted_test, average= 'weighted')}")

xgb.fit(X_train, y_train)

y_test_predicted = xgb.predict(test)

#print(y_test_predicted)


Submission = pd.DataFrame()
Submission['ID'] = test_ids
Submission['attempts_range'] = y_test_predicted

test1 = pd.read_csv('Recommendation Engine Challenge/test_submissions.csv')
test1 = test1.merge(Submission, on='ID')

test1.to_csv('test_predictions.csv', index=False)