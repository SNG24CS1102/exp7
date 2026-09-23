import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import classification_report, accuracy_score,f1_score

categories = ['comp.graphics', 'rec.sport.baseball', 'sci.med', 'talk.politics.misc']
print("Fetching datasets.....")
train_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
test_data = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

y_train = train_data.target
y_test = test_data.target
print(f"Loaded {len(train_data.data)} training documents and {len(test_data.data)} testing documents.")


mnb_vectorizer = CountVectorizer(stop_words='english', max_features=10000)
x_train_counts = mnb_vectorizer.fit_transform(train_data.data)
x_test_counts = mnb_vectorizer.transform(test_data.data)


bnb_vectorizer = CountVectorizer(stop_words='english', binary=True, max_features=10000)
x_train_binary = bnb_vectorizer.fit_transform(train_data.data)
x_test_binary = bnb_vectorizer.transform(test_data.data)

print("Vectorization strategies initialized and applied.")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(x_train_counts, y_train)
y_pred_mnb = mnb.predict(x_test_counts)

bnb = BernoulliNB(alpha=1.0)
bnb.fit(x_train_binary, y_train)
y_pred_bnb = bnb.predict(x_test_binary)

results = {
    "MultinomialNB": {
        "accuracy": accuracy_score(y_test, y_pred_mnb),
        "f1_score": f1_score(y_test, y_pred_mnb, average='weighted')
    },
    "BernoulliNB": {
        "accuracy": accuracy_score(y_test, y_pred_bnb),
        "f1_score": f1_score(y_test, y_pred_bnb, average='weighted')
    }
}       

import pandas as pd
df_results = pd.DataFrame(results, index=['accuracy', 'f1_score'])  
print(df_results.round(4))

print(""+"="*25+ "MultinomialNB Classification Report" + "="*25)
print(classification_report(y_test, y_pred_mnb, target_names=categories))
print(""+"="*26+ "BernoulliNB Classification Report" + "="*26)
print(classification_report(y_test, y_pred_bnb, target_names=categories))
