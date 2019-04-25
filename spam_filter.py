import re
import csv
from pprint import pprint
import numpy
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline,FeatureUnion
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.base import TransformerMixin
import matplotlib.pyplot as plt
import scikitplot as skplt

#custom transformers
class DataFrameColumnExtracter(TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]

class BoolVectorizer(TransformerMixin):
	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		return numpy.array([int(x == True) for x in X.values]).reshape(-1, 1)

#lemmatokenizer
class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()

	def __call__(self, mail):
		return [self.wnl.lemmatize(word.lower()) for word in mail if word.lower() not in stopwords.words('english')]

if __name__== "__main__":

	data_set = pd.read_csv('data/nlp_assignment_dataset.csv', header=0)

	# split and shuffle dataset
	data_set_shuffled = shuffle(data_set)
	train_set, test_set = train_test_split(data_set_shuffled, test_size=0.2)

	text_tf_idf_vectorizer = TfidfVectorizer(max_features=500,tokenizer=LemmaTokenizer(),ngram_range=(1,2))
	text_pipeline = Pipeline([
						       ('raw_feature_extraction',DataFrameColumnExtracter('text')),
						       ('text_tf_idf_vectorizer',text_tf_idf_vectorizer)
	])

	link_vectorizer = BoolVectorizer()
	link_pipeline = Pipeline([
						       ('raw_feature_extraction',DataFrameColumnExtracter('has_link')),
						       ('link_vectorizer',link_vectorizer)
	])

	image_vectorizer = BoolVectorizer()
	image_pipeline = Pipeline([
						       ('raw_feature_extraction',DataFrameColumnExtracter('has_image')),
						       ('image_vectorizer',image_vectorizer)
	])


	chi2_feature_selection = SelectKBest(chi2, k=175)
	nb_classifier = MultinomialNB(fit_prior=True)

	spam_classifier_pipeline = Pipeline([
										('features', FeatureUnion([
									      ('text_vectorizer', text_pipeline), # extract text features
									      ('link_vectorizer', link_pipeline), # extract link features
									      ('image_vectorizer', image_pipeline), # extract image features
									    ])),
										('chi2', chi2_feature_selection), #select top features
										('nb_classifier', nb_classifier)]) #classification

	# Train a classifier of your choice
	spam_classifier_pipeline.fit(train_set[['text','has_link','has_image']],train_set[['label']].values)

	# Print the performance of your model (the performance itself does not matter at this stage)
	probabilities = spam_classifier_pipeline.predict_proba(test_set[['text','has_link','has_image']])
	skplt.metrics.plot_precision_recall_curve(test_set[['label']].values, probabilities)
	plt.show()
