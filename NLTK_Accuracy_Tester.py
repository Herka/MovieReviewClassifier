import pickle
import nltk
import json
import re
import nltk.classify.util
from nltk.corpus import movie_reviews


f = open("Ebert_Classifier.pickle")
classifier = pickle.load(f)
f.close()




def word_feats(words):
    return dict([(word, True) for word in words])
 
negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')
 
negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'negative') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'positive') for f in posids]
 

testfeats = negfeats + posfeats


print "Accuracy basierend auf dem Movie Review Corpus ist: ", nltk.classify.util.accuracy(classifier, testfeats)