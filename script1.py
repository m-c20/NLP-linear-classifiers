import sys
# splitting/loading data 
from sklearn.model_selection import train_test_split
# turn text into vectors and grams
from sklearn.feature_extraction.text import CountVectorizer
# naive bayes
from sklearn.naive_bayes import MultinomialNB
# logistic regression 
from sklearn.linear_model import LogisticRegression
# F1, recall, accuracy
from sklearn.metrics import accuracy_score, recall_score, f1_score
# pos tags
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))


def load_dataset(filename):
	X = []
	y = []
	with open(filename, "r") as file:
		for line in file:
			parts = line.strip().split("\t",1)
			label, text = parts
			y.append(int(label))
			X.append(text)
	return X, y



def ngram_naive_bayes(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)
	vectorizer= CountVectorizer(ngram_range=(1,1))
	X_train_vec = vectorizer.fit_transform(X_train)
	X_test_vec = vectorizer.transform(X_test)
	
	nb = MultinomialNB()
	nb.fit(X_train_vec, y_train)

	y_pred = nb.predict(X_test_vec)

	acc = accuracy_score(y_test, y_pred)
	rec = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)
	
	print("========nb=======")
	print("")
	print("acc:	", acc)
	print("recall:	", rec)
	print("f1:	",f1)
	
def ngram_logreg(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)
	vectorizer= CountVectorizer(ngram_range=(1,1))
	X_train_vec = vectorizer.fit_transform(X_train)
	X_test_vec = vectorizer.transform(X_test)


	logreg = LogisticRegression(max_iter=1000)
	logreg.fit(X_train_vec,y_train)
	
	y_pred = logreg.predict(X_test_vec)

	acc = accuracy_score(y_test, y_pred)
	rec = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)

	print("========LR=======")
	print("")
	print("acc:	", acc)
	print("recall:	", rec)
	print("f1:	",f1)
def pos_convert(s):
	tokens = nltk.word_tokenize(s)
	tagged = nltk.pos_tag(tokens)
	pos_tags = []
	for t in tagged:
		pos_tags.append(t[1])
	return " ".join(pos_tags)

def pos_naive_bayes(X,y):
	print("++++++++pos++++++")
	X_pos = []
	for s in X:
		X_pos.append(pos_convert(s))
	 
	ngram_naive_bayes(X_pos,y)
	
def pos_logreg(X,y):
	print("+++++++pos++++++++")
	X_pos = []
	for s in X:
		X_pos.append(pos_convert(s))
	ngram_logreg(X_pos,y)

def rm_stopwords(s):
	tokens = nltk.word_tokenize(s)
	s_new = []
	for t in tokens:
		if t.lower() not in stop_words:
			s_new.append(t)
	return " ".join(s_new)

def stopwords_logreg(X,y):
	#print("+++++++++Stopwords+++++++")
	X_new = []
	for s in X:	
		X_new.append(rm_stopwords(s))
	ngram_logreg(X_new,y)
def stopwords_nb(X,y):
       # print("+++++++++Stopwords+++++++")
        X_new = []
        for s in X:
                X_new.append(rm_stopwords(s))
        ngram_naive_bayes(X_new,y)

	



if __name__ == "__main__":
	filename = sys.argv[1]
	X,y = load_dataset(filename)
	print("Experiment Results on " + filename)
	print("")
	print("======Naive Bayes======")
	print("--------n-gram-------")
	ngram_naive_bayes(X,y)
	print("")
	print("")
	print("--------POS Tags-------")
	pos_naive_bayes(X,y)
	print("")
	print("")
	print("--------removed Stopwords-------")	
	stopwords_nb(X,y)
	print("======Logistic Regression======")
	print("")
	print("--------n-gram-------")
	ngram_logreg(X,y)
	print("")
	print("")
	print("--------POS Tags-------")
	pos_logreg(X,y)
	print("")
	print("")
	print("--------removed Stopwords-------")
	stopwords_logreg(X,y)

