from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import Loaddata as l
import Filteremail as fl 

e_mails = []
labels = []
l.load(e_mails, labels)
cleaned_emails = fl._clean_text(e_mails)
cv = CountVectorizer(stop_words="english", max_features=500)

X_train, X_test, Y_train, Y_test = train_test_split(cleaned_emails, labels, test_size=0.3)
term_docs_train = cv.fit_transform(X_train)#feature train       
label_index = fl._get_label_index(Y_train)
prior = fl._get_prior(label_index)
smoothing = 1
likelihood = fl._get_likelihood(term_docs_train, label_index, smoothing)
term_docs_test = cv.transform(X_test)#feature test      
posterior = fl._get_posterior(term_docs_test, prior, likelihood)
correct = 0.0
for pred, actual in zip(posterior, Y_test):
	if actual == 1:
		if pred[1] >= 0.5:
			correct += 1
	elif pred[0] > 0.5:
		correct += 1
print('the accuracy on {0} testing samples = {1:.1f}%'.format(len(Y_test), correct/len(Y_test)*100))

