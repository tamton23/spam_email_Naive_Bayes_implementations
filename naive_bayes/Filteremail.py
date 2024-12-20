import numpy as np
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

def _clean_text(docs):
	all_names = set(names.words())
	lemmatizer = WordNetLemmatizer()
	cleaned_docs = []
	for doc in docs:
		cleaned_docs.append(' '.join([lemmatizer.lemmatize(word.lower())
		for word in doc.split()
			if word.isalpha()
				and word not in all_names]))
	return cleaned_docs


def _get_feature(cleaned_data):
	cv = CountVectorizer(stop_words="english", max_features=500)
	# stop word 'english' (and, the, him ... duoc cho la khong cung cap tinh hieu)
	# max_feature '500' 500 shape co the nang cao len de dat do chinh sac cao hon
	return cv.fit_transform(cleaned_data)

def _get_label_index(labels):
        """group cac labels lai de ra mot varible label_index
        
        """
        label_index = defaultdict(list)
        for index, label in enumerate(labels):
                label_index[label].append(index)
                
        return label_index

def _get_prior(label_index):
	
	"""	Compute prior based on training samples
	Args:
		label_index (group cac label)
	Returns:
		dictionary, with class label as key, corresponding prior as the value
	"""
	prior = {label: len(index) for label, index in label_index.items()}
	total_count = sum(prior.values())
	for label in prior:
		prior[label] /= float(total_count)
		"""P[spam] and P[ham]
                        VD:P[spam] = 1500/5172 = 0.290..
                """
	return prior

def _get_likelihood(term_document_matrix, label_index, smoothing = 0):
	""" Compute likelihood based on training samples
	Args:
		term_document_matrix (sparse matrix)
		label_index (grouped sample indices by class)
		smoothing (integer, additive Laplace smoothing parameter)
	Returns:
		dictionary, with class as key, corresponding conditional probability P(feature|class) vector as value
	"""

	likelihood = {}
	for label, index in label_index.items():
		likelihood[label] = term_document_matrix[index, :].sum(axis=0) + smoothing
		likelihood[label] = np.asarray(likelihood[label])[0]
		total_count = likelihood[label].sum()
		# Sum of all feature from class spam or ham
		likelihood[label] = likelihood[label] / float(total_count)

	return likelihood

def _get_posterior(term_document_matrix, prior, likelihood):
        
	""" Compute posterior of testing samples, based on prior and likelihood
	Args:
		term_document_matrix (sparse matrix)
		prior (dictionary, with class label as key, corresponding prior as the value)
		likelihood (dictionary, with class label as key,
		corresponding conditional probability vector as value)
	Returns:
		dictionary, with class label as key, corresponding posterior as value
        """
	num_docs = term_document_matrix.shape[0]
	posteriors = []
	for i in range(num_docs):
	# posterior is proportional to prior * likelihood
	# = exp(log(prior * likelihood))
	# = exp(log(prior) + log(likelihood))
                posterior = {key: np.log(prior_label) for key, prior_label in prior.items()}
                for label, likelihood_label in likelihood.items():
                        term_document_vector = term_document_matrix.getrow(i)
                        counts = term_document_vector.data
                        indices = term_document_vector.indices
                        for count, index in zip(counts, indices):
                                posterior[label] += np.log(likelihood_label[index]) * count
		# exp(-1000):exp(-999) will cause zero division error,
		# however it equates to exp(0):exp(1)
                min_log_posterior = min(posterior.values())
                for label in posterior:
                        try:
                                posterior[label] = np.exp(posterior[label] - min_log_posterior, dtype=np.float128)
                        except:
			# if one's log value is excessively large, assign it infinity
                                posterior[label] = float('inf')
                # normalize so that all sums up to 1
                sum_posterior = sum(posterior.values())
                for label in posterior:
                        if posterior[label] == float('inf'):
                                posterior[label] = 1.0
                        else:
                                posterior[label] /= sum_posterior
                posteriors.append(posterior.copy())
	return posteriors
##
