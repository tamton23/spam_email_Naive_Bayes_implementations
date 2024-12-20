import glob
import os

def load(e_mails = [], labels = []):
	file_path_spam = 'data/enron3/spam/'
	for filename in glob.glob(os.path.join(file_path_spam, '*.txt')):
		with open(filename, 'r', encoding = "ISO-8859-1") as f:
			e_mails.append(f.read())
			labels.append(1)# nhãn của spam la 1 
			
	file_path_ham = 'data/enron3/ham/'
	for filename in glob.glob(os.path.join(file_path_ham, '*.txt')):
        	with open(filename, 'r', encoding = "ISO-8859-1") as infile:
                	e_mails.append(infile.read())
                	labels.append(0)# hợp lý là 0 
