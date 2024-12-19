spam_email_Naive_Bayes_implementations-ebook-packt-
# Introduction

 - Use Naive Bayes
   ![Screenshot from 2024-11-15 21-04-45](https://github.com/user-attachments/assets/f3a4df29-b7e9-4f65-bb33-99ce1f9e8020)
   + Feature - P(Ei|H)
   + Prior - P(H)
   + Posterior - P(H|E)
 - python3.12 in venv
 - library: nltk, scikit-learn, numpy
   
    
# Data  [install data email] 
    http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz.
# In Linux

    ls -l enron1/ham/*.txt |wc -l 
    ls -l enron1/spam/*.txt |wc -l 
to check folder constrain file NoSpam and Spam
# Implement Module
step: loadSpamEmails_labels.py -> loadHamEmails_labels.py -> clean_data.py -> exreacting_features.py -> prior.py -> likelihood.py -> posterior.py (test on some content emails)

1. use exec(open("file.py").read()) to run module or create init call module.
        exp: exec(open("loadHamEmails_labels.py).read())
2. exec Module loadSpamEmail and loadHamEmail with label have elements is 0(ham email) and 1(spam email).
       -> Here there are two variables e_emails(length = 5172) and labels(length = 5172)
   #
        exec(open("loadSpamEmails_labels.py").read())
        exec(open("loadHamEmails_labels.py").read())
        print(f"length emails: {len(e_mails)} \nlength labels: {len(labels)}")
4. use Module clean date to clean files email. 
       -> Module includes func clean_text(parameter is e_mails) to reformat characters and formats in each email file.
    #
       exec(open("clean_data.py").read())
       cleaned_emails = clean_text(e_mails)
   and we can show one content of email.
    #
       print(cleaned_emails[0])
6. after we format data, we will extracting feature, and file "extracting_features.py"  will do its job.
        -> the "max_feature" parameter set 500 and it can be modified for better accuracy.
   #
       exec(open("extracting_features.py").read())
8. Starting with the prior, we first group the data by label: use module prior.py
   # P(S)
       exec(open("prior.py").read())
       prior = get_prior(labels_index)
9. likelihood: probability of that feature appearing on the label. P(feature(word)|label)
   # P(Xi|S)
       exec(open("prior.py").read())
       smoothing = 1
       likelihood = get_likelihood(term_docs,label_index, smoothing)
   smoothing parameter: which can also be 0 for no smoothing and any other positive value, as long as high classification performance is achieved: 
   check 5 probability of feature - Ham emails
   #
       print(likelihood[0][:5])
   and 5 probability of feature - Spam emails
   # 
       print(likelihood[1][:5])
11. posterior: probabilities P(feature | class)
    #
        exec(open("posterior.py").read())
        # test two emails spam and ham
        email_test = [e_mail[0], e_mails[5000]]
        cleaned_test = clean_text(email_test)
        term_docs_test = cv.transform(cleaned_test)
        posterior = get_posterior(term_docs_test, prior, likelihood)
        print(posterior)
    first email is spam email and second email is ham emails
12. train naive Bayes
    #
        exec(open("train_naiveBayes_emails.py").read())
    The naive Bayes classifier we just developed line by line correctly classifies ~=
91% of emails!
