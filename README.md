# spam_email_Naive_Bayes_implementations-ebook-by-Packt
# Data  [install data email] 
    http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz
# In Linux

    ls -l enron1/ham/*.txt |wc -l 
    ls -l enron1/spam/*.txt |wc -l 
to check folder constrain file NoSpam and Spam
# Implement Module
step: loadSpamEmails_labels.py -> loadHamEmails_labels.py -> clean_data.py -> exreacting_features.py -> prior.py -> likelihood.py -> posterior.py (test on some content emails)

1. use exec(open("file.py").read()) to run module or create init call module.
        exp: exec(open("loadHamEmails_labels.py).read())
2. exec Module loadSpamEmail and loadHamEmail with label 0 and 1.
       -> Here there are two variables e_emails(length = 5172) and labels(length = 5172)
   #
        exec(open("loadSamEmails_labels.py").read())
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
7. Starting with the prior, we first group the data by label: use module prior.py
