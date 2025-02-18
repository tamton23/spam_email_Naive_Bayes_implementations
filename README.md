# spam_email_Naive_Bayes_implementations-ebook-by-Packt
# Data  [install data email] 
    http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz
# In Linux

    ls -l enron1/ham/*.txt |wc -l 
    ls -l enron1/spam/*.txt |wc -l 
to check folder constrain file NoSpam and Spam
# Implement Module
step: loadSpamEmails_labels.py -> loadHamEmails_labels.py -> clean_data.py -> exreacting_features.py -> prior.py -> likelihood.py -> posterior.py (test on some content emails)
Chương 1: Dữ liệu và source code.
1.1. Dữ liệu.
Dữ liệu để train và test được lấy trong ebook, có 6 cơ sở dữ liệu:
Link: http://www.aueb.gr/users/ion/data/enron-spam/
Link tải trực tiếp 1 trong các cơ sở dữ liệu( link dự phòng)
Link: http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz
1.2. Source code:
Yêu cầu:
+ Python (khiến nghị phiên bản mới nhất)
+ Thư viện nltk
+ Thư viện sklearn (khi tải 2 thư viện trên python sẽ tự động tích hợp các thư viện yêu cầu)
+ Môi trường python(khiến nghị, tránh xung đột)
Code bao gồm: 
	+ Filteremail.py
+ NaiveBayes_emails.py
+ Loaddata.py
Link source: https://github.com/tamton23/spam_email_Naive_Bayes_implementations
Chương 2: Thực thi tập lệnh
Trước khi thực thi trong tập lệnh Loaddata.py cập nhật đường link dữ liệu.
2.1. Thực thi trực tiếp ra kết quả: 2 cách chạy 
Cách 1:
    • python NaiveBayes_emails.py
Cách 2: Trong giao diện lập trình python sử dụng lệnh

	>> exec(open(“NaiveBayes_emails.py”).read())
Kết quả của dòng lệnh:

2.2. Thực thi chi tiết:
 B1. Load dữ liệu và gán nhãn
 B2. Định dạng lại nội dung email - ( _clean_data())
 B3. Features - ( _get_features())
 B4. Tính prior( xác suất tiên nghiệm) - (_get_prior())
 B5. Likelihood: P(features|labels) - (_get_likelihood())
 B6. Posterior(xác suất hậu nghiệm) - ( _get_posterior())
 B7. Train - test
2.2.1. Load dữ liệu và gán nhãn:
Trong môi trường Python chạy lệnh python.

	>>> e_mails = [ ]
	>>> labels = [ ]
	>>> import Loaddata as l
	>>> l.load(e_mails, labels)
kiểm tra danh sách xem đã có dữ liệu chưa.

	>>> len(e_mails)
5512
Hiển thị: 

2.2.2. Định dạng nội dung email: sử dụng func _clean_text(tham số) trong Filteremail.py

	>>> import Filteremail as fl
	>>> cleaned_emails = fl._clean_text(e_mails)
So sánh email chưa định dạng và email đã định dạng

	>>> e_mail[0]
	>>> cleaned_emails[0]
Hiển thị: 


2.2.3. Features: _get_features. của Filteremail.py
Nhập tiếp tục: 

	>>> term_docs = fl._get_feature(cleaned_emails)
Xem có bao nhiêu thuật ngữ xuất hiện trong email.

	>>> print(term_docs[0])
2.2.4. Prior: _get_label_index(), _get_prior()
Ta cần group nhãn trước khi tính xác suất tiên nghiệm.

	>>> label_index = fl._get_label_index(labels)
Tính prior.

	>>> prior = fl._get_prior(label_index)
Xem kết quả:

2.2.5. Likelihood: _get_likelihood(tham số 1,ts 2,ts 3)
Đưa vào 3 đối số: feature, label_index và số làm mềm 

	>>> likelihood = fl._get_likelihood(term_docs, label_index, 1)
Xem xác suất của 5 email thuộc nhãn 1

	>>> likelihood[1][:5]

2.2.6. Posterior(xác suất hậu nghiệm):_get_posterior
Khi đã có term_docs, prior, likelihood ta sẽ tính posterior.

	>>> posterior = fl._get_posterior(term_docs, prior, likelihood)
Xem xác suất của 5 email đầu.

	>> posterior[:5]
2.2.7. Train - Test:

	>>> exec(open(“NaiveBayes_emails.py").read())
Tính được xác suất sấp xỉ 94% với 70% email train và 30% email test

