import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics,
svm
data = pd.read_csv('assignment1_data.csv', encoding='latin-1')
'''
设置选取高频字符/单词个数
'''
most_freq_numbers = 20
'''
统计在消息中出现频率前30的符号/单词
'''
count1 = Counter("
".join(data[data['spam']==0]["text"]).split()).most_common(most_freq_numbers)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words in non-spam", 1 : "count"})
count2 = Counter("
".join(data[data['spam']==1]["text"]).split()).most_common(most_freq_numbers)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0: "words in spam", 1 : "count_"})
'''
用柱装图显示非垃圾信息中的高频字符/单词及其频数，由高到低显示
'''
df1.plot.bar(legend = False, color = 'red')
y_pos = np.arange(len(df1["words in non-spam"]))
plt.xticks(y_pos, df1["words in non-spam"])
plt.title('More frequent words in non-spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()

'''
用柱装图显示垃圾信息中的高频字符/单词及其频数，由高到低显示
'''
df2.plot.bar(legend = False, color = 'orange')
y_pos = np.arange(len(df2["words in spam"]))
plt.xticks(y_pos, df2["words in spam"])
plt.title('More frequent words in spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()

f = feature_extraction.text.CountVectorizer(stop_words = 'english')
X = f.fit_transform(data["text"])
'''
将数据集切割成训练集和测试集，测试集比例为0.3
'''
data["spam"]=data["spam"].map({1:1,0:0})
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
data['spam'], test_size=0.3, random_state=42)
'''
设置参数选择范围
'''
list_alpha = np.arange(1.0e-10, most_freq_numbers, 0.1)
score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test= np.zeros(len(list_alpha))
count = 0
for alpha in list_alpha:
bayes = naive_bayes.MultinomialNB(alpha=alpha)
bayes.fit(X_train, y_train)
score_train[count] = bayes.score(X_train, y_train)
score_test[count]= bayes.score(X_test, y_test)
recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
precision_test[count] = metrics.precision_score(y_test,
bayes.predict(X_test))
count = count + 1
matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test,
precision_test])
models = pd.DataFrame(data = matrix, columns = ['alpha', 'Train Accuracy',
'Test Accuracy', 'Test Recall', 'Test Precision'])
best_index = models['Test Precision'].idxmax()
print('**********保证Precision最大时的结果*************')
print(models.iloc[best_index, :])
#设置测试精确度阈值为1,保证正常邮件不会被误判
best_index = models[models['Test Precision']==1]['Test Accuracy'].idxmax()
bayes = naive_bayes.MultinomialNB(alpha=list_alpha[best_index])
bayes.fit(X_train, y_train)
print('**********保证Precision=1时Accuracy最大的结果**************')
print(models.iloc[best_index, :])
m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
print('***********判断结果***********')
#打印分类结果
print(pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0',
'Predicted 1'],index = ['Actual 0', 'Actual 1']))
'''
对于非垃圾邮件全部识别，不会误判，对于垃圾邮件拦截率大约50%
'''