from sklearn import *
import sklearn
import numpy as np
import pandas as pd
import re

from multiprocessing import Pool, cpu_count
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
#import nltk
#nltk.download() 

try:
    cpus = cpu_count()
except NotImplementedError:
    cpus = 2   # arbitrary default


paper_stop_words = [ "fig", "figure", "et", "al", "table", 
            "data", "analysis", "analyze", "study",
            "method", "result", "conclusion", "author",
            "find", "found", "show", "perform",
            "demonstrate", "evaluate", "discuss"]

def clean_str(s):
    """Cleaning text """
    # s = re.sub("(\(\d*\))","",s)
    # s = re.sub(r"\/"," ",s)
    # s = re.sub(r">"," ",s)
    # s = re.sub(r"<"," ",s)
    # s = re.sub(r"-"," ",s)
    # s = re.sub(r"–"," ",s)
    # s = re.sub(r"\'s", " 's", s)
    # s = re.sub(r"\'ve", " 've", s)
    # s = re.sub(r"n\'t", " n't", s)
    # s = re.sub(r"\'re", " 're", s)
    # s = re.sub(r"\'d", " 'd", s)
    # s = re.sub(r"\'ll", " 'll", s)
    # s = re.sub(r" a "," ",s)
    # s = re.sub(r" the "," ",s)
    # s = re.sub(r"(\(Fig\..*\))"," ",s)
    # s = re.sub(r"(\(Table\..*\))"," ",s)
    # s = re.sub(r"Fig.\S+ "," ",s)
    # s = re.sub(r"Table "," ",s)
    # s = re.sub(r", "," ",s)
    # s = re.sub(r"[().?\[\]!;:+]", " ", s)
    # s = re.sub(r" '\S+", "", s)
    # s = re.sub(' +',' ',s)
    s = re.sub(r" \d*", " ", s)
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(s.lower())
    stemmer = SnowballStemmer("porter")
    words = [w for w in words if not w in stopwords.words("english")]
    words = [w for w in words if not w in paper_stop_words]
    words = [stemmer.stem(w) for w in words]
    s = " ".join(words)
    return s

def load_data_and_labels(variants_file,text_file):
    """Load text and Classes"""
    var = pd.read_csv(variants_file)
    txt = pd.read_csv(text_file, sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
    txt_var = pd.merge(var, txt, how='left', on='ID')

    for index, row in txt_var.iterrows():
        if len(row['Text']) < 10:
            txt_var.drop(index,inplace=True)

    x_text = list(txt_var['Text'])
    print ("Cleaning data ...\n")

    # x_text = [clean_str(sent) for sent in x_text]

    pool = Pool(processes=cpus)
    x_text = pool.map(clean_str, x_text)
    pool.close()
    pool.join()  

    # Map the actual classes to one hot labels
    try:
        labels = sorted(list(set(txt_var['Class'].tolist())))
        one_hot = np.zeros((len(labels), len(labels)), int)
        np.fill_diagonal(one_hot, 1)
        label_dict = dict(zip(labels, one_hot))
        y = txt_var['Class'].apply(lambda y: label_dict[y]).tolist()
    except:
        y = None
        labels = None
    return [x_text, y, labels]



class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x.drop(['Gene', 'Variation','ID','Text'],axis=1).values
        return x

class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.key].apply(str)



def gene_related_features(variants_file,text_file):

    var = pd.read_csv(variants_file)
    txt = pd.read_csv(text_file, sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

    train = pd.merge(var, txt, how='left', on='ID')

    for index, row in train.iterrows():
        if len(row['Text']) < 10:
            train.drop(index,inplace=True)

    train = train.drop(['Class'], axis=1)
    train['Gene_Share'] = train.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
    train['Variation_Share'] = train.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)


    for i in range(56):
        train['Gene_'+str(i)] = train['Gene'].map(lambda x: str(x[i]) if len(x)>i else '')
        train['Variation'+str(i)] = train['Variation'].map(lambda x: str(x[i]) if len(x)>i else '')


    gen_var_lst = sorted(list(train.Gene.unique()) + list(train.Variation.unique()))
    print(len(gen_var_lst))
    gen_var_lst = [x for x in gen_var_lst if len(x.split(' '))==1]
    print(len(gen_var_lst))
    i_ = 0

    for gen_var_lst_itm in gen_var_lst:
       if i_ % 100 == 0: print(i_)
       train['GV_'+str(gen_var_lst_itm)] = train['Text'].map(lambda x: str(x).count(str(gen_var_lst_itm)))
       i_ += 1

    for c in train.columns:
        if train[c].dtype == 'object':
            if c in ['Gene','Variation']:
                lbl = preprocessing.LabelEncoder()
                train[c+'_lbl_enc'] = lbl.fit_transform(train[c].values)  
                train[c+'_len'] = train[c].map(lambda x: len(str(x)))
                train[c+'_words'] = train[c].map(lambda x: len(str(x).split(' ')))
            elif c != 'Text':
                lbl = preprocessing.LabelEncoder()
                train[c] = lbl.fit_transform(train[c].values)
            if c=='Text': 
                train[c+'_len'] = train[c].map(lambda x: len(str(x)))
                train[c+'_words'] = train[c].map(lambda x: len(str(x).split(' '))) 


    # print('Pipeline...')

    # #Defining Transformers:

    # count_Gene  = feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))
    # tsvd1 = decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12)
    # count_Variation = feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))
    # tsvd2 = decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12)

    # pi1 = pipeline.Pipeline([('Gene', cust_txt_col('Gene')), ('count_Gene', count_Gene),('tsvd1',tsvd1)])
    # pi2 = pipeline.Pipeline([('Variation', cust_txt_col('Variation')), ('count_Variation',count_Variation), ('tsvd2',tsvd2)])
    # # pi3 = pipeline.Pipeline([('Text', cust_txt_col('Text')), ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))), ('tsvd3', decomposition.TruncatedSVD(n_components=50, n_iter=25, random_state=12))])

    # fp = pipeline.Pipeline([
    #     ('union', pipeline.FeatureUnion(
    #         n_jobs = -1,
    #         transformer_list = [
    #             ('standard', cust_regression_vals()),
    #             ('pi1', pi1),
    #             ('pi2', pi2),
    #             #commented for Kaggle Limits
    #             #('pi3', pi3)
    #         ])
    #     )])

    # train = fp.fit_transform(train).astype(int)
    # print(train.shape)

    return train

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Iterate the data batch by batch"""

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]