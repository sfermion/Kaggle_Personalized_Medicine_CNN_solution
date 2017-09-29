import numpy as np
import pandas as pd
import re

from multiprocessing import Pool, cpu_count
from nltk.corpus import stopwords
#nltk.download() 

try:
    cpus = cpu_count()
except NotImplementedError:
    cpus = 2   # arbitrary default



def remove_stopword(s):
    s_list = s.split()
    words = [w for w in s_list if not w in stopwords.words("english")]
    return words

def clean_str(s):
    """Cleaning text """
    s = re.sub("(\(\d*\))","",s)
    s = re.sub(r"\/"," ",s)
    s = re.sub(r">"," ",s)
    s = re.sub(r"<"," ",s)
    s = re.sub(r"-"," ",s)
    s = re.sub(r"–"," ",s)
    s = re.sub(r"\'s", " 's", s)
    s = re.sub(r"\'ve", " 've", s)
    s = re.sub(r"n\'t", " n't", s)
    s = re.sub(r"\'re", " 're", s)
    s = re.sub(r"\'d", " 'd", s)
    s = re.sub(r"\'ll", " 'll", s)
    s = re.sub(r"\(", " ( ", s)
    s = re.sub(r"\)", " ) ", s)
    s = re.sub(r"\?", " ? ", s)
    s = re.sub(r" a "," ",s)
    s = re.sub(r" the "," ",s)
    s = re.sub(r"(\(Fig\..*\))"," ",s)
    s = re.sub(r"(\(Table\..*\))"," ",s)
    s = re.sub(r"Fig.\S+ "," ",s)
    s = re.sub(r"Table "," ",s)
    s = re.sub(r", "," ",s)
    s = re.sub(r"[().?\[\]!;:+]", " ", s)
    s = re.sub(r" '\S+", "", s)
    s = re.sub(r"   "," ",s)
    s = re.sub(r"  "," ",s)
    words = remove_stopword(s)
    # s = " ".join(set(words))
    s = " ".join(words)
    return s.strip()

def load_data_and_labels(variants_file,text_file):
    """Load text and Classes"""
    var = pd.read_csv(variants_file)
    txt = pd.read_csv(text_file, sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
    x_text = list(txt['Text'])
    print ("Cleaning data ...\n")
    # x_text = [clean_str(sent) for sent in x_text]

    pool = Pool(processes=cpus)
    x_text = pool.map(clean_str, x_text)
    pool.close()
    pool.join()  
    # Map the actual classes to one hot labels
    try:
        # class_vec = np.array(list(var['Class']-1))
        # m = class_vec.size
        # n = 9 #num_of_class
        # y = np.zeros((m, n))
        # y[np.arange(m), class_vec] = 1
        labels = sorted(list(set(var['Class'].tolist())))
        one_hot = np.zeros((len(labels), len(labels)), int)
        np.fill_diagonal(one_hot, 1)
        label_dict = dict(zip(labels, one_hot))
        y = var['Class'].apply(lambda y: label_dict[y]).tolist()
    except:
        y = None
        labels = None
    return [x_text, y, labels]

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