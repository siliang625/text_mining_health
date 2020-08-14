## Requried modules
# !pip list | grep gensim
# !pip install --upgrade gensim
# !pip install nltk
# !pip install spacy
# !pip install gensim
# !pip install pyLDAvis
# !pip install -U tmtoolkit
#!python -m spacy download en_core_web_lg

"""# Loading libraries and data"""
import pandas as pd
import numpy as np
import os
import sklearn
import sys
np.set_printoptions(threshold=sys.maxsize)
from sklearn.model_selection import train_test_split
import re
from pprint import pprint
from glob import glob
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.nmf import Nmf
# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
# %matplotlib inline
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from pathlib import Path
import glob
from bs4 import BeautifulSoup
import operator
import pickle as pkl
from gensim.models.coherencemodel import CoherenceModel
import tmtoolkit

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
# %matplotlib inline

# global variable 
#root_path = "/content/drive/My Drive/RA"
root_path = "."
# def shellquote(s):
#     return "'" + s.replace("'", "'\\''") + "'"
# root_path = shellquote(root_path)
# root_path
data_path = root_path + "/data"
result_path = root_path + "/result"
# def shellquote(s):
#     return "'" + s.replace("'", "'\\''") + "'"
# data_path = shellquote(data_path)
#! ls {cur_path} | wc -l

# use shard memeory for google colab
# !df -h | grep shm
# %env JOBLIB_TEMP_FOLDER=/tmp


def get_abstract(soup):
  if soup.find("abstract") is None or soup.find("abstract") == -1:
      return "NaN"
  
  abstract = soup.find('abstract')
  return " ".join([p.text for p in abstract.findChildren("p")])


def get_pub_date(soup):
  if soup.find('pub-date') is None:
    return None

  pub_d = soup.find('pub-date')
  if pub_d.find('year') is None:
    return None

  return pub_d.find('year').text

def get_title(soup):
  if not soup.find("article-title"):
    return None
  else:
    return soup.find("article-title").text

def get_sample_data(cur_path):
  '''
  save metadata.csv
  save all paper's abstract, pub_date
  '''
  path_list = Path(cur_path).glob('**/*.xml')

  abstracts = []
  pub_dates = []
  pmc_ids = []
  titles = []

  for path in pathlist:
      
      # because path is object not string
      path_in_str = str(path)
      # name_str = path_in_str.split('/')[-1]

      soup = BeautifulSoup(open(path_in_str, 'r'))

      # processing abstract
      abstract = get_abstract(soup)
      title = get_title(soup)

      if abstract == "NaN":
        continue

      # processing time
      pub_d = get_pub_date(soup)
      if pub_d is None:
        continue
      
      pub_dates.append(int(pub_d))
      abstracts.append(abstract)
      titles.append(title)
      
      pmc_id = str(path).split('/')[-1].split('_')[0]
      pmc_ids.append(pmc_id)


  pub_dates = pd.Series(data=pub_dates)
  metadata = pd.DataFrame()
  metadata['pub_date'] = pub_dates
  metadata['pmc_id'] = pmc_ids
  metadata['title'] = titles

  return abstracts, pub_dates, metadata


def save_to_drive(file_name, data): 
  file_path = f'{result_path}/{file_name}.pkl'
  directory = os.path.dirname(file_path)
  if not os.path.exists(directory):
    os.makedirs(directory)
  with open(file_path, 'wb') as file:
    pkl.dump(data, file)


"""# Text Preprocessinng

### remove section heading
"""

# Remove section heading
# some cases: section heading inside the p tag, eg: pmc_id = 7320979
data = [re.sub('[A-Za-z]*:', '', each) for each in X]
# Remove new line characters
data = [re.sub('\s+', ' ', each) for each in data]

data[1]

"""### tokenization"""

def tokenize_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(tokenize_to_words(data))
#data_words
#data_words[:1]

"""### Co-location using genism library this time"""

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold, fewer phrases to form
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
quadgram= gensim.models.Phrases(trigram[bigram[data_words]], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
quad_mod = gensim.models.phrases.Phraser(quadgram)

# See example
print(bigram_mod[data_words[10]])
print(trigram_mod[bigram_mod[data_words[10]]])
print(quad_mod[trigram_mod[bigram_mod[data_words[10]]]])



# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en', disable=['parser', 'ner'])

# Define functions for stopwords, bigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def make_quadgrams(texts):
    return [quad_mod[trigram_mod[bigram_mod[doc]]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
      doc = nlp(" ".join(sent)) 
      texts_out.append([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags])
    return texts_out

"""### Remove Stopwords, Make quadgram and Lemmatize"""

nltk.download('stopwords')
nltk.download('punkt')
stop_words = stopwords.words('english')
#stop_words.extend(['from', 'subject', 're', 'edu', 'use']) # TODO: can add more

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Quadgrams
data_words_bigrams = make_quadgrams(data_words_nostops)

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:2])
# len(data_lemmatized)

save_to_drive("data_lemmatized", data_lemmatized)

data_lemmatized = load_from_drive(root_path,"data_lemmatized")

"""# Build Models

- model title follow pattern: **{model}_{library}_{number_of_topics}**, eg: lda_sklearn_10

- 10, 20, 100 topics

## LDA using Sklearn

### Transform tokenized words into correct format for Countvectorizer
"""

# transform data_lematized to a format where lda use (list of strings, each doc is a string)
res_data = []
for doc in data_lemmatized:
  res_data.append(" ".join(doc))
print(len(res_data))

res_data[1]

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
cv = CountVectorizer(analyzer='word', min_df = 3,    # max_df_ as default, # max_df=0.80, max_df=0.80, max_features=10000
                     token_pattern='[a-zA-Z0-9]{3,}', lowercase=True,  
                     stop_words = 'english')
#cv_feature_names = cv.get_feature_names()#
# now using consistent data pre-processing results 
data_vectorized = cv.fit_transform(res_data)

cv_feature_names = cv.get_feature_names()

data_vectorized.shape

# Materialize the sparse data
data_dense = data_vectorized.todense()

# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")

"""### build model"""

from sklearn.decomposition import LatentDirichletAllocation
lda_sklearn_10 = LatentDirichletAllocation(
    n_components=10,               # Number of topics
    max_iter=10,               # Max learning iterations
    learning_method='online',   
    random_state=100,          # Random state
    batch_size=128,            # n docs in each learning iter
    evaluate_every = -1,       # compute perplexity every n iters, default: Don't
    n_jobs = -1,               # Use all available CPUs
)
lda_sklearn_10.fit(data_vectorized)

# saving sklearn model to local

lda_sklearn_10_path = f'{result_path}/lda_sklearn_10.model'
directory = os.path.dirname(lda_sklearn_10_path)
if not os.path.exists(directory):
  os.makedirs(directory)

with open(lda_sklearn_10_path, 'wb') as file:
  pkl.dump(lda_sklearn_10, file)

# load sklearn model 
def load_model_sklearn(model_name):
  lda_sklearn_path = f'{result_path}/{model_name}.model'
  with open(lda_sklearn_path, 'rb') as f:
    model = pkl.load(f)
  return model

lda_sklearn_10 = load_model_sklearn("lda_sklearn_10")

# def get_lda_topics(model, num_topics):
#     word_dict = {};
#     for i in range(num_topics):
#         words = model.show_topic(i, topn = 20)
#         word_dict['Topic #' + '{:02d}'.format(i+1)] = [i[0] for i in words]
#     return pd.DataFrame(word_dict)

# get_lda_topics(lda_model1, 15)

for index, topic in enumerate(lda_sklearn_10.components_):
    #print(f'Top 20 words for Topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-20:]])
    #print('\n')

pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda_sklearn_100, data_vectorized, cv, mds='tsne')
panel



"""## LDA using Gensim

Create the Dictionary and Corpus needed for Topic Modeling

two main inputs to the LDA topic model using GenSim library are the dictionary(id2word) and the corpus.

eg: for the single doc, produce: a mapping of (word_id, word_frequency), (0,1) 1st word in the doc appear once
"""

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])
print(corpus[1:2])

len(corpus)

# id-word mapping:
id2word[4]

# Item-frequency
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]] # cp as sentence, corpus as each doc -> all docs

lda_gensim_10 = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

lda_gensim_20 = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

lda_gensim_100 = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=100, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# saving gensim model to local
import pickle
lda_gensim_10_path = f'{result_path}/lda_gensim_10.model'
directory = os.path.dirname(lda_gensim_10_path)
if not os.path.exists(directory):
  os.makedirs(directory)
lda_gensim_10.save(lda_gensim_10_path)

# loading the save model from local
file_path = f'{result_path}/lda_gensim_100.model'
lda_gensim_100 =  gensim.models.LdaModel.load(file_path)
lda_gensim_100

def get_lda_topics(model, num_topics):
    word_dict = {};
    for i in range(num_topics):
        words = model.show_topic(i, topn = 20)
        #print(f"topic {i}")
        print([i[0] for i in words])
        word_dict['Topic #' + '{:02d}'.format(i)] = [i[0] for i in words]
    return pd.DataFrame(word_dict).T 

# Print the Keyword in the 20 topics
get_lda_topics(lda_gensim_10, 10)
#df.to_csv ('lda_gensim_10.csv', index = False, header=True)

#lda_gensim_10.show_topic(0, 20)

"""Visualize the topics-keywords"""

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_gensim_100, corpus, id2word)
vis



"""## NMF using sklearn"""

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        #print("Topic {}".format(topic_idx))
        print([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(min_df=3, stop_words='english')   #max_df=0.95, 
tfidf = tfidf_vectorizer.fit_transform(res_data)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# Run NMF
#nmf_sklearn_100 = NMF(n_components=100, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

display_topics(nmf_sklearn_10, tfidf_feature_names, 20)

# saving sklearn model to local
import pickle
nmf_sklearn_100_path = f'{result_path}/nmf_sklearn_100.model'
directory = os.path.dirname(nmf_sklearn_100_path)
if not os.path.exists(directory):
  os.makedirs(directory)

with open(nmf_sklearn_100_path, 'wb') as file:
  pkl.dump(nmf_sklearn_100, file)

# load sklearn model 
nmf_sklearn_10_path = f'{result_path}/nmf_sklearn_10.model'
with open(nmf_sklearn_10_path, 'rb') as f:
  nmf_sklearn_10 = pkl.load(f)
nmf_sklearn_10



"""## NMF using gensim"""

nmf_gensim_20 = Nmf(
        corpus=corpus,
        num_topics=20,
        id2word=id2word,
        chunksize=100,
        passes=10,
        kappa=.1,
        minimum_probability=0.01,
        w_max_iter=300,
        w_stop_condition=0.0001,
        h_max_iter=100,
        h_stop_condition=0.001,
        eval_every=10,
        normalize=True,
        random_state=42
    )

# saving gensim model to local
nmf_gensim_100_path = f'{result_path}/nmf_gensim_100.model'
directory = os.path.dirname(nmf_gensim_100_path)
if not os.path.exists(directory):
  os.makedirs(directory)

nmf_gensim_100.save(nmf_gensim_100_path)

# loading the save model from local
def load_model_gensim(model_name):
  file_path = f'{result_path}/{model_name}.model'
  #print(file_path)
  model =  gensim.models.LdaModel.load(file_path)
  return model

def get_lda_topics(model, num_topics):
    word_dict = {};
    for i in range(num_topics):
        words = model.show_topic(i, topn = 20)
        #print(f"topic {i}")
        print([i[0] for i in words])
        word_dict['Topic #' + '{:02d}'.format(i)] = [i[0] for i in words]
    return pd.DataFrame(word_dict)

# Print the Keyword in the 20 topics
get_lda_topics(nmf_gensim_20, 20)

"""# Document-topic distribution

### Using Sklearn

#### doc-topic distribution by year
"""

k = 20
results = {}

for year in pub_dates.unique():
  topic_prob_for_year = []
  for i in pub_dates[pub_dates==year].index:
  
    # vectorize transform 
    mytext_4 = cv.transform([res_data[i]])

    # LDA transform
    topic_probability_scores = lda_sklearn_20.transform(mytext_4)
  
    topic_prob_for_year.append(topic_probability_scores)
  
  topic_prob_for_year=np.mean(np.array(topic_prob_for_year).squeeze(),axis=0)
  #print(f"before sort: {topic_probability_scores}")
  topk = topic_prob_for_year.argsort()[-k:][::-1]
  #print(f"after sort: {topic_probability_scores}")
  print(year, topk, topic_prob_for_year)

  # save to a dict
  results[year] = topic_prob_for_year
results

# Create Document - Topic Matrix
nmf_output = nmf_sklearn_10.transform(tfidf)  ## data_vectorized for lda, tfidf for nmf

# column names
topicnames = ["Topic" + str(i) for i in range(nmf_sklearn_10.n_components)] 

# index names
docnames = [str(i) for i in range(len(res_data))]

# Make the pandas dataframe
df_dominant_topic = pd.DataFrame(np.round(nmf_output, 6), columns=topicnames, index=docnames)   # todo   lda_output
#print(len(np.max(df_document_topic.values, axis=1)))
df_document_topic_copy = df_dominant_topic.copy()
# Get dominant topic for each document

dominant_topic = np.argmax(df_dominant_topic.values, axis=1)
df_dominant_topic['Dominant_Topic'] = dominant_topic
# df_dominant_topic['Pub_Dates'] = pub_dates.to_list()
df_dominant_topic['Topic_Perc_Contrib'] = np.max(df_document_topic_copy.values, axis=1)
df_dominant_topic.head(15)

"""#### TSNE plot for NMF model"""

from sklearn.manifold import TSNE

nmf_embedding = nmf_output
nmf_embedding = (nmf_embedding - nmf_embedding.mean(axis=0))/nmf_embedding.std(axis=0)

tsne = TSNE(random_state=3211)
tsne_embedding = tsne.fit_transform(nmf_embedding)
tsne_embedding = pd.DataFrame(tsne_embedding,columns=['x','y'])
tsne_embedding['hue'] = nmf_embedding.argmax(axis=1)



matplotlib.rc('font',family='monospace')
plt.style.use('ggplot')


fig, axs = plt.subplots(2,2, figsize=(20, 20), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .1, wspace=0)

axs = axs.ravel()

count = 0
legend = []

data = tsne_embedding
scatter = axs[0].scatter(data=data,x='x',y='y',c=data['hue'],cmap="Set1")

fig.legend(legend_list,topics,loc=(0.1,0.89),ncol=3)
plt.subplots_adjust(top=0.85)

plt.show()

df_topic_distribution = df_dominant_topic['Dominant_Topic'].value_counts().reset_index(name="Num Documents")
df_topic_distribution.columns = ['Topic Num', 'Num Documents']
df_topic_distribution

print(df_dominant_topic.Dominant_Topic.unique())
print(df_dominant_topic.Pub_Dates.unique())

#df_dominant_topic[11260:11261]

for topic_no in df_dominant_topic['Dominant_Topic'].unique():
  #topic_no = df_dominant_topic['Dominant_Topic'].unique()
  print(f"current topic is: {topic_no}")
  temp_df = df_dominant_topic.loc[df_dominant_topic['Dominant_Topic']==topic_no]
  #print("#########################")
  temp_df = df_dominant_topic.loc[df_dominant_topic['Dominant_Topic']==topic_no]
  paper_indexs = temp_df['Topic_Perc_Contrib'].nlargest(15).index
  paper_probs = temp_df['Topic_Perc_Contrib'].nlargest(15).values
  # res = temp_df['Topic_Perc_Contrib'].nlargest(15)
  # print(res)
  print(paper_indexs)
  print(paper_probs)

X[6487]







"""### Gensim"""

# load the model from local (dup)
file_path = f'{result_path}/lda_gensim_10.model'
lda_gensim_10 =  gensim.models.LdaModel.load(file_path)
lda_gensim_10

"""#### calculate average dominant topics per year"""

data_lemmatized[0]    # array format for the first doc
corpus[0]             # bow format for the first doc

topic_probs = nmf_gensim_20.get_document_topics(corpus[0],minimum_probability=0)
sum([p for t,p in topic_probs])

# import itertools
# wrong way of calculating
# for year in pub_dates.unique():
#   corpus_by_year = list(itertools.chain(*[corpus[i] for i in pub_dates[pub_dates==year].index]))
#   # optimal_model[corpus_by_year]
#   print(year, optimal_model[corpus_by_year])

k = 20
results = {}

for year in pub_dates.unique():
  #print(f"year!!!!!!!!!!: {year}")
  topic_prob_for_year = []
  for i in pub_dates[pub_dates==year].index:

    # list of topic-probability given a doc
    topic_ps = nmf_gensim_20.get_document_topics(corpus[i],minimum_probability=0)
    topic_prob_for_year.append([p for t,p in topic_ps])
  topic_prob_for_year = np.mean(np.array(topic_prob_for_year),axis=0)

  topk = topic_prob_for_year.argsort()[-k:][::-1]
  print(year, topk, topic_prob_for_year)
  
  # save to a dict
  results[year] = topic_prob_for_year

  # save to local 
  # file_path = f'{root_path}/topic_probs.pkl'
  # directory = os.path.dirname(file_path)
  # if not os.path.exists(directory):
  #   os.makedirs(directory)
  # with open(file_path, 'wb') as file:
  #   pkl.dump(results, file)

graph = pd.DataFrame(results)
columnsTitles = [i for i in range(2000,2021)]
after_graph = graph.reindex(columns=columnsTitles)
after_graph

#after_graph.loc[0]

# build plot
import matplotlib.pyplot as plt
import numpy as np
 
x=np.array(columnsTitles)
 
fig=plt.figure(figsize=(20,10))

ax=fig.add_subplot(111)
 
ax.plot(x,after_graph.loc[0],c='b',marker="^",ls='--',label='1',fillstyle='none')
ax.plot(x,after_graph.loc[1],c='g',marker=(8,2,0),ls='--',label='2')
ax.plot(x,after_graph.loc[2],c='k',ls='-',label='3')
ax.plot(x,after_graph.loc[3],c='r',marker="v",ls='-',label='4')
ax.plot(x,after_graph.loc[4],c='m',marker="o",ls='--',label='5',fillstyle='none')
ax.plot(x,after_graph.loc[5],c='w',marker="+",ls=':',label='6')
ax.plot(x,after_graph.loc[6],c='c',marker="^",ls='--',label='7',fillstyle='none')
ax.plot(x,after_graph.loc[7],c='y',marker=(8,2,0),ls='--',label='8')

ax.plot(x,after_graph.loc[8],c='b',ls='-',label='9')
ax.plot(x,after_graph.loc[9],c='g',marker="v",ls='-',label='10')
ax.plot(x,after_graph.loc[10],c='k',marker="o",ls='--',label='11',fillstyle='none')
ax.plot(x,after_graph.loc[11],c='r',marker="+",ls=':',label='12')
ax.plot(x,after_graph.loc[12],c='m',marker="^",ls='--',label='13',fillstyle='none')
ax.plot(x,after_graph.loc[13],c='w',marker=(8,2,0),ls='--',label='14')
ax.plot(x,after_graph.loc[14],c='c',ls='-',label='15')
ax.plot(x,after_graph.loc[15],c='y',marker="v",ls='-',label='16')

ax.plot(x,after_graph.loc[16],c='b',marker="o",ls='--',label='17',fillstyle='none')
ax.plot(x,after_graph.loc[17],c='g',marker="+",ls=':',label='18')
ax.plot(x,after_graph.loc[18],c='k',marker="^",ls='--',label='19',fillstyle='none')
ax.plot(x,after_graph.loc[19],c='r',marker=(8,2,0),ls='--',label='20')
 
plt.legend(loc=2)
plt.show()

# load local year-topic-distribution file 
file_path = f'{root_path}/topic_probs.pkl'
with open(file_path, 'rb') as f:
    results = pkl.load(f)
results

# given topic number, show key words associated with the topic
wp = optimal_model.show_topic(11)
topic_keywords = ", ".join([word for word, prop in wp])
topic_keywords

"""#### doc-topic distribution"""

topic_probs = nmf_gensim_20.get_document_topics(corpus[0],minimum_probability=0)
sum([p for t,p in topic_probs])

# for each doc, get the topic distribution 
def doc_dominant_topic(ldamodel, texts=data_lemmatized):
  doc_topics_df = pd.DataFrame()
  for d in texts:
      bow = id2word.doc2bow(d)
      list_of_topics = ldamodel.get_document_topics(bow, minimum_probability=0)
      #print(list_of_topics)
      # [(4, 0.76345414), (8, 0.014767735), (10, 0.064257324), (11, 0.021145009), (17, 0.012042195), (18, 0.04192921), (19, 0.020142527)]
      list_of_topics_sorted = sorted(list_of_topics, key=lambda x: (x[1]), reverse=True) 
      #print(list_of_topics_sorted)

      # Get the Dominant topic, Perc Contribution and Keywords for each document
      for j, (topic_num, prop_topic) in enumerate(list_of_topics_sorted):
        if j == 0:  # => dominant topic
          wp = ldamodel.show_topic(topic_num)
          #print(f"wp is {wp}")
          topic_keywords = ", ".join([word for word, prop in wp])
          #print(f"topic_keywords is {topic_keywords}")
          doc_topics_df = doc_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
          #print(f"df is {doc_topics_df}")
        else:
          break
  doc_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
  contents = pd.Series(texts)
  doc_topics_df = pd.concat([doc_topics_df, contents], axis=1)
  doc_topics_df['Pub_Dates'] = pub_dates

  return(doc_topics_df)

doc_topics_df = doc_dominant_topic(nmf_gensim_20, data_lemmatized)
# doc_topics_df

df_dominant_topic = doc_topics_df.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords','Text', 'Pub_Dates']
df_dominant_topic.head(50)
#df_dominant_topic.Dominant_Topic.unique()

# temp = df_dominant_topic.loc[df_dominant_topic['Dominant_Topic'] == 'nan']
# temp

df_dominant_topic['Pub_Dates'].unique()

df_dominant_topic['Dominant_Topic'].unique()

#for topic_no in df_dominant_topic['Dominant_Topic'].unique():
topic_no = 15
#topic_no = df_dominant_topic['Dominant_Topic'].unique()
print(f"current topic is: {topic_no}")
temp_df = df_dominant_topic.loc[df_dominant_topic['Dominant_Topic']==topic_no]
# paper_id = temp_df['Topic_Perc_Contrib'].argmax()
#print("#########################")
temp_df = df_dominant_topic.loc[df_dominant_topic['Dominant_Topic']==topic_no]
paper_indexs = temp_df['Topic_Perc_Contrib'].nlargest(15).index
paper_probs = temp_df['Topic_Perc_Contrib'].nlargest(15).values
# res = temp_df['Topic_Perc_Contrib'].nlargest(15)
# print(res)
print(paper_indexs)
print(paper_probs)

## word wrap
from IPython.display import HTML, display

def set_css():
  display(HTML('''
  <style>
    pre {
        white-space: pre-wrap;
    }
  </style>
  '''))
get_ipython().events.register('pre_run_cell', set_css)

set(abstracts[462].split()).intersection(set(abstracts[658].split()))

abstracts[7915]

# Find the most representative document for each topic
# to help with understanding the topic, you can find the documents a given topic has 
# contributed to the most and infer the topic by reading that document. 

# Group top 5 sentences under each topic
respresentiative_doc = pd.DataFrame()

doc_topics_df_grp = doc_topics_df.groupby('Dominant_Topic')

for i, grp in doc_topics_df_grp:
    # if i < 10:
    #   print(i)
    #   print(grp)
    respresentiative_doc = pd.concat([respresentiative_doc, 
                                      grp.sort_values(['Perc_Contribution'], 
                                                      ascending=False).head(1)], 
                                     axis=0)

# Reset Index    
respresentiative_doc.reset_index(drop=True, inplace=True)

# Format
respresentiative_doc.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text", "pub_date"]

# Show
respresentiative_doc

len(respresentiative_doc['pub_date'])

# Topic distribution across documents
# Number of Documents for Each Topic
# topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
# print(len(topic_counts))
# print(topic_counts)

# # Percentage of Documents for Each Topic
# topic_contribution = round(topic_counts/topic_counts.sum(), 4)
# print(topic_contribution)

# # Topic Number and Keywords
# topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]
# print(topic_num_keywords)
# # Concatenate Column wise
# df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# # Change Column names
# df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# # Show
# df_dominant_topics[:10]





"""# Evluation Metrics: how to find the optimal number of topics

- LDA 
  - cohenrence score based
  - perplexity

- NMF 
  - cohenrence score based
  
- TODO: Bayesian nonparametric topic model 
    - HDP: https://datascience.stackexchange.com/questions/128/latent-dirichlet-allocation-vs-hierarchical-dirichlet-process
      - pro: As far as pros and cons, HDP has the advantage that the maximum number of topics can be unbounded and learned from the data rather than specified in advance.

12 models: coherence score (u_mass) /(npmi) /(w2v) / Perplexity (if any)
  - lda_sklearn_10
  - lda_sklearn_20
  - lda_sklearn_100
  - lda_gensim_10
  - lda_gemsim_20
  - lda_gensim_100
  - nmf_sklearn_10
  - nmf_sklearn_20
  - nmf_sklearn_100
  - nmf_gensim_10
  - nmf_gensim_20
  - nmf_gensim_100
"""

## load models
lda_sklearn_10 = load_model_sklearn("lda_sklearn_10")
lda_sklearn_20 = load_model_sklearn("lda_sklearn_20")
lda_sklearn_100 = load_model_sklearn("lda_sklearn_100")
lda_gensim_10 = load_model_gensim("lda_gensim_10")
lda_gensim_20 = load_model_gensim("lda_gensim_20")
lda_gensim_100 = load_model_gensim("lda_gensim_100")
nmf_sklearn_10 = load_model_sklearn("nmf_sklearn_10")
nmf_sklearn_20 = load_model_sklearn("nmf_sklearn_20")
nmf_sklearn_100 = load_model_sklearn("nmf_sklearn_100")
nmf_gensim_10 = load_model_gensim("nmf_gensim_10")
nmf_gensim_20 = load_model_gensim("nmf_gensim_20")
nmf_gensim_100 = load_model_gensim("nmf_gensim_100")

"""### Coherence 

- https://labs.imaginea.com/how-to-measure-topic-coherence/

#### u_mass / npmi measure
"""

#compute_coherence_score_sklearn(nmf_sklearn_100, tfidf, tfidf_vectorizer)
#compute_coherence_score_gensim(nmf_gensim_100)
data_lemmatized[:2]

def compute_coherence_score_sklearn(model, doc_word_matrix, vectorizer, top_words):
  # 3rd party library
  scores = tmtoolkit.topicmod.evaluate.metric_coherence_gensim(
                        measure='u_mass', 
                        top_n=top_words, 
                        topic_word_distrib=model.components_, 
                        dtm=doc_word_matrix,   #data_vectorized, tfidf
                        vocab=np.array([x for x in vectorizer.vocabulary_.keys()]),   # cv, tfidf_vectorizer
                        #texts = data_lemmatized,
                        #return_coh_model = True,
                        return_mean=False)
  print(scores)
  print(np.mean(scores))
  print(np.median(scores))
  print(np.std(scores))

compute_coherence_score_sklearn(lda_sklearn_10, data_vectorized, cv, 5)
compute_coherence_score_sklearn(lda_sklearn_10, data_vectorized, cv, 10)
compute_coherence_score_sklearn(lda_sklearn_10, data_vectorized, cv, 15)
compute_coherence_score_sklearn(lda_sklearn_10, data_vectorized, cv, 20)

compute_coherence_score_sklearn(lda_sklearn_20, data_vectorized, cv, 5)
compute_coherence_score_sklearn(lda_sklearn_20, data_vectorized, cv, 10)
compute_coherence_score_sklearn(lda_sklearn_20, data_vectorized, cv, 15)
compute_coherence_score_sklearn(lda_sklearn_20, data_vectorized, cv, 20)

compute_coherence_score_sklearn(lda_sklearn_100, data_vectorized, cv, 5)
compute_coherence_score_sklearn(lda_sklearn_100, data_vectorized, cv, 10)
compute_coherence_score_sklearn(lda_sklearn_100, data_vectorized, cv, 15)
compute_coherence_score_sklearn(lda_sklearn_100, data_vectorized, cv, 20)

compute_coherence_score_sklearn(nmf_sklearn_10, tfidf, tfidf_vectorizer, 5)
compute_coherence_score_sklearn(nmf_sklearn_10, tfidf, tfidf_vectorizer, 10)
compute_coherence_score_sklearn(nmf_sklearn_10, tfidf, tfidf_vectorizer, 15)
compute_coherence_score_sklearn(nmf_sklearn_10, tfidf, tfidf_vectorizer, 20)

compute_coherence_score_sklearn(nmf_sklearn_20, tfidf, tfidf_vectorizer, 5)
compute_coherence_score_sklearn(nmf_sklearn_20, tfidf, tfidf_vectorizer, 10)
compute_coherence_score_sklearn(nmf_sklearn_20, tfidf, tfidf_vectorizer, 15)
compute_coherence_score_sklearn(nmf_sklearn_20, tfidf, tfidf_vectorizer, 20)

compute_coherence_score_sklearn(nmf_sklearn_100, tfidf, tfidf_vectorizer, 5)
compute_coherence_score_sklearn(nmf_sklearn_100, tfidf, tfidf_vectorizer, 10)
compute_coherence_score_sklearn(nmf_sklearn_100, tfidf, tfidf_vectorizer, 15)
compute_coherence_score_sklearn(nmf_sklearn_100, tfidf, tfidf_vectorizer, 20)

from gensim.models.coherencemodel import CoherenceModel
# Run the coherence model to get the score
def compute_coherence_score_gensim(model, measure, top_words):
  cm = CoherenceModel(
    model=model,
    texts=data_lemmatized,
    dictionary=id2word,
    topn = top_words,
    coherence=measure       
  )
  topics_scores = cm.get_coherence_per_topic() ##cm_sklearn.get_coherence()
  print(topics_scores)
  print(np.mean(topics_scores))
  print(np.median(topics_scores))
  print(np.std(topics_scores))

compute_coherence_score_gensim(lda_gensim_10, 'u_mass', 5)
compute_coherence_score_gensim(lda_gensim_10, 'u_mass', 10)
compute_coherence_score_gensim(lda_gensim_10, 'u_mass', 15)
compute_coherence_score_gensim(lda_gensim_10, 'u_mass', 20)

compute_coherence_score_gensim(lda_gensim_20, 'u_mass', 5)
compute_coherence_score_gensim(lda_gensim_20, 'u_mass', 10)
compute_coherence_score_gensim(lda_gensim_20, 'u_mass', 15)
compute_coherence_score_gensim(lda_gensim_20, 'u_mass', 20)

compute_coherence_score_gensim(lda_gensim_100, 'u_mass', 5)
compute_coherence_score_gensim(lda_gensim_100, 'u_mass', 10)
compute_coherence_score_gensim(lda_gensim_100, 'u_mass', 15)
compute_coherence_score_gensim(lda_gensim_100, 'u_mass', 20)

compute_coherence_score_gensim(nmf_gensim_10, 'u_mass', 5)
compute_coherence_score_gensim(nmf_gensim_10, 'u_mass', 10)
compute_coherence_score_gensim(nmf_gensim_10, 'u_mass', 15)
compute_coherence_score_gensim(nmf_gensim_10, 'u_mass', 20)

compute_coherence_score_gensim(nmf_gensim_20, 'u_mass', 5)
compute_coherence_score_gensim(nmf_gensim_20, 'u_mass', 10)
compute_coherence_score_gensim(nmf_gensim_20, 'u_mass', 15)
compute_coherence_score_gensim(nmf_gensim_20, 'u_mass', 20)

compute_coherence_score_gensim(nmf_gensim_100, 'u_mass', 5)
compute_coherence_score_gensim(nmf_gensim_100, 'u_mass', 10)
compute_coherence_score_gensim(nmf_gensim_100, 'u_mass', 15)
compute_coherence_score_gensim(nmf_gensim_100, 'u_mass', 20)

"""### npmi measure"""

def compute_coherence_score_sklearn_npmi(model, doc_word_matrix, vectorizer, top_words):
  # 3rd party library
  scores = tmtoolkit.topicmod.evaluate.metric_coherence_gensim(
                        measure='c_npmi', 
                        top_n=top_words, 
                        topic_word_distrib=model.components_, 
                        dtm=doc_word_matrix,   #data_vectorized, tfidf
                        vocab=np.array([x for x in vectorizer.vocabulary_.keys()]),   # cv, tfidf_vectorizer
                        texts = data_lemmatized,
                        #return_coh_model = True,
                        return_mean=False)
  print(scores)
  print(np.mean(scores))
  print(np.median(scores))
  print(np.std(scores))

compute_coherence_score_sklearn_npmi(lda_sklearn_10, data_vectorized, cv, 5)
# compute_coherence_score_sklearn_npmi(lda_sklearn_10, data_vectorized, cv, 10)
# compute_coherence_score_sklearn_npmi(lda_sklearn_10, data_vectorized, cv, 15)
# compute_coherence_score_sklearn_npmi(lda_sklearn_10, data_vectorized, cv, 20)

compute_coherence_score_sklearn_npmi(lda_sklearn_20, data_vectorized, cv, 5)
compute_coherence_score_sklearn_npmi(lda_sklearn_20, data_vectorized, cv, 10)
compute_coherence_score_sklearn_npmi(lda_sklearn_20, data_vectorized, cv, 15)
compute_coherence_score_sklearn_npmi(lda_sklearn_20, data_vectorized, cv, 20)

compute_coherence_score_sklearn_npmi(lda_sklearn_100, data_vectorized, cv, 5)
compute_coherence_score_sklearn_npmi(lda_sklearn_100, data_vectorized, cv, 10)
compute_coherence_score_sklearn_npmi(lda_sklearn_100, data_vectorized, cv, 15)
compute_coherence_score_sklearn_npmi(lda_sklearn_100, data_vectorized, cv, 20)

arr2=[-0.2530815866787999, -0.2110595441997567, -0.2890689182584824, -0.30306173269991094, -0.2712104509445966, 
     -0.3137798523846763, -0.2885466005974111, -0.3018120105544748, -0.29609119367281844, -0.29204684663793457, 
     -0.24706089722083016, -0.263380086935577, -0.3237411162350228, -0.2634226384649042, -0.27932829194020264, -0.23129391824538703, 
     -0.26869873822001183, -0.26374287789447876, -0.34871878686637364, -0.24224141013947814, -0.2915317991842902,
     -0.28616281175267483, -0.2600039251414954, -0.25900994218342455, -0.25239416004275134, -0.32681626339279324, -0.27111804659725064, 
      -0.30855587805438944,  -0.3150056681845446,-0.28506102041975634, -0.2955643837258039, -0.2654188769532181,
     -0.28385051580594206, -0.2949554396542446, -0.28388596808983074,  -0.31534583247184517, -0.28739628361775243,  -0.28324907841393065, 
     -0.23178249671191525, 
    -0.2280316532493157, -0.29493514992511866, -0.26023929954230507, -0.3139282022469753, -0.23481507528695594, -0.2054122723459226, 
     -0.26504114538104817, -0.31048145509512703, -0.26964478422139443, -0.2528264580050723, -0.2631466625503066, -0.2801353734324979, 
     -0.2703332953027257, -0.237877076742323, -0.25638208021230335, -0.28151665989817276, -0.30704861221651136, -0.2878318485701684, 
     -0.2770084083868357, -0.27372599768158073, -0.2606422799537799,  -0.2610912093898454, -0.3164220470495907, 
     -0.19542637405674107, -0.2886124528967663, -0.27440192691299725, -0.22286472518273134, -0.2810653218236627,
     -0.2740676812684537, -0.255682600389814, 
   -0.27297859577218087, -0.33035371782043693,  -0.2500885526699485,  -0.2508552861046433, -0.25916363733407166]
arr3=[-0.2461633108412026, -0.2201565445306994,  -0.2932574910482859, -0.27289322862001103, -0.2820485043217518, 
      -0.29022857677165675, -0.30707974821842426,-0.2995454726751024, -0.2714326381275319, -0.2814876467864898, -0.251116558062967,
      -0.2975192246032651, -0.26692674016687173, -0.28328725846625474, -0.24696186841707543, -0.2740396395831532, -0.2618576069594942, 
      -0.33218209461592835, -0.2572233940570704, -0.2778130876567809, -0.2721186480806758, -0.2679174306812186, 
      -0.27734733206222867, -0.2922576397269703, -0.2718434597882841,  -0.30447194851413045,
      -0.29292851970355294,  -0.28149934965326595, -0.28858670304512607, -0.2869192086969012, -0.2966159748322258, -0.3002539389884608, 
      -0.2713718146406605, -0.2908464511811586, -0.2520175789750032,  -0.22991053478228296, -0.2787505540045008, 
      -0.27115048023579075, -0.30386474650529294, -0.23333981258031192, -0.24978493424075157, -0.27998800361771964, -0.31327380378905123,
      -0.27360127532841, -0.2837546830212887, -0.2589298031240503, -0.2767100982851809, -0.26036507539382975, -0.232972654861659, 
       -0.26193674480360774, -0.3006121193590924, -0.321284899732254, -0.2936538363195475,  -0.2484515662613346,
      -0.2818240803644825, -0.24229529475461797, -0.27974320303179884, -0.27295143553307977, -0.2736211933129593, 
      -0.2816094243823691,  -0.26025003326470625, 
       -0.25161400535499556, -0.3091600233097033,  -0.25201523908758433,  -0.2897626244034403]
arr1 = [-0.30186429747118637, -0.1485383916263771,-0.32801841503277734,  -0.24113528134057233, -0.27577101452904285, 
        -0.32450899404299227, -0.31136287441110894, -0.28347749899581903,-0.3123007345710428, -0.2727223651644334, -0.16563458037327977,
        0.30662749884876694, -0.27365229146855674, -0.2960098038921074, -0.31051266304424613, -0.2319899663932244, -0.2930677634202589, 
        -0.3257631100548939, -0.32162419526123776, -0.20939018509683494, -0.30760355771392456, -0.32080435945443037, 
        -0.30166036331353346,  -0.2460449170083885, -0.24611038816783068, -0.3438053800934066, -0.2852218511429342, 
        -0.2945770142088999,  -0.29790755901882177, -0.23994236143257047, -0.33804179227527076, -0.27540925373605696, 
        -0.2577843105027705, -0.27619717443283515, -0.30579964264984255, -0.31711524956170273, -0.32273327272051033, -0.2479977206385579,
        -0.27787866863535915, -0.2199541751533848, -0.271031413167956, -0.2334131602311435, -0.305537364016055, -0.1967546023997075,
        -0.27855039751886285, -0.25324662536516634, -0.17776651773552285, -0.29041533036681483, -0.28498877541059414, -0.2503358602226719, 
        -0.26991050149245105, -0.29120858428175944,-0.3115596205335537, -0.281057702653581, -0.2444385326882715, -0.22988155253307005, 
        0.2617104506817291, -0.28494575208470846, -0.274110205478648, -0.3078329169975075, -0.19797512378751178, -0.26673377692273664, 
        -0.2932737998626644, -0.21575382754410524, -0.27801387085288587, -0.2728038162583577, -0.22355532490992863, -0.26519795543518493,
        -0.2773452092841894, -0.1678889250323194, -0.3112904354504435, -0.29685310591596326, -0.20126208365275083, -0.2354287437809473, 
        -0.3034380836646452, -0.27760108241623993, -0.30683825778134727, -0.3612865559836406, 
        -0.2843573873380158,-0.2894167495879403, -0.2923823969019646, -0.2362558893752548, -0.28723404895957383]
arr4 = [-0.2664651054517439,  -0.30445857880079075, -0.2591264035836911, -0.2837865662085763, 
        -0.28224139702833256, -0.2944564917038906,  -0.29050194414919034, -0.27728318089800946, -0.301426787318719,
        -0.2424124879596933, -0.2857067704187231, -0.2679078421803395, -0.27757669487055453, -0.2446266250740704, 
        -0.2520810866221406, -0.3162483296647593, -0.27799042852552974, -0.26921780620542024, 
        -0.26517125917961454, -0.287944778010303, -0.27986117347540235,  -0.28966386947948036, 
        -0.295852641981828,  -0.2747749642759466,  -0.29308732793481573, -0.289550911225815, -0.2806217249206625, 
        -0.28406022484799537,-0.2842583314967722, -0.2553788198898853, -0.2521930413194589, -0.28168810234747976,
        -0.29261059872074285, -0.2501125931167952, -0.2557814784827439, -0.2782985633377309, -0.3252935011451687, -0.2893837957186294,
        -0.2718900346359263,  -0.2583826094508884, -0.26316458284467087, -0.29862966772898625, -0.3242999799242275, 
        -0.28548990623716003,  -0.2663133683152567,  -0.28079498228769084, -0.2538287724713355, -0.2800149814373196, 
        -0.26132181334695553, -0.276768257212141,  -0.26208029577233927,
        -0.2569847845911122, -0.3036409503756699, -0.25840272559584476]
arr=[-0.3084419471179255, -0.31670528154147065,-0.29257814449832975, -0.29529824790483855,-0.30422847854351304, -0.28150333276995815, -0.2982417005506343, -0.2844361946715257]

print(np.mean(arr))
print(np.median(arr))
print(np.std(arr))

compute_coherence_score_sklearn_npmi(nmf_sklearn_10, tfidf, tfidf_vectorizer, 5)
compute_coherence_score_sklearn_npmi(nmf_sklearn_10, tfidf, tfidf_vectorizer, 10)
compute_coherence_score_sklearn_npmi(nmf_sklearn_10, tfidf, tfidf_vectorizer, 15)
compute_coherence_score_sklearn_npmi(nmf_sklearn_10, tfidf, tfidf_vectorizer, 20)

compute_coherence_score_sklearn_npmi(nmf_sklearn_20, tfidf, tfidf_vectorizer, 5)
compute_coherence_score_sklearn_npmi(nmf_sklearn_20, tfidf, tfidf_vectorizer, 10)
compute_coherence_score_sklearn_npmi(nmf_sklearn_20, tfidf, tfidf_vectorizer, 15)
compute_coherence_score_sklearn_npmi(nmf_sklearn_20, tfidf, tfidf_vectorizer, 20)

compute_coherence_score_sklearn_npmi(nmf_sklearn_100, tfidf, tfidf_vectorizer, 5)
compute_coherence_score_sklearn_npmi(nmf_sklearn_100, tfidf, tfidf_vectorizer, 10)
compute_coherence_score_sklearn_npmi(nmf_sklearn_100, tfidf, tfidf_vectorizer, 15)
compute_coherence_score_sklearn_npmi(nmf_sklearn_100, tfidf, tfidf_vectorizer, 20)

compute_coherence_score_gensim(lda_gensim_10, 'c_npmi', 5)
compute_coherence_score_gensim(lda_gensim_10, 'c_npmi', 10)
compute_coherence_score_gensim(lda_gensim_10, 'c_npmi', 15)
compute_coherence_score_gensim(lda_gensim_10, 'c_npmi', 20)

compute_coherence_score_gensim(lda_gensim_20, 'c_npmi', 5)
compute_coherence_score_gensim(lda_gensim_20, 'c_npmi', 10)
compute_coherence_score_gensim(lda_gensim_20, 'c_npmi', 15)
compute_coherence_score_gensim(lda_gensim_20, 'c_npmi', 20)

compute_coherence_score_gensim(lda_gensim_100, 'c_npmi', 5)
compute_coherence_score_gensim(lda_gensim_100, 'c_npmi', 10)
compute_coherence_score_gensim(lda_gensim_100, 'c_npmi', 15)
compute_coherence_score_gensim(lda_gensim_100, 'c_npmi', 20)

compute_coherence_score_gensim(nmf_gensim_10, 'c_npmi', 5)
compute_coherence_score_gensim(nmf_gensim_10, 'c_npmi', 10)
compute_coherence_score_gensim(nmf_gensim_10, 'c_npmi', 15)
compute_coherence_score_gensim(nmf_gensim_10, 'c_npmi', 20)

compute_coherence_score_gensim(nmf_gensim_20, 'c_npmi', 5)
compute_coherence_score_gensim(nmf_gensim_20, 'c_npmi', 10)
compute_coherence_score_gensim(nmf_gensim_20, 'c_npmi', 15)
compute_coherence_score_gensim(nmf_gensim_20, 'c_npmi', 20)

compute_coherence_score_gensim(nmf_gensim_100, 'c_npmi', 5)
compute_coherence_score_gensim(nmf_gensim_100, 'c_npmi', 10)
compute_coherence_score_gensim(nmf_gensim_100, 'c_npmi', 15)
compute_coherence_score_gensim(nmf_gensim_100, 'c_npmi', 20)

"""### w2v measure
https://www.sciencedirect.com/science/article/pii/S0957417415001633
"""

import en_core_web_lg
nlp = en_core_web_lg.load()

def compute_w2v_gensim(model, topics, top_n):
  gensim_model = model
  topics_num = topics
  # Create list of topics, each topic is described by the num_top_words words with highest score
  topics_words = []
  for topic_index in range(topics_num):
    # for each cluster of words from a topic, get all 20 words from this cluster
    words = gensim_model.show_topic(topic_index, topn = top_n)
    topics_words.append([i[0] for i in words])

  tc_w2v = compute_TC_W2V(topics_words)
  print("TC_W2V is {}".format(tc_w2v))
  print(np.mean(tc_w2v))
  print(np.median(tc_w2v))
  print(np.std(tc_w2v))

compute_w2v_gensim(lda_gensim_10, 5)
compute_w2v_gensim(lda_gensim_10, 10)
compute_w2v_gensim(lda_gensim_10, 15)
compute_w2v_gensim(lda_gensim_10, 20)

compute_w2v_gensim(lda_gensim_20, 5)
compute_w2v_gensim(lda_gensim_20, 10)
compute_w2v_gensim(lda_gensim_20, 15)
compute_w2v_gensim(lda_gensim_20, 20)

compute_w2v_gensim(lda_gensim_100, 5)
compute_w2v_gensim(lda_gensim_100, 10)
compute_w2v_gensim(lda_gensim_100, 15)
compute_w2v_gensim(lda_gensim_100, 20)

compute_w2v_gensim(nmf_gensim_10, 5)
compute_w2v_gensim(nmf_gensim_10, 10)
compute_w2v_gensim(nmf_gensim_10, 15)
compute_w2v_gensim(nmf_gensim_10, 20)

compute_w2v_gensim(nmf_gensim_20, 5)
compute_w2v_gensim(nmf_gensim_20, 10)
compute_w2v_gensim(nmf_gensim_20, 15)
compute_w2v_gensim(nmf_gensim_20, 20)

compute_w2v_gensim(nmf_gensim_100, 5)
compute_w2v_gensim(nmf_gensim_100, 10)
compute_w2v_gensim(nmf_gensim_100, 15)
compute_w2v_gensim(nmf_gensim_100, 20)

def compute_w2v_sklearn(model, doc_word_matrix, feature_names, top_n):
  sklearn_model = model
  lda_W = sklearn_model.fit_transform(doc_word_matrix) #data_vectorized, tfidf
  lda_H = sklearn_model.components_
  topics = len(lda_H)
  # Create list of topics, each topic is described by the num_top_words words with highest score
  topics_words = []
  for topic_index in range(topics):
      # for each cluster of words from a topic, get all 20 words from this cluster
      topics_words.append(get_words_per_topic(lda_H[topic_index], feature_names, top_n)) # cv_feature_names, tfidf_feature_names

  tc_w2v = compute_TC_W2V(topics_words)
  print("TC_W2V is {}".format(tc_w2v))
  print(np.mean(tc_w2v))
  print(np.median(tc_w2v))
  print(np.std(tc_w2v))

compute_w2v_sklearn(lda_sklearn_10, data_vectorized, cv_feature_names, 5)
#compute_w2v_sklearn(lda_sklearn_10, data_vectorized, cv_feature_names, 10)
#compute_w2v_sklearn(lda_sklearn_10, data_vectorized, cv_feature_names, 15)
compute_w2v_sklearn(lda_sklearn_10, data_vectorized, cv_feature_names, 20)

compute_w2v_sklearn(lda_sklearn_20, data_vectorized, cv_feature_names, 5)
compute_w2v_sklearn(lda_sklearn_20, data_vectorized, cv_feature_names, 10)#?
compute_w2v_sklearn(lda_sklearn_20, data_vectorized, cv_feature_names, 15)
compute_w2v_sklearn(lda_sklearn_20, data_vectorized, cv_feature_names, 20)

#compute_w2v_sklearn(lda_sklearn_100, data_vectorized, cv_feature_names, 5) #?
compute_w2v_sklearn(lda_sklearn_100, data_vectorized, cv_feature_names, 10)
compute_w2v_sklearn(lda_sklearn_100, data_vectorized, cv_feature_names, 15)
compute_w2v_sklearn(lda_sklearn_100, data_vectorized, cv_feature_names, 20)

compute_w2v_sklearn(nmf_sklearn_10, tfidf, tfidf_feature_names, 5)
compute_w2v_sklearn(nmf_sklearn_10, tfidf, tfidf_feature_names, 10)
compute_w2v_sklearn(nmf_sklearn_10, tfidf, tfidf_feature_names, 15)
compute_w2v_sklearn(nmf_sklearn_10, tfidf, tfidf_feature_names, 20)

compute_w2v_sklearn(nmf_sklearn_20, tfidf, tfidf_feature_names, 5)
compute_w2v_sklearn(nmf_sklearn_20, tfidf, tfidf_feature_names, 10)
compute_w2v_sklearn(nmf_sklearn_20, tfidf, tfidf_feature_names, 15)
compute_w2v_sklearn(nmf_sklearn_20, tfidf, tfidf_feature_names, 20)

compute_w2v_sklearn(nmf_sklearn_100, tfidf, tfidf_feature_names, 5)
compute_w2v_sklearn(nmf_sklearn_100, tfidf, tfidf_feature_names, 10)
compute_w2v_sklearn(nmf_sklearn_100, tfidf, tfidf_feature_names, 15)
compute_w2v_sklearn(nmf_sklearn_100, tfidf, tfidf_feature_names, 20)

## online word to vev measure 
nmf_topics = 10   # var
nmf_model = nmf_sklearn_10
nmf_W = nmf_model.fit_transform(tfidf)
nmf_H = nmf_model.components_
# Create list of topics, each topic is described by the num_top_words words with highest score
topics_words = []
for topic_index in range(nmf_topics):
    # for each cluster of words from a topic, get cluster words 
    topics_words.append(get_words_per_topic(nmf_H[topic_index], tfidf_feature_names, 20))

# Compute the coherence for the topics for model with k topics

tc_w2v = compute_TC_W2V(w2v_lookup, topics_words)
print("TC_W2V is {}".format(tc_w2v))

from itertools import combinations

def compute_TC_W2V(topics_words):
    '''
    Compute TC_W2V for the topics of a model using the w2v_lookup.
    TC_W2V is calculated for all possible pairs of words in the topic and 
    then averaged with the mean for that topic.
    The total TC_W2V for the model is the mean over all topics.
    '''
    total_coherence = []
    for topic_index in range(len(topics_words)):
        # Compute coherence per pair of words
        pair_scores = []
        for pair in combinations(topics_words[topic_index], 2):
           
                #pair_scores.append(w2v_lookup.similarity(pair[0], pair[1]))
            pair_0,pair_1 = nlp(pair[0]), nlp(pair[1]) 
            if (pair_1.vector_norm):
                
                pair_scores.append(pair_0.similarity(pair_1))
            else: #except KeyError as e:
                # If word is not in the word2vec model then as score 0.5
                #print(e)
                pair_scores.append(0.5)  
        # get the mean over all pairs in this topic
        topic_score = sum(pair_scores) / len(pair_scores)
        total_coherence.append(topic_score)
    # get the mean score across all topics
    return total_coherence #/ len(topics_words)

def get_words_per_topic(topic_vec, feature_names, num_top_words):
    '''
    Returns a list with the num_top_words with the highest score for the topic given
    '''
    return [feature_names[i] for i in topic_vec.argsort()[:-num_top_words - 1:-1]]

# Train our own word2vec model on the blog posts. The size is the number of dimensions of the embedding space and the min_count is the number of times a word needs to
# appear in the corpus to be considered
#word_gen = WordGenerator(data)
#from gensim.models import Word2Vec
#w2v_model = Word2Vec(common_texts, size=500, window=5, min_count=3, workers=4)
#w2v_model = gensim.models.Word2Vec(data_lemmatized, size=500, min_count=3, sg=1)
#print("The w2v model has been trained on %d terms" % len(w2v_model.wv.vocab))
#w2v_model.save(f"{root_path}/w2v_model.bin")
# w2v_lookup = w2v_model.wv
# w2v_model.save(f"{root_path}/w2v_model.bin")

#w2v_model = gensim.models.Word2Vec.load(f"{root_path}/w2v_model.bin")
#w2v_lookup = w2v_model.wv

# Create this generator to feed words into the Word2Vec model
# class WordGenerator:
#     '''
#     Given a document it tokenises it (split in words) and yields one a at a time.
#     '''
#     def __init__(self, blogs):
#         self.blogs = blogs

#     def __iter__( self ):
#         for blog in self.blogs:
#             sentence_tokens = nlp(str(blog))        
#             tokens = []
#             for tok in sentence_tokens:
#                 if len(tok) >= 2:
#                     tokens.append(tok.text)
#             yield tokens

"""### Perplexity"""

nmf_gensim_10.num_topics

new_topics = nmf_gensim_10.get_document_topics(corpus[0], normalize=True)





#Perplexity is a statistical measure of how well a probability model predicts a sample
#lda_sklearn_10.perplexity(data_vectorized)  
print("Log Likelihood: ", lda_sklearn_10.score(data_vectorized) / data_vectorized.shape[0]) #bigger is better

# Compute Perplexity # a measure of how good the model is. lower the better.
print(lda_sklearn_10.perplexity(data_vectorized))
print(lda_sklearn_20.perplexity(data_vectorized))
print(lda_sklearn_100.perplexity(data_vectorized))

print(np.exp(-1 * lda_gensim_10.log_perplexity(corpus)))
print(np.exp(-1 * lda_gensim_20.log_perplexity(corpus)))
print(np.exp(-1 * lda_gensim_100.log_perplexity(corpus)))

print(nmf_sklearn_10.perplexity(data_vectorized))
print(nmf_sklearn_20.perplexity(data_vectorized))
print(nmf_sklearn_100.perplexity(data_vectorized))

print(np.exp(-1 * nmf_gensim_10.log_perplexity(corpus)))
print(np.exp(-1 * nmf_gensim_20.log_perplexity(corpus)))
print(np.exp(-1 * nmf_gensim_100.log_perplexity(corpus)))



def plot_graph_score(coherence_values):
  # Show graph
  limit=100; start=2; step=6;
  x = range(start, limit, step)
  plt.plot(x, coherence_values)
  plt.xlabel("Num Topics")
  plt.ylabel("Coherence score")
  plt.legend(("coherence_values"), loc='best')
  plt.show()

  # Print the coherence scores
  for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

"""### Using LDA"""

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        #model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=100, step=6)
plot_graph_score(coherence_values)
#plot_score(coherence_values)

root_path

# Select the model and print the topics (Topics = 62)
optimal_model = model_list[3]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))

# # saving model to local
# import pickle
# file_path = f'{root_path}/lda.model'
# directory = os.path.dirname(file_path)
# if not os.path.exists(directory):
#   os.makedirs(directory)
# optimal_model.save(file_path)

optimal_model =  gensim.models.LdaModel.load(file_path)
optimal_model

"""### Using NMF"""

def compute_coherence_values_nmf(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of NMF topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = Nmf(
        corpus=corpus,
        num_topics=num_topics,
        id2word=id2word,
        chunksize=100,
        passes=10,
        kappa=.1,
        minimum_probability=0.01,
        w_max_iter=300,
        w_stop_condition=0.0001,
        h_max_iter=100,
        h_stop_condition=0.001,
        eval_every=10,
        normalize=True,
        random_state=42
    )
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# Can take a long time to run.
model_list_nmf, coherence_values_nmf = compute_coherence_values_nmf(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=100, step=6)
plot_graph_score(coherence_values_nmf)
#plot_score(coherence_values_nmf)

# Select the model and print the topics 
optimal_model = model_list[2]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))

"""### Using HDP"""

# https://radimrehurek.com/gensim/models/hdpmodel.html
# Unlike its finite counterpart, latent Dirichlet allocation, the HDP topic model infers the number of topics from the data.
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models import HdpModel
hdp = HdpModel(corpus, id2word)

# Number of topics for which most probable num_words words will be fetched
topic_info = hdp.print_topics(num_topics=-1, num_words=10)
# topic_info = hdp.print_topics() # default by 20 topics
print(len(topic_info))
print(topic_info[0])
# HDP will calculate as many topics as the assigned truncation level. However, it may be the case that many of these topics have 
# basically zero probability of occurring.

# a function that performs a rough estimate of the topics' probability weights(alpha values) associated with each topic. 
# Note that this is a rough metric only: it does not account for the probability associated with each word. Even so, 
# it provides a pretty good metric for which topics are meaningful and which aren't:

# def topic_prob_extractor(gensim_hdp):
#     shown_topics = gensim_hdp.show_topics(num_topics=-1, formatted=False)
#     topics_nos = [x[0] for x in shown_topics ]
#     weights = [ sum([item[1] for item in shown_topics[topicN][1]]) for topicN in topics_nos ]

#     return pd.DataFrame({'topic_id' : topics_nos, 'weight' : weights})

def topic_prob_extractor(gensim_hdp, t=-1, w=25, isSorted=True):
    """
    Input the gensim model to get the rough topics' probabilities
    """
    shown_topics = gensim_hdp.show_topics(num_topics=t, num_words=w ,formatted=False)
    topics_nos = [x[0] for x in shown_topics ]
    weights = [ sum([item[1] for item in shown_topics[topicN][1]]) for topicN in topics_nos ]
    if (isSorted):
        return pd.DataFrame({'topic_id' : topics_nos, 'weight' : weights}).sort_values(by = "weight", ascending=False);
    else:
        return pd.DataFrame({'topic_id' : topics_nos, 'weight' : weights});


topic_prob_extractor(hdp)

#alpha = hdp.hdp_to_lda()[0];


if __name__ == "__main__":

    #main()