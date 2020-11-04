'''
Processing raw text data
Build models and save model artifacts

'''

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

# global variable 
root_path = "."
data_path = root_path + "/data"
result_path = root_path + "/result"

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


"""# Text Preprocessinng"""

def data_cleaning(data):
# Remove section heading
# some cases: section heading inside the p tag, eg: pmc_id = 7320979
  data = [re.sub('[A-Za-z]*:', '', each) for each in X]
# Remove new line characters
  data = [re.sub('\s+', ' ', each) for each in data]
  
  return data


"""### tokenization"""
def tokenize_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en', disable=['parser', 'ner'])

# Define functions for stopwords, bigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

"""### Co-location using genism library this time"""
def make_quadgrams(texts):
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
  return [quad_mod[trigram_mod[bigram_mod[doc]]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
      doc = nlp(" ".join(sent)) 
      texts_out.append([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags])
    return texts_out

"""### Remove Stopwords, Make quadgram and Lemmatize"""
def remove_stop_words(data_words):
  nltk.download('stopwords')
  nltk.download('punkt')
  stop_words = stopwords.words('english')
  #stop_words.extend(['from', 'subject', 're', 'edu', 'use']) # TODO: can add more

  # Remove Stop Words
  data_words_nostops = remove_stopwords(data_words)
  return data_words_nostops

"""# Build Models

- model title follow pattern: **{model}_{library}_{number_of_topics}**, eg: lda_sklearn_10
- 10, 20, 100 topics

## LDA using Sklearn
## Transform tokenized words into correct format for Countvectorizer
"""
def sklearn_data_preprocess(data_lemmatized):
# transform data_lemmatized to a format where lda use (list of strings, each doc is a string)
  res_data = []
  for doc in data_lemmatized:
    res_data.append(" ".join(doc))
  return res_data

def sklearn_lda_vectorizer(res_data):
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
  from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
  cv = CountVectorizer(analyzer='word', min_df = 3,    # max_df_ as default, # max_df=0.80, max_df=0.80, max_features=10000
                      token_pattern='[a-zA-Z0-9]{3,}', lowercase=True,  
                      stop_words = 'english')
  # now using consistent data pre-processing results 
  data_vectorized = cv.fit_transform(res_data)
  cv_feature_names = cv.get_feature_names()

  return cv, data_vectorized, cv_feature_names

def check_sparsicity(data_vector):

  # Materialize the sparse data
  data_dense = data_vectorized.todense()

  # Compute Sparsicity = Percentage of Non-Zero cells
  print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")

"""### build model"""
def generate_sklearn_lda(num_topics):
  from sklearn.decomposition import LatentDirichletAllocation
  model = LatentDirichletAllocation(
      n_components=num_topics,               # Number of topics
      max_iter=10,               # Max learning iterations
      learning_method='online',   
      random_state=100,          # Random state
      batch_size=128,            # n docs in each learning iter
      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
      n_jobs = -1,               # Use all available CPUs
  )
  model.fit(data_vectorized)
  save_sklearn_model(model, f"lda_sklearn_{num_topics}")
  return model
  
def save_sklearn_model(model, name):
  # saving sklearn model to local
  model_path = f'{result_path}/{name}.model'
  directory = os.path.dirname(model_path)
  if not os.path.exists(directory):
    os.makedirs(directory)

  with open(model_path, 'wb') as file:
    pkl.dump(model, file)


def load_model_sklearn(model_name):
  # load sklearn model 
  lda_sklearn_path = f'{result_path}/{model_name}.model'
  with open(lda_sklearn_path, 'rb') as f:
    model = pkl.load(f)
  return model

#lda_sklearn_10 = load_model_sklearn("lda_sklearn_10")

def get_lda_topics(model):
  for index, topic in enumerate(model.components_):
    #print(f'Top 20 words for Topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-20:]])
    #print('\n')

def plot_sklearn_model(model, data_vectorized):
  pyLDAvis.enable_notebook()
  panel = pyLDAvis.sklearn.prepare(model, data_vectorized, cv, mds='tsne')
  panel



"""## LDA using Gensim

Create the Dictionary and Corpus needed for Topic Modeling

two main inputs to the LDA topic model using GenSim library are the dictionary(id2word) and the corpus.

eg: for the single doc, produce: a mapping of (word_id, word_frequency), (0,1) 1st word in the doc appear once
"""
def gensim_data_preprocess(data_lemmatized):
  # Create Dictionary
  id2word = corpora.Dictionary(data_lemmatized)
  # Create Corpus
  texts = data_lemmatized
  # Term Document Frequency
  corpus = [id2word.doc2bow(text) for text in texts]

  # View
  # print(corpus[:1])
  # print(corpus[1:2])

  # id-word mapping:
  # id2word[4]

  # Item-frequency
  # [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]] # cp as sentence, corpus as each doc -> all docs
  return id2word, corpus

def generate_gensim_lda(num_topics):
  model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

  save_gensim_model(model, f"lda_gensim_{num_topics}")                            
  return model

def save_gensim_model(mode, name):
  # saving gensim model to local
  import pickle
  model_path = f'{result_path}/{name}.model'
  directory = os.path.dirname(model_path)
  if not os.path.exists(directory):
    os.makedirs(directory)
  model.save(model_path)


# loading the save model from local
def load_model_gensim(model_name):
  file_path = f'{result_path}/{model_name}.model'
  #print(file_path)
  model =  gensim.models.LdaModel.load(file_path)
  return model


def get_lda_topics(model, num_topics):
    word_dict = {}
    for i in range(num_topics):
        words = model.show_topic(i, topn = 20)
        #print(f"topic {i}")
        print([i[0] for i in words])
        word_dict['Topic #' + '{:02d}'.format(i)] = [i[0] for i in words]
    return pd.DataFrame(word_dict).T 


"""Visualize the topics-keywords"""
def plot_sklearn_model(model, corpus, id2word):
  pyLDAvis.enable_notebook()
  vis = pyLDAvis.gensim.prepare(model, corpus, id2word)
  vis


"""## NMF using sklearn"""
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        #print("Topic {}".format(topic_idx))
        print([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])

def sklearn_nmf_vectorizer(res_data):
  from sklearn.decomposition import NMF
  from sklearn.feature_extraction.text import TfidfVectorizer
  # NMF is able to use tf-idf
  tfidf_vectorizer = TfidfVectorizer(min_df=3, stop_words='english')   #max_df=0.95, 
  tfidf = tfidf_vectorizer.fit_transform(res_data)
  tfidf_feature_names = tfidf_vectorizer.get_feature_names()

  return tfidf_vectorizer, tfidf, tfidf_feature_names

# Run NMF
def generate_sklearn_lda(num_topics):
  model = NMF(n_components=num_topics, 
                      andom_state=1, alpha=.1, 
                      l1_ratio=.5, init='nndsvd').fit(tfidf)
  save_sklearn_model(model, f"nmf_sklearn_{num_topics}")                   
  return model
    


## ## NMF using gensim
def generate_gensim_nmf(num_topics):
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
  save_gensim_model(mode, f"nmf_gensim_{num_topics}")
  return model



"""# Document-topic distribution

### Using Sklearn
"""
#### doc-topic distribution by year
def hot_topic_by_year_sklearn():
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
    return results



"""#### TSNE plot for NMF model"""
def plot_nmf(nmf_output):
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



def doc_dominant_topic_sklearn(model, data_vector):
  # Create Document - Topic Matrix
  nmf_output = model.transform(data_vector)  ## data_vectorized for lda, tfidf for nmf

  # column names
  topicnames = ["Topic" + str(i) for i in range(model_topics.n_components)] 

  # index names
  docnames = [str(i) for i in range(len(res_data))]

  # Make the pandas dataframe
  df_dominant_topic = pd.DataFrame(np.round(nmf_output, 6), columns=topicnames, index=docnames)   # todo   lda_output
  df_document_topic_copy = df_dominant_topic.copy()
  
  # Get dominant topic for each document
  dominant_topic = np.argmax(df_dominant_topic.values, axis=1)
  df_dominant_topic['Dominant_Topic'] = dominant_topic
  # df_dominant_topic['Pub_Dates'] = pub_dates.to_list()
  df_dominant_topic['Topic_Perc_Contrib'] = np.max(df_document_topic_copy.values, axis=1)
  df_dominant_topic.head(15)

  df_topic_distribution = df_dominant_topic['Dominant_Topic'].value_counts().reset_index(name="Num Documents")
  df_topic_distribution.columns = ['Topic Num', 'Num Documents']
  

  print(df_dominant_topic.Dominant_Topic.unique())
  print(df_dominant_topic.Pub_Dates.unique())

  ### prinnt info for each topic
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

  return df_topic_distribution



def hot_topic_by_year_gensim():
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


def plot_hot_topic(result):
  graph = pd.DataFrame(results)
  columnsTitles = [i for i in range(2000,2021)]
  after_graph = graph.reindex(columns=columnsTitles)
  after_graph

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


"""#### doc-topic distribution"""
# topic_probs = nmf_gensim_20.get_document_topics(corpus[0],minimum_probability=0)
# sum([p for t,p in topic_probs])

# for each doc, get the topic distribution 
def doc_dominant_topic_gensim(model, texts=data_lemmatized):
  doc_topics_df = pd.DataFrame()
  for d in texts:
      bow = id2word.doc2bow(d)
      list_of_topics = model.get_document_topics(bow, minimum_probability=0)
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

  #return(doc_topics_df)


# doc_topics_df

  df_dominant_topic = doc_topics_df.reset_index()
  df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords','Text', 'Pub_Dates']
  df_dominant_topic.head(50)
  #df_dominant_topic.Dominant_Topic.unique()

  # temp = df_dominant_topic.loc[df_dominant_topic['Dominant_Topic'] == 'nan']
  # temp

  df_dominant_topic['Pub_Dates'].unique()

  df_dominant_topic['Dominant_Topic'].unique()

  for topic_no in df_dominant_topic['Dominant_Topic'].unique():
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

  return df_dominant_topic


if __name__ == "__main__":
    # Traning dataset
    abstracts, pub_dates, metadata = get_sample_data(data_path)
    save_to_drive("abstracts", abstracts); save_to_drive("pub_dates", pub_dates); save_to_drive("metadata", metadata)
    metadata.to_csv(f'{result_path}/metadata.csv')
    X = abstracts

    # Text Preprocessinng
    ## clearning
    data = data_cleaning(X)
    ## tokenization
    data_words = list(tokenize_to_words(data))
    data_words_nostops = remove_stop_words(data_words)
    ## Form Quadgrams
    data_words_bigrams = make_quadgrams(data_words_nostops)
    ## Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    save_to_drive("data_lemmatized", data_lemmatized)


    # Build Model
    ## sklearn lda 
    res_data = sklearn_data_preprocess(data_lemmatized)
    cv, data_vectorized, cv_feature_names = sklearn_lda_vectorizer(res_data)
    lda_sklearn_model = generate_sklearn_lda(10)  # 20, 100
    get_lda_topics(lda_sklearn_model)  


    ## gensim lda
    id2word, corpus = gensim_data_preprocess(data_lemmatized)
    lda_gensim_model = generate_gensim_lda(10)  # 20, 100
    get_lda_topics(lda_gensim_model)  #print


    ## sklearn nmf
    tfidf_vectorizer, tfidf, tfidf_feature_names = sklearn_nmf_vectorizer(res_data)
    nmf_sklearn_model = generate_sklearn_lda(10)  # 20, 100
    display_topics(nmf_sklearn_model, tfidf_feature_names, 20) # 10 100
    plot_nmf(nmf_output)

    ## gensim nmf
    nmf_gensim_model = generate_gensim_nmf(20) #10, 100
    get_lda_topics(nmf_gensim_20, 20)
    plot_nmf(nmf_output)


    # doc-topic distribution
    # data_vectorized for lda, tfidf for nmf skleanr
    doc_topics_df = doc_dominant_topic_sklearn(lda_sklearn_model, data_vectorized)
    doc_topics_df = doc_dominant_topic_sklearn(nmf_sklearn_model, tfidf)
    # gensim 
    doc_topics_df = doc_dominant_topic_gensim(gensim_model, data_lemmatized)
