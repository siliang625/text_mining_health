'''
How to compare and pick the best model based on computational metrics
'''

def load_model_sklearn(model_name):
  # load sklearn model 
  lda_sklearn_path = f'{result_path}/{model_name}.model'
  with open(lda_sklearn_path, 'rb') as f:
    model = pkl.load(f)
  return model

# loading the save model from local
def load_model_gensim(model_name):
  file_path = f'{result_path}/{model_name}.model'
  #print(file_path)
  model =  gensim.models.LdaModel.load(file_path)
  return model
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



def compute_coherence_score_gensim(model, measure, top_words):
  from gensim.models.coherencemodel import CoherenceModel
  # Run the coherence model to get the score
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

  def compute_TC_W2V(topics_words):
    from itertools import combinations
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

### PERPELEXITY

"""### Perplexity"""



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

if __name__ == "__main__":
    abstracts, pub_dates = load_from_drive(root_path, "abstracts"), load_from_drive(root_path, "pub_dates")
    
    # umass
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

    compute_coherence_score_sklearn_npmi(lda_sklearn_10, data_vectorized, cv, 5)
    compute_coherence_score_sklearn_npmi(lda_sklearn_10, data_vectorized, cv, 10)
    compute_coherence_score_sklearn_npmi(lda_sklearn_10, data_vectorized, cv, 15)
    compute_coherence_score_sklearn_npmi(lda_sklearn_10, data_vectorized, cv, 20)

    compute_coherence_score_sklearn_npmi(lda_sklearn_20, data_vectorized, cv, 5)
    compute_coherence_score_sklearn_npmi(lda_sklearn_20, data_vectorized, cv, 10)
    compute_coherence_score_sklearn_npmi(lda_sklearn_20, data_vectorized, cv, 15)
    compute_coherence_score_sklearn_npmi(lda_sklearn_20, data_vectorized, cv, 20)

    compute_coherence_score_sklearn_npmi(lda_sklearn_100, data_vectorized, cv, 5)
    compute_coherence_score_sklearn_npmi(lda_sklearn_100, data_vectorized, cv, 10)
    compute_coherence_score_sklearn_npmi(lda_sklearn_100, data_vectorized, cv, 15)
    compute_coherence_score_sklearn_npmi(lda_sklearn_100, data_vectorized, cv, 20)

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


    compute_w2v_sklearn(lda_sklearn_10, data_vectorized, cv_feature_names, 5)
    compute_w2v_sklearn(lda_sklearn_10, data_vectorized, cv_feature_names, 10)
    compute_w2v_sklearn(lda_sklearn_10, data_vectorized, cv_feature_names, 15)
    compute_w2v_sklearn(lda_sklearn_10, data_vectorized, cv_feature_names, 20)

    compute_w2v_sklearn(lda_sklearn_20, data_vectorized, cv_feature_names, 5)
    compute_w2v_sklearn(lda_sklearn_20, data_vectorized, cv_feature_names, 10)
    compute_w2v_sklearn(lda_sklearn_20, data_vectorized, cv_feature_names, 15)
    compute_w2v_sklearn(lda_sklearn_20, data_vectorized, cv_feature_names, 20)

    compute_w2v_sklearn(lda_sklearn_100, data_vectorized, cv_feature_names, 5) 
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
    #main()