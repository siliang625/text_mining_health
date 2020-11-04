import matplotlib.pyplot as plt
import numpy as np


def hot_topic_by_year(model, k, corpus):
    '''
    hot topic by year using for lda gensim model 

    model: lda gensim model 
    k: number of topics from the model 
    corpus: gensim lda coorpus

    return: a dictionary {year: list of topic with probabrility by decensing order }

    '''
    results = {}

    for year in pub_dates.unique():
        topic_prob_for_year = []
        for i in pub_dates[pub_dates==year].index:

        # list of topic-probability given a doc
            topic_ps = model.get_document_topics(corpus[i],minimum_probability=0)
            topic_prob_for_year.append([p for t,p in topic_ps])
    topic_prob_for_year = np.mean(np.array(topic_prob_for_year),axis=0)

    topk = topic_prob_for_year.argsort()[-k:][::-1]
    print(year, topk, topic_prob_for_year)
    
    # save to a dict
    results[year] = topic_prob_for_year
    return results
    

def plot_hot_topic(result):
    graph = pd.DataFrame(results)
    columnsTitles = [i for i in range(2000,2021)]
    after_graph = graph.reindex(columns=columnsTitles)

    # build plot
    x=np.array(columnsTitles)
    
    fig=plt.figure(figsize=(20,10))

    ax=fig.add_subplot(111)
    
    ax.plot(x,after_graph.loc[0],c='b',marker="^",ls='--',label='1',fillstyle='none')
    ax.plot(x,after_graph.loc[1],c='g',marker=(8,2,0),ls='--',label='2')
    ax.plot(x,after_graph.loc[2],c='k',ls='-',label='3')
    ax.plot(x,after_graph.loc[3],c='r',marker="v",ls='-',label='4')
    ax.plot(x,after_graph.loc[4],c='m',marker="o",ls='--',label='5',fillstyle='none')
    ax.plot(x,after_graph.loc[5],c='b',marker="+",ls=':',label='6')
    ax.plot(x,after_graph.loc[6],c='c',marker="^",ls='--',label='7',fillstyle='none')
    ax.plot(x,after_graph.loc[7],c='y',marker=(8,2,0),ls='--',label='8')
    ax.plot(x,after_graph.loc[8],c='b',ls='-',label='9')
    ax.plot(x,after_graph.loc[9],c='g',marker="v",ls='-',label='10')
    ax.plot(x,after_graph.loc[10],c='k',marker="o",ls='--',label='11',fillstyle='none')
    ax.plot(x,after_graph.loc[11],c='r',marker="+",ls=':',label='12')
    ax.plot(x,after_graph.loc[12],c='m',marker="^",ls='--',label='13',fillstyle='none')
    ax.plot(x,after_graph.loc[13],c='g',marker=(8,2,0),ls='--',label='14')
    ax.plot(x,after_graph.loc[14],c='c',ls='-',label='15')
    ax.plot(x,after_graph.loc[15],c='y',marker="v",ls='-',label='16')
    ax.plot(x,after_graph.loc[16],c='b',marker="o",ls='--',label='17',fillstyle='none')
    ax.plot(x,after_graph.loc[17],c='g',marker="+",ls=':',label='18')
    ax.plot(x,after_graph.loc[18],c='k',marker="^",ls='--',label='19',fillstyle='none')
    ax.plot(x,after_graph.loc[19],c='r',marker=(8,2,0),ls='--',label='20')
    
    plt.legend(loc=2)
    plt.show()



if __name__ == "__main__":
    abstracts, pub_dates = load_from_drive(root_path, "abstracts"), load_from_drive(root_path, "pub_dates")
    
    ## TODO: LOAD argumnets
    results = hot_topic_by_year(nmf_sklearn_20, 20, corpus)
    plot_hot_topic(results)
    #main()