"""
## Problem Statment: 
download xml of all related paper from pmc

## requried libraries:
!pip install BeautifulSoup4
!pip install fastprogress  #tqdm

"""
## import libraries
# %matplotlib inline
import pandas as pd
import requests
from urllib.request import urlopen
from fastprogress.fastprogress import master_bar, progress_bar
from urllib.parse import quote
from bs4 import BeautifulSoup
import os
import time
from importlib import reload  # Not needed in Python 2
import logging
#reload(logging)
#logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logging.basicConfig(level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger("Parsing_Data")
# logger = logging.getLogger("Parsing_Data")
# logger.setLevel(logging.DEBUG)
from concurrent.futures import ThreadPoolExecutor
import threading
import os.path
from os import path
import json


# global variables
base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
search_url = "esearch.fcgi?db=pubmed&term="
fetch_url = "efetch.fcgi?db=pmc"
application = "&tool=digital_public_health"
email = "&email=siliang.liu@alumni.ubc.ca"
base_url_converter = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
root_path = "."
data_path = root_path + "/data/"


def load_key_words():
    """# key words:
    Loading in key words provided by Alice and Muyi 
    
    Return: 2 dataframe: column1 (list of key words) and column2 (list of key words)
    """ 
    path = "./search_term_1.csv"
    search_terms_df = pd.read_csv(path, header=None, keep_default_na=False)
    search_terms_c1 = search_terms_df.iloc[:,0]
    search_terms_c2 = search_terms_df.iloc[:,1]
    return search_terms_c1, search_terms_c2

"""Define search query
Let's define the query: we will use OR operator to connect all terms in column 0, as well as column 1, 
and use AND operator to connect subquery of column 0 and column 1. 
"""
def form_query_term(df):
    query_string = "(“mhealth” OR “m-health“ OR “ehealth“ OR “e-health“ OR “virtual health“ OR “mobile health“ OR “online health“ OR “internet-based health“ OR “computer-based health“ OR “health informatics“ OR “social media“ OR “predictive algorithms“ OR “artificial intelligence“ OR “machine learning methods“ OR “big data“ OR “electronic health“ OR “telemedicine“ OR “digit*” OR “web*”) AND (“public health“ OR “health promotion“ OR “health prevention“ OR “health+protection“ OR “health policy” OR “health determinants“ OR “health evaluation“ OR “health economics“ OR “public health ethics“ OR “risk assessment“ OR “epidemiology“ OR “community health“ OR “emergency preparedness“ OR “emergency response“ OR “health equity“ OR “social justice“ OR “social determinants“ OR “surveillance“) AND “last 20 years”[dp] AND “english”[la]"
    logger.debug(query_string)
    return query_string 


def extract_ids(query_term, return_size=10):
    '''
    args: query_string 
    return: find and set max_return_size 
            return list of related ids
    '''
    ret_max_term = "&RetMax=" + str(return_size)
    
    base_url_key_word = base_url + search_url + quote(query_term) + ret_max_term #+ application + email

    
    page_kw = urlopen(base_url_key_word)
    soup_kw = BeautifulSoup(page_kw, "xml")
    
    # get text form of everything in the xml file
    # soup_kw.get_text()
    
    return_size = soup_kw.find('Count').text    
    id_list = soup_kw.find_all('Id')
    
    return_ids = []
    for each_id in id_list:
        return_ids.append(each_id.text)
    
    # save to txt file
    save_pmc_list(return_ids, "pmc_en_list.txt")

    return return_size, return_ids


def id_convert(pmid):
    """
    convert pmid to pmc_id
    input; pmid
    return: pmc_id
    """

    api_url = base_url_converter + "?ids=" + pmid + "&format=json"
    response = requests.get(api_url)

    if response.status_code == 200:
        res = json.loads(response.content.decode('utf-8'))
        if 'pmcid' not in res['records'][0]:
            print("this pm_id : {} is not valid in pmc".format(pmid))
            return None    
        else:
            pmc_id = res['records'][0]['pmcid']
            print("Corresponding pmcid is: {}".format(pmc_id))
            return pmc_id

def load_txt_file(file_path, file_name):
    return_ids = []
    with open(f'{file_path}/{file_name}','r') as f:
      line = f.read()
      return_ids.append(line)
    f.close()
    return return_ids

def save_pmc_list(return_ids, file_name):
    f=open(f'{root_path}/{file_name}','w')
    for id in return_ids:
        f.write(id+'\n')
    f.close()
    logger.debug(len(return_ids))


def filter_valid_pmc():
    valid_pmc_ids = []

    for each in return_ids:
        res = id_convert(each)
        if res is not None:
            valid_pmc_ids.append(res)
    print(len(valid_pmc_ids))  

    save_pmc_list(valid_pmc_ids, "pmc_en_valid.txt")
    return valid_pmc_ids


def extract_xml(paper_id):
    sem = threading.Semaphore(3)

    # check if exist
    file = root_path + '/data/{}_paper.xml'.format(paper_id)
    if str(path.isfile(file)) is True:
      return

    query_id = "&id=" + str(paper_id)
    base_url_content = base_url + fetch_url + query_id
    print(base_url_content)
    sem.acquire()
    try:
      response = requests.get(base_url_content)
      with open(root_path + '/data/{}_paper.xml'.format(paper_id), 'wb') as file:
          file.write(response.content)
      file.close()
    finally:
        sem.release()
    
    # get text form of everything in the xml file
    #soup.get_text()
    time.sleep(1)


def num_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def download_xml(pmc_return_ids):
    '''
    download paper into data folder, in xml form
    '''
    start = time.time()
    # with ThreadPoolExecutor(max_workers=num_cpus()//2) as executor:
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(extract_xml, pmc_return_ids)
    #     jobs = (executor.submit(process, chunk) for chunk in df_generator)
    print('Total time taken: {}'.format(time.time() - start))


# def shellquote(s):
#     return "'" + s.replace("'", "'\\''") + "'"
# cur_path = shellquote(cur_path)

def file_count(cur_path):
    ''' 
    check how many files under given directory
    input: directory
    return: the numebr of files 

    # ! ls {cur_path} | wc -l
    '''
    from pathlib import Path
    count = 0
    
    for path in Path(cur_path).glob('*.xml'):
        count += 1
    return count


if __name__ == "__main__":
    ## get key search terms 
    col1, col2 = load_key_words()
    ## hard coded the search query
    query_string = form_query_term()

    ## get a list of related papers' pmid
    return_size = extract_ids(query_string)[0]
    logger.debug("there are {} related papers".format(return_size))
    pm_ids = extract_ids(query_string, return_size)[1]

    ## convert pm_id to pmc_ids, and save to pmc_en_valid.txt
    pmc_ids = filter_valid_pmc(pm_ids)

    ## download data
    download_xml(pmc_ids)  

    # some utility functions
    num_cpus()
    file_count(data_path)
    #main()
