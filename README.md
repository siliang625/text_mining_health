# text_mining_health

## Files and Folders Information
`search_term_1.csv`: key search terms provided by public health domain experts

`data` folder: data used for training

`result` folder: all model resutls generated from data_pipeline.py

`experiment` folder: all notebooks experiemnts for EDA

## Saved Model Artifacts
Some of the generated results were saved here as a pickle file, to check the saved resutls, simply load the pkl file in the scripts:

`pub_datas.pkl`: the publication year for all papers

`metadata.pkl`: the metadata(reference count, pmc_id, etc) of all papers

`abstracts.pkl`: the abstract of all papers

`data_lematized.pkl`: the processed text of all papers

## Requirement
`pip install BeautifulSoup4`

`pip install tqdm`

`pip install nltk`

`pip install spacy`

`pip install gensim`

`pip install pyLDAvis`

`pip install -U tmtoolkit`

`python -m spacy download en_core_web_lg`

## Scirpts and How To:
To download the data: `python data_loader.py`

To preprocess the data and build models: `python data_pipeline.py`

To find the best model basec on computaitonal metrics: `python metrics.py`

To generate analysis based on the best model: `python analysis.py`
