abstracts, pub_dates, metadata = get_sample_data(data_path)
#len(abstracts)

save_to_drive("abstracts", abstracts); save_to_drive("pub_dates", pub_dates); save_to_drive("metadata", metadata)



metadata.to_csv(f'{result_path}/metadata.csv')

metadata.shape

X = abstracts

len(abstracts), len(pub_dates)

pub_dates.unique()
