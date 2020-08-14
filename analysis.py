


def load_from_drive(path, file_name):
  file_path = f'{path}/{file_name}.pkl'
  with open(file_path, 'rb') as f:
      return pkl.load(f)




if __name__ == "__main__":
    abstracts, pub_dates = load_from_drive(root_path, "abstracts"), load_from_drive(root_path, "pub_dates")
    #main()