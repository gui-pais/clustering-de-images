import pickle

def save(data,save_name):
    try:
        with open(f'{save_name}.pkl', 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        raise ValueError(f"{e}")

def load(file_name):
    try:
        with open(f'{file_name}.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise ValueError(f"{e}")