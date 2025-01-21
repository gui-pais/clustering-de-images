import pickle

def save_data(data, filename):
    try:
        with open(f'{filename}.pkl', 'wb') as file:
            pickle.dump(data, file)
    except Exception as e:
        raise ValueError(f"Error saving data: {e}")

def load_data(filename):
    try:
        with open(f'{filename}.pkl', 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")