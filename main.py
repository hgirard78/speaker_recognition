import librosa
import glob
import random
import pickle
from mpire import WorkerPool

NB_SPEAKERS = 251

def average(list):
    return [sum(l)/len(l) for l in list]

def get_features(file):
    y, sr = librosa.load(file, mono=True, duration=15)
    feature = librosa.feature.mfcc(y, sr, n_mfcc=100)
    return average(feature)

def get_files(nb_speakers):
    all_files = []
    id_list = []
    for i in range(nb_speakers):
        print(f"Get: {i + 1}/{nb_speakers}")
        files = glob.glob(f'train-dataset/{i}/*.flac')
        all_files += files
        id_list += [i for x in range (len(files))]
    return all_files, id_list

def extract_train():
    all_files, id_list = get_files(nb_speakers=NB_SPEAKERS)        
    features = []
    if __name__ == '__main__':
        with WorkerPool(n_jobs=12) as pool:
            features = pool.map(get_features, all_files, progress_bar=True)
    if features == []:
        print("No features")
        exit(2)
    temp = list(zip(features, id_list))
    random.shuffle(temp)
    features, id_list = zip(*temp)
    pickle.dump((features, id_list), open('dataset.p', 'wb'))

def get_features_test(file):
    y, sr = librosa.load(file, mono=True, duration=15)
    feature = librosa.feature.mfcc(y, sr, n_mfcc=100)
    return average(feature) 

def extract_test():
    test_files = []
    category = []
    for i in range(4034):
        test_files.append(f'test-dataset-comp/{i}.flac')
        category.append(i)
    if __name__ == '__main__':
        with WorkerPool(n_jobs=12) as pool:
            features = pool.map(get_features_test, test_files, progress_bar=True)
        pickle.dump((features, category), open('dataset-test.p', 'wb'))

extract_test()
extract_train()
