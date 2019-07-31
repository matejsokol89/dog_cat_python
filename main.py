import csv
import glob
import os
import librosa
import numpy as np


if __name__ == '__main__':

    def extract_feature(file_name):
        X, sample_rate = librosa.load(file_name)
        stft = np.abs(librosa.stft(X))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                                  sr=sample_rate).T, axis=0)
        return mfccs, chroma, mel, contrast, tonnetz


    def parse_audio_files(parent_dir, sub_dirs, file_ext="*.wav"):
        full_list = []
        features, labels = np.empty((0, 193)), np.empty(0)
        for label, sub_dir in enumerate(sub_dirs):

            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                varim= fn.split('/')[2]
                # print(varim)
                try:
                    mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
                except Exception as e:
                    print("Error encountered while parsing file: ", fn)
                    continue
                ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
                features = np.vstack([features, ext_features])
                labels = np.append(labels, fn.split('/')[2])
                # print(var)
                # print(features)
                new_dict={varim:ext_features}
                print(new_dict)
                full_list.append(new_dict)
                # value = np.array(features, dtype=np.int), np.array(labels, dtype=np.int)
        with open('dog_cat.csv', 'w') as f:
            wr = csv.writer(f)
            wr.writerow(full_list)
        return features,labels


    def one_hot_encode(labels):
        n_labels = len(labels)
        n_unique_labels = len(np.unique(labels))
        one_hot_encode = np.zeros((n_labels, n_unique_labels))
        # one_hot_encode[np.arange(n_labels), labels] = 1
        return one_hot_encode


    parent_dir = 'cats_dogs'
    tr_sub_dirs = ["fold1"]
    file_ext1 = "*.wav"

    tr_features, tr_labels = parse_audio_files(parent_dir, tr_sub_dirs)

    tr_labels = one_hot_encode(tr_labels)

