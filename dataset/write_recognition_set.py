import glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def get_df_list():
    list_filenames = ["Excited_01", "Angry_3", "Curious_01", "Surprised_01", "Interested_2",
                      "Disappointed_1", "Bored_01", "Frustrated_1", "Happy_01", "Puzzled_1"]
    files = sorted(glob.iglob('./train/*.csv', recursive=True))
    list_df = []
    list_df_size = []
    list_df_name = []
    for f in files:
        label = f.split("/")[-1].split(".csv")[0]
        if(label in list_filenames):
            df = pd.read_csv(f, header=None)
            list_df.append(df)
            list_df_size.append(len(df))
            list_df_name.append(f.split("/")[-1].split(".csv")[0])

    return list_df, list_df_size, list_df_name


def pca_lf_noise(df, n_components=5, freq=0.01, bound_scaling_factor=0.1, bound_shift=0.1):
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(df)

    timesteps = pca_data.shape[0]
    smooth_noise = np.sin(np.linspace(0, 2 * np.pi * freq * timesteps, timesteps)) * 0.1
    noisy_pca_data = pca_data.copy()
    for i in range(noisy_pca_data.shape[1]):
        noisy_pca_data[:, i] += smooth_noise

    scaling_factor = np.random.uniform(-bound_scaling_factor, bound_scaling_factor)
    shift = np.random.uniform(-bound_shift, bound_shift)

    noisy_pca_data = noisy_pca_data * (1 + scaling_factor) + shift
    new_data = pca.inverse_transform(noisy_pca_data)
    df_new = pd.DataFrame(new_data, columns=df.columns)

    return df_new


if __name__ == "__main__":

    list_df, list_df_size, list_df_name = get_df_list()
    for df, df_name in zip(list_df, list_df_name):
        n_components = 5
        freq = 0.01
        scaling_factor = 0.1
        bound_scaling_factor = scaling_factor
        bound_shift = scaling_factor
        df_new = pca_lf_noise(df, n_components, freq, bound_scaling_factor, bound_shift)
        filename = "./recognition/" + df_name + "_noisy.csv"
        df_new.to_csv(filename, index=False, header=None)
        print(filename)
