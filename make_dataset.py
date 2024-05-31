import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore, skew, kurtosis
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA


def extract_features(df):
    features = {}

    features['mean'] = df.mean()
    features['median'] = df.median()

    features['variance'] = df.var()
    features['std_dev'] = df.std()
    features['range'] = df.max() - df.min()

    features['skewness'] = df.apply(skew)
    features['kurtosis'] = df.apply(kurtosis)

    features['corr_matrix'] = df.corr()

    features['rolling_mean'] = df.rolling(window=3, min_periods=1).mean()

    return features


def load_and_process(path):
    df = pd.read_csv(path)  # 'src/data/Socialmedia_Bot_Prediction_Set1.csv'
    pd.option_context('display.max_rows', 5, 'display.max_columns', None)

    print("Missing values detected: ")
    print(df.isnull().sum())
    df.dropna(inplace=True)
    print("Missing values removed")

    print("Duplicate values detected: ")
    print(df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    print("Duplicate values removed")

    outliers_checks = ['length_name', 'follower_follow_rate', 'ave_comment', 'ave_repost',
                       'ave_attitudes', 'source_num', 'post_rate', 'ave_1', 'cvar_1',
                       'ave_2', 'cvar_2', 'ave_url', 'cvar_url', 'cvar_textlength',
                       'pun_var', 'pun_cvar', 'word_ave', 'word_cvar', 'ave_emotionnum',
                       'cvar_pic_num', 'ave_time', 'cvar_time', 'ave_max_time',
                       'ave_min_time', 'time_arg_1', 'time_arg_2', 'info_complete', 'urank']

    z = np.abs(zscore(df[outliers_checks]))
    outliers = df[(z > 3).any(axis=1)]
    print("Outliers detected: ")
    print(outliers)
    df = df[(z <= 3).all(axis=1)]
    print("Outliers removed")

    print("Changing data types")
    df['length_name'] = df['length_name'].astype('int')
    df['default_name'] = df['default_name'].astype('int')
    df['info_complete'] = df['info_complete'].astype('int')
    df['urank'] = df['urank'].astype('int')
    df['icon'] = df['icon'].astype('int')
    df['is_bot'] = df['is_bot'].astype('int')
    #plot_initial_features(df)
    return df


def add_noise_and_normalize(df):
    numerical_features = ['length_name', 'follower_follow_rate', 'ave_comment', 'ave_repost',
                          'ave_attitudes', 'source_num', 'post_rate', 'ave_1', 'cvar_1',
                          'ave_2', 'cvar_2', 'ave_url', 'cvar_url', 'cvar_textlength',
                          'pun_var', 'pun_cvar', 'word_ave', 'word_cvar', 'ave_emotionnum',
                          'cvar_pic_num', 'ave_time', 'cvar_time', 'ave_max_time',
                          'ave_min_time', 'time_arg_1', 'time_arg_2', 'default_name', 'info_complete', 'urank', 'icon']
    # categorical_features = ['default_name', 'info_complete', 'urank', 'icon']
    target = 'is_bot'

    print("Adding noise to the data")
    noise_factor = 0.05
    noisy_df = df.copy()
    for feature in numerical_features:
        noise = np.random.normal(0, noise_factor, size=(len(noisy_df),))
        noisy_df[feature] += noise
    numerical_features.append(target)
    
    #plot_noisy_features(noisy_df)

    print("Normalizing the data")
    scaler = MinMaxScaler()
    df_noisy_norm = pd.DataFrame(scaler.fit_transform(
        noisy_df[numerical_features]), columns=numerical_features)

    #plot_standard_features(df_noisy_norm)
    return df_noisy_norm


def correlation_matrix(df, name):
    print("Correlation Matrix")
    corr_matrix = df.corr(numeric_only=False)
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig("visualization/"+name+"_correlation_matrix.png")


def balancing(df, target):
    X = df.drop(columns=[target], axis=1)
    y = df[target]

    print("Balancing the data with SMOTE")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    #plot_balanced_features(X_resampled)
    return X_resampled, y_resampled


def perform_pca(X_resampled, y_resampled, target):
    print("Perform PCA")

    pca = PCA()
    X_pca = pca.fit_transform(X_resampled)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Number of components to explain 95% of variance: {num_components}")

    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(X_resampled)

    df_pca = pd.DataFrame(data=X_pca, columns=[
        f'PC{i+1}' for i in range(num_components)])
    df_pca[target] = y_resampled.values
    #plot_pca_features(df_pca)
    return df_pca, pca


def plot_pca_variance(pca, name):
    print("Explained Variance by Principal Components")
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance Explained')
    plt.title('Explained Variance by Principal Components')
    plt.grid(True)
    plt.savefig("visualization/"+name+"_exp_var_principal_components.png")


def split_data(df, target):
    print("Splitting the data")
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target, axis=1), df[target], test_size=0.3, random_state=42, stratify=df[target])
    return X_train, X_test, y_train, y_test


def process_dataset(path, target):
    df = load_and_process(path)
    features = extract_features(df)
    correlation_matrix(df, "original")
    X_train, X_test, y_train, y_test = split_data(df, target)
    print("Data processing completed")
    return X_train, X_test, y_train, y_test, features

def process_dataset_noisy(path, target):
    df = load_and_process(path)
    features = extract_features(df)
    correlation_matrix(df, "noisy_norm")
    df_noisy_norm = add_noise_and_normalize(df)
    noisy_features = extract_features(df_noisy_norm)
    X_resampled, y_resampled = balancing(df_noisy_norm, target)
    X_resampled[target] = y_resampled
    X_train, X_test, y_train, y_test = split_data(X_resampled, target)
    print("Data processing completed")
    return X_train, X_test, y_train, y_test, noisy_features, features


def process_dataset_noisy_norm_pca(path, target):
    df = load_and_process(path)
    features = extract_features(df)
    correlation_matrix(df, "norm_pca")
    df_noisy_norm = add_noise_and_normalize(df)
    noisy_features = extract_features(df_noisy_norm)
    X_resampled, y_resampled = balancing(df_noisy_norm, target)
    df_pca, pca = perform_pca(X_resampled, y_resampled, target)
    plot_pca_variance(pca, "norm_pca")
    X_train, X_test, y_train, y_test = split_data(df_pca, target)
    print("Data processing completed")
    return X_train, X_test, y_train, y_test, noisy_features, features


def plot_initial_features(df):
    numerical_features = ['length_name', 'follower_follow_rate', 'ave_comment', 'ave_repost',
                          'ave_attitudes', 'source_num', 'post_rate', 'ave_1', 'cvar_1',
                          'ave_2', 'cvar_2', 'ave_url', 'cvar_url', 'cvar_textlength',
                          'pun_var', 'pun_cvar', 'word_ave', 'word_cvar', 'ave_emotionnum',
                          'cvar_pic_num', 'ave_time', 'cvar_time', 'ave_max_time',
                          'ave_min_time', 'time_arg_1', 'time_arg_2']
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(numerical_features):
        plt.subplot(6, 5, i + 1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig("visualization/preprocessing/initial_features.png")

def plot_noisy_features(df):
    numerical_features = ['length_name', 'follower_follow_rate', 'ave_comment', 'ave_repost',
                          'ave_attitudes', 'source_num', 'post_rate', 'ave_1', 'cvar_1',
                          'ave_2', 'cvar_2', 'ave_url', 'cvar_url', 'cvar_textlength',
                          'pun_var', 'pun_cvar', 'word_ave', 'word_cvar', 'ave_emotionnum',
                          'cvar_pic_num', 'ave_time', 'cvar_time', 'ave_max_time',
                          'ave_min_time', 'time_arg_1', 'time_arg_2']
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(numerical_features):
        plt.subplot(6, 5, i + 1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature} (Noisy)')
    plt.tight_layout()
    plt.savefig("visualization/preprocessing/noisy_features.png")

def plot_standard_features(df):
    numerical_features = ['length_name', 'follower_follow_rate', 'ave_comment', 'ave_repost',
                          'ave_attitudes', 'source_num', 'post_rate', 'ave_1', 'cvar_1',
                          'ave_2', 'cvar_2', 'ave_url', 'cvar_url', 'cvar_textlength',
                          'pun_var', 'pun_cvar', 'word_ave', 'word_cvar', 'ave_emotionnum',
                          'cvar_pic_num', 'ave_time', 'cvar_time', 'ave_max_time',
                          'ave_min_time', 'time_arg_1', 'time_arg_2']
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(numerical_features):
        plt.subplot(6, 5, i + 1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature} (Standardized)')
    plt.tight_layout()
    plt.savefig("visualization/preprocessing/standardized_features.png")

def plot_pca_features(df):
    num_components = df.shape[1] - 1
    plt.figure(figsize=(20, 15))
    for i in range(num_components):
        plt.subplot(4, 5, i + 1)
        sns.histplot(df[f'PC{i+1}'], kde=True)
        plt.title(f'Distribution of PC{i+1}')
    plt.tight_layout()
    plt.savefig("visualization/preprocessing/pca_features.png")

def plot_balanced_features(df):
    numerical_features = ['length_name', 'follower_follow_rate', 'ave_comment', 'ave_repost',
                          'ave_attitudes', 'source_num', 'post_rate', 'ave_1', 'cvar_1',
                          'ave_2', 'cvar_2', 'ave_url', 'cvar_url', 'cvar_textlength',
                          'pun_var', 'pun_cvar', 'word_ave', 'word_cvar', 'ave_emotionnum',
                          'cvar_pic_num', 'ave_time', 'cvar_time', 'ave_max_time',
                          'ave_min_time', 'time_arg_1', 'time_arg_2']
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(numerical_features):
        plt.subplot(6, 5, i + 1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature} (Balanced)')
    plt.tight_layout()
    plt.savefig("visualization/preprocessing/balanced_features.png")

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, noisy_features, features = process_dataset_noisy_norm_pca('data/Socialmedia_Bot_Prediction_Set1.csv', 'is_bot')
    print(X_train.head())
    print("Data processing completed")
