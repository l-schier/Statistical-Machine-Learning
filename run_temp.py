from make_dataset import process_dataset_noisy_norm_pca
from random_forest import grid_search_rf, evaluate_rf, save_model, load_model
from svm import grid_search_svm, evaluate_svm

path = "data/Socialmedia_Bot_Prediction_Set1.csv"
target = "is_bot"

X_train, X_test, y_train, y_test, noisy_features, features = process_dataset_noisy_norm_pca(
    path, target)

best_rf = load_model("model/random_forest_norm_pca.pkl")
evaluate_rf(best_rf, X_test, y_test, X_train, "TEMP_norm_pca")

best_svm = load_model("model/svm_norm_pca.pkl")
evaluate_svm(best_svm, X_test, y_test, X_train, "TEMP_norm_pca")
