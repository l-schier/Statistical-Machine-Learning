from make_dataset import process_dataset, process_dataset_noisy, process_dataset_noisy_norm_pca
from random_forest import grid_search_rf, evaluate_rf, save_model
from svm import grid_search_svm, evaluate_svm

path = "data/Socialmedia_Bot_Prediction_Set1.csv"
target = "is_bot"

X_train, X_test, y_train, y_test, features = process_dataset(
    path, target)

print("Train svm with original data")
best_svm_orig = grid_search_svm(X_train, y_train)
evaluate_svm(best_svm_orig, X_test, y_test, X_train, "original")
save_model(best_svm_orig, "model/svm.pkl")

print("Train rf with original data")
best_rf_orig = grid_search_rf(X_train, y_train)
evaluate_rf(best_rf_orig, X_test, y_test, X_train, "original")
save_model(best_rf_orig, "model/random_forest.pkl")


X_train, X_test, y_train, y_test, noisy_features, features = process_dataset_noisy_norm_pca(
    path, target)

print("Train svm with noisy data and pca")
best_norm_pca_svm = grid_search_svm(X_train, y_train)
evaluate_svm(best_norm_pca_svm, X_test, y_test, X_train, "norm_pca")
save_model(best_norm_pca_svm, "model/svm_norm_pca.pkl")

print("Train rf with noisy data and pca")
best_norm_pca_rf = grid_search_rf(X_train, y_train)
evaluate_rf(best_norm_pca_rf, X_test, y_test, X_train, "norm_pca")
save_model(best_norm_pca_rf, "model/random_forest_norm_pca.pkl")

X_train, X_test, y_train, y_test, noisy_features, features = process_dataset_noisy(
    path, target)

print("Train svm with noisy data")
best_noisy_svm = grid_search_svm(X_train, y_train)
save_model(best_noisy_svm, "model/svm_noisy.pkl")
evaluate_svm(best_noisy_svm, X_test, y_test, X_train, "noisy")

print("Train rf with noisy data")
best_noisy_rf = grid_search_rf(X_train, y_train)
evaluate_rf(best_noisy_rf, X_test, y_test, X_train, "noisy")
save_model(best_noisy_rf, "model/random_forest_noisy.pkl")

print("Done!")
