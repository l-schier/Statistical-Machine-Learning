import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from make_dataset import process_dataset_noisy_norm_pca
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
from pickle import dump, load


def grid_search_rf(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)

    rf.fit(X_train, y_train)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    print("Starting Grid Search with parameters: ", param_grid)
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=3, n_jobs=15, verbose=2)

    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)

    best_rf = grid_search.best_estimator_
    return best_rf


def best_rf(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)

    best_params = {
        'n_estimators': 300,
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt'
    }

    best_params_noisy = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt'
    }
    print("Using best parameters: ", best_params)
    rf.set_params(**best_params)
    rf.fit(X_train, y_train)
    print("Model trained!")
    return rf


def evaluate_rf(rf, X_test, y_test, X_train, name=""):
    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)

    print(classification_report(y_test, y_pred))
    feature_importances = rf.feature_importances_
    features = X_train.columns
    importance_df = pd.DataFrame(
        {'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.savefig("visualization/random_forest_"+name+"_feature_importance.png")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
                'Not Robot', 'Robot'], yticklabels=['Not Robot', 'Robot'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig("visualization/random_forest_"+name+"_confusion_matrix.png")

    y_prob = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig("visualization/random_forest_"+name+"_roc_curve.png")


def save_model(model, path="model/random_forest.pkl"):
    with open(path, 'wb') as file:
        dump(model, file, protocol=5)


def load_model(path="model/random_forest.pkl"):
    with open(path, 'rb') as file:
        return load(file)


if __name__ == "__main__":
    path = "data/Socialmedia_Bot_Prediction_Set1.csv"
    target = "is_bot"

    X_train, X_test, y_train, y_test, noisy_features, features = process_dataset_noisy_norm_pca(
        path, target)
    # best_rf = grid_search_rf(X_train, y_train)
    best_rf = best_rf(X_train, y_train)
    evaluate_rf(best_rf, X_test, y_test, X_train)
    print("Done!")
