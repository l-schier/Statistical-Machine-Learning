import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from make_dataset import process_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

def grid_search_svm(X_train, y_train):
    svm = SVC(probability=True, random_state=42)

    svm.fit(X_train, y_train)


    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }
    print("Starting Grid Search with parameters: ", param_grid)
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=3, n_jobs=15, verbose=2)

    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)

    best_svm = grid_search.best_estimator_
    return best_svm

def best_svm(X_train, y_train):
    svm = SVC(probability=True, random_state=42)

    best_params = {
        'C': 10,
        'gamma': 1,
        'kernel': 'rbf'
    }
    print("Using best parameters: ", best_params)
    svm.set_params(**best_params)
    svm.fit(X_train, y_train)
    print("Model trained!")
    return svm


def evaluate(svm, X_test, y_test, X_train):

    y_pred = best_svm.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Robot', 'Robot'], yticklabels=['Not Robot', 'Robot'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig("visualization/svm_confusion_matrix.png")

    y_prob = best_svm.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig("visualization/svm_roc_curve.png")


if __name__ == "__main__":
    path = "data/Socialmedia_Bot_Prediction_Set1.csv"
    target = "is_bot"

    X_train, X_test, y_train, y_test, noisy_features, features = process_dataset(
        path, target)
    #best_svm = grid_search_svm(X_train, y_train)
    best_svm = best_svm(X_train, y_train)
    evaluate(best_svm, X_test, y_test, X_train)
    print("Done!")