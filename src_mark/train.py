import joblib
import time
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from threading import Thread

def display_timer():
    start_time = time.time()
    while True:
        elapsed_time = time.time() - start_time
        print(f"\rProcessing... {int(elapsed_time)} seconds elapsed", end="")
        time.sleep(1)

def train_model_with_grid_search(X_train, y_train):
    # param_grid = {
    #     'C': [0.1, 1, 10],
    #     'kernel': ['linear', 'poly', 'rbf']
    # }

    param_grid = {
        'C': [10],
        'kernel': ['rbf'],
        'gamma': [0.01, 0.1]
    }

    model = svm.SVC(kernel='rbf', gamma='scale', C=10, max_iter=10000, probability=True)
    # grid_search = GridSearchCV(model, param_grid, cv=3, verbose=3)
    model.fit(X_train, y_train)
    return model

def save_best_models(model):
    # best_params = grid_search.best_params_
    # best_model_name = f'svm_best_C{best_params["C"]}_{best_params["kernel"]}_{best_params["gamma"]}.joblib'
    # joblib.dump(grid_search.best_estimator_, f'src_mark/model-mark')
    joblib.dump(model, "src_mark/model-anthony.joblib")

if __name__ == '__main__':
    X_train, y_train = joblib.load('src_mark/training_data.joblib')

    print("Starting GridSearchCV...")
    timer_thread = Thread(target=display_timer)
    timer_thread.daemon = True
    timer_thread.start()

    grid_search = train_model_with_grid_search(X_train, y_train)
    print("\nGridSearchCV completed.")

    print("Saving models...")
    save_best_models(grid_search)
    print("Models saved successfully.")