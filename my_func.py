import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def kfold_split(num_objects, num_folds):
    fold_size = num_objects // num_folds
    flag = num_objects % num_folds
    ans = list()
    for i in range(num_folds):
        x = np.arange(num_objects)
        mask1 = (x >= (i + 1) * fold_size) | (x < i * fold_size)
        mask2 = (x < (i + 1) * fold_size) & (x >= i * fold_size)
        if flag and i == num_folds - 1:
            ans.append((x[x < i * fold_size], x[x >= i * fold_size]))
            break
        ans.append((x[mask1], x[mask2]))
    return ans


def cv_score(X, y, parameters, folds, knn_class):
    ans = []
    scaler = TfidfVectorizer(max_df=0.8, min_df=10, stop_words='english')
    for sc in parameters['scores']:
        ans.append({})
        for n in parameters['n_neighbors']:
            for w in parameters['weights']:
                neigh = knn_class(n_neighbors=n, metric='cosine', weights=w[1])
                i = parameters['scores'].index(sc)
                ans[i][(n, w[0])] = 0
                for f in folds:
                    scaler.fit(X[f[0]])
                    X_train = scaler.transform(X[f[0]])
                    X_test = scaler.transform(X[f[1]])
                    neigh.fit(X_train, y[f[0]])
                    ans[i][(n, w[0])] += sc(y[f[1]], neigh.predict(X_test))
                ans[i][(n, w[0])] /= len(folds)
    return ans