

```python
from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
import sklearn.metrics as metrics
```


```python
X_train, y_train, feature_names, label_names = load_dataset('emotions', 'train')
X_test, y_test, _, _ = load_dataset('emotions', 'test')
```

    emotions:train - does not exists downloading
    Downloaded emotions-train
    emotions:test - does not exists downloading
    Downloaded emotions-test
    


```python
clf = BinaryRelevance(
    classifier=SVC(gamma="auto"),
    require_dense=[False, True]
)
clf.fit(X_train, y_train)
```




    BinaryRelevance(classifier=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
            require_dense=[False, True])




```python
prediction = clf.predict(X_test)
```


```python
metrics.hamming_loss(y_test, prediction)
```




    0.26485148514851486




```python
metrics.accuracy_score(y_test, prediction)
```




    0.14356435643564355




```python

```
