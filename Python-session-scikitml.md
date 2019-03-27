
# Sample `scikit-multilearn` work session


```python
import skmultilearn.cluster as cluster
from skmultilearn.dataset import available_data_sets
from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.adapt import MLkNN
from sklearn.svm import SVC
import sklearn.metrics as metrics
```

## See the available datasets in the scikit-multilearn repository


```python
for x in available_data_sets().keys():
    print(x)
```

    ('bibtex', 'undivided')
    ('bibtex', 'test')
    ('bibtex', 'train')
    ('birds', 'undivided')
    ('birds', 'test')
    ('birds', 'train')
    ('Corel5k', 'undivided')
    ('Corel5k', 'test')
    ('Corel5k', 'train')
    ('delicious', 'undivided')
    ('delicious', 'test')
    ('delicious', 'train')
    ('emotions', 'undivided')
    ('emotions', 'test')
    ('emotions', 'train')
    ('enron', 'undivided')
    ('enron', 'test')
    ('enron', 'train')
    ('genbase', 'undivided')
    ('genbase', 'test')
    ('genbase', 'train')
    ('mediamill', 'undivided')
    ('mediamill', 'test')
    ('mediamill', 'train')
    ('medical', 'undivided')
    ('medical', 'test')
    ('medical', 'train')
    ('rcv1subset1', 'undivided')
    ('rcv1subset1', 'test')
    ('rcv1subset1', 'train')
    ('rcv1subset2', 'undivided')
    ('rcv1subset2', 'test')
    ('rcv1subset2', 'train')
    ('rcv1subset3', 'undivided')
    ('rcv1subset3', 'test')
    ('rcv1subset3', 'train')
    ('rcv1subset4', 'undivided')
    ('rcv1subset4', 'test')
    ('rcv1subset4', 'train')
    ('rcv1subset5', 'undivided')
    ('rcv1subset5', 'test')
    ('rcv1subset5', 'train')
    ('scene', 'undivided')
    ('scene', 'test')
    ('scene', 'train')
    ('tmc2007_500', 'undivided')
    ('tmc2007_500', 'test')
    ('tmc2007_500', 'train')
    ('yeast', 'undivided')
    ('yeast', 'test')
    ('yeast', 'train')
    

## Load and explore a dataset


```python
emotions_X, emotions_Y, attributes, labels = load_dataset('emotions', 'undivided')
labels
```

    emotions:undivided - exists, not redownloading
    




    [('amazed-suprised', ['0', '1']),
     ('happy-pleased', ['0', '1']),
     ('relaxing-calm', ['0', '1']),
     ('quiet-still', ['0', '1']),
     ('sad-lonely', ['0', '1']),
     ('angry-aggresive', ['0', '1'])]




```python
concurrence_builder = cluster.LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
concurrence = concurrence_builder.transform(emotions_Y)
print(concurrence)
```

    {(1, 2): 91.0, (0, 5): 92.0, (1, 5): 12.0, (0, 1): 56.0, (2, 3): 104.0, (2, 4): 95.0, (3, 4): 105.0, (4, 5): 20.0, (2, 5): 7.0, (0, 4): 10.0, (1, 4): 1.0, (0, 2): 13.0, (1, 3): 7.0, (3, 5): 2.0}
    

## Load train and test partitions to fit two classifiers


```python
emotions_X_train, emotions_Y_train, _, _ = load_dataset('emotions', 'train')
emotions_X_test,  emotions_Y_test, _, _  = load_dataset('emotions', 'test')
```

    emotions:train - exists, not redownloading
    emotions:test - exists, not redownloading
    


```python
class1 = BinaryRelevance(classifier=SVC(gamma="auto"))
class1.fit(emotions_X_train, emotions_Y_train)
```




    BinaryRelevance(classifier=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
            require_dense=[True, True])




```python
class2 = MLkNN()
class2.fit(emotions_X_train, emotions_Y_train)
```




    MLkNN(ignore_first_neighbours=0, k=10, s=1.0)




```python
prediction = class1.predict(emotions_X_test)
print('Hamming loss: ', metrics.hamming_loss(emotions_Y_test, prediction))
print('Accuracy: ', metrics.accuracy_score(emotions_Y_test, prediction))
```

    Hamming loss:  0.26485148514851486
    Accuracy:  0.14356435643564355
    


```python
prediction = class2.predict(emotions_X_test)
print('Hamming loss: ', metrics.hamming_loss(emotions_Y_test, prediction))
print('Accuracy: ', metrics.accuracy_score(emotions_Y_test, prediction))
```

    Hamming loss:  0.30363036303630364
    Accuracy:  0.13366336633663367
    
