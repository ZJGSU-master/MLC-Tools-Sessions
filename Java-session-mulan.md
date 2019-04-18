

```Java
import mulan.classifier.*;
import mulan.data.*;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.lazy.MLkNN;
import mulan.evaluation.*;
import mulan.evaluation.measure.*;
import weka.classifiers.trees.J48;
import weka.core.Utils
```

# Load and explore the emotions dataset


```Java
var emotions = new MultiLabelInstances("emotions.arff", "emotions.xml");
var statistics = new Statistics();
statistics.calculateStats(emotions);
statistics
```




    Examples: 593
    Predictors: 72
    --Nominal: 0
    --Numeric: 72
    Labels: 6
    
    Cardinality: 1.8684654300168635
    Density: 0.3114109050028106
    Distinct Labelsets: 27
    
    Percentage of examples with label 1: 0.2917369308600337
    Percentage of examples with label 2: 0.2799325463743676
    Percentage of examples with label 3: 0.4451939291736931
    Percentage of examples with label 4: 0.24957841483979765
    Percentage of examples with label 5: 0.28330522765598654
    Percentage of examples with label 6: 0.31871838111298484
    
    Examples of cardinality 0: 0.0
    Examples of cardinality 1: 178.0
    Examples of cardinality 2: 315.0
    Examples of cardinality 3: 100.0
    Examples of cardinality 4: 0.0
    Examples of cardinality 5: 0.0
    Examples of cardinality 6: 0.0
    
    Examples of combination 000111: 1
    Examples of combination 000011: 12
    Examples of combination 000110: 37
    Examples of combination 001001: 3
    Examples of combination 001100: 30
    Examples of combination 001101: 1
    Examples of combination 110000: 38
    Examples of combination 100000: 24
    Examples of combination 110001: 7
    Examples of combination 010010: 1
    Examples of combination 011000: 74
    Examples of combination 011100: 6
    Examples of combination 111000: 11
    Examples of combination 001010: 25
    Examples of combination 001110: 67
    Examples of combination 001011: 3
    Examples of combination 000100: 5
    Examples of combination 000001: 72
    Examples of combination 100010: 6
    Examples of combination 101000: 2
    Examples of combination 100011: 4
    Examples of combination 010000: 23
    Examples of combination 010001: 5
    Examples of combination 010100: 1
    Examples of combination 100001: 81
    Examples of combination 001000: 42
    Examples of combination 000010: 12
    



# Partition the dataset


```Java
var partitioner = new IterativeStratification();
var folds = partitioner.stratify(emotions, 5);
statistics.calculateStats(folds[1]);
statistics
```




    Examples: 116
    Predictors: 72
    --Nominal: 0
    --Numeric: 72
    Labels: 6
    
    Cardinality: 1.9137931034482758
    Density: 0.3189655172413793
    Distinct Labelsets: 21
    
    Percentage of examples with label 1: 0.3017241379310345
    Percentage of examples with label 2: 0.28448275862068967
    Percentage of examples with label 3: 0.45689655172413796
    Percentage of examples with label 4: 0.25862068965517243
    Percentage of examples with label 5: 0.28448275862068967
    Percentage of examples with label 6: 0.3275862068965517
    
    Examples of cardinality 0: 0.0
    Examples of cardinality 1: 32.0
    Examples of cardinality 2: 62.0
    Examples of cardinality 3: 22.0
    Examples of cardinality 4: 0.0
    Examples of cardinality 5: 0.0
    Examples of cardinality 6: 0.0
    
    Examples of combination 000001: 14
    Examples of combination 000110: 8
    Examples of combination 000011: 1
    Examples of combination 001100: 7
    Examples of combination 100010: 1
    Examples of combination 101000: 1
    Examples of combination 110000: 7
    Examples of combination 100011: 1
    Examples of combination 110001: 2
    Examples of combination 100000: 3
    Examples of combination 010000: 6
    Examples of combination 011000: 11
    Examples of combination 011100: 2
    Examples of combination 010001: 2
    Examples of combination 100001: 17
    Examples of combination 111000: 3
    Examples of combination 001010: 7
    Examples of combination 001110: 13
    Examples of combination 001011: 1
    Examples of combination 001000: 8
    Examples of combination 000010: 1
    



# Load training and testing partitions


```Java
// Load the train and test partitions from ARFF files
var train = new MultiLabelInstances("emotions-train.arff", "emotions.xml");
var test = new MultiLabelInstances("emotions-test.arff", "emotions.xml")
```


```Java
// Some data exploration
System.out.println("Instances: " + train.getNumInstances() + ", Labels: " + train.getNumLabels() + ", Card: " + train.getCardinality())
```

    Instances: 391, Labels: 6, Card: 1.813299232736573
    


```Java
// A transformation-based classifier and an adapted classifier
var class1 = new ClassifierChain(new J48());
var class2 = new MLkNN(5, 1.0);
```


```Java
// Train both classifiers
class1.build(train);
class2.build(train)
```


```Java
// Define the evaluation metrics to obtain
var evaluator = new Evaluator();
List<Measure> metrics = Arrays.asList(new HammingLoss(), new ExampleBasedAccuracy(), new ExampleBasedFMeasure());
```


```Java
// Evaluate the ClassifierChain classifier
evaluator.evaluate(class1, test, metrics)
```




    Hamming Loss: 0,2896
    Example-Based Accuracy: 0,4277
    Example-Based F Measure: 0,5145
    




```Java
// Evaluate the MLkNN classifier
evaluator.evaluate(class2, test, metrics)
```




    Hamming Loss: 0,2120
    Example-Based Accuracy: 0,5165
    Example-Based F Measure: 0,5959
    




```Java
evaluator.evaluate(class2, train, test)
```




    Hamming Loss: 0,1479
    Subset Accuracy: 0,4092
    Example-Based Precision: 0,7592
    Example-Based Recall: 0,6790
    Example-Based F Measure: 0,6934
    Example-Based Accuracy: 0,6240
    Example-Based Specificity: 0,9318
    Micro-averaged Precision: 0,8037
    Micro-averaged Recall: 0,6756
    Micro-averaged F-Measure: 0,7341
    Micro-averaged Specificity: 0,9285
    Macro-averaged Precision: 0,8124
    amazed-suprised: 0,7222 happy-pleased: 0,7162 relaxing-calm: 0,8152 quiet-still: 0,8659 sad-lonely: 0,9143 angry-aggresive: 0,8407 
    Macro-averaged Recall: 0,6506
    amazed-suprised: 0,6555 happy-pleased: 0,4953 relaxing-calm: 0,8929 quiet-still: 0,7978 sad-lonely: 0,3368 angry-aggresive: 0,7252 
    Macro-averaged F-Measure: 0,7044
    amazed-suprised: 0,6872 happy-pleased: 0,5856 relaxing-calm: 0,8523 quiet-still: 0,8304 sad-lonely: 0,4923 angry-aggresive: 0,7787 
    Macro-averaged Specificity: 0,9246
    amazed-suprised: 0,8897 happy-pleased: 0,9261 relaxing-calm: 0,8475 quiet-still: 0,9636 sad-lonely: 0,9899 angry-aggresive: 0,9308 
    Average Precision: 0,8619
    Coverage: 1,3811
    OneError: 0,1893
    IsError: 0,3555
    ErrorSetSize: 0,6777
    Ranking Loss: 0,1005
    Mean Average Precision: 0,8087
    amazed-suprised: 0,7073 happy-pleased: 0,6567 relaxing-calm: 0,9344 quiet-still: 0,9271 sad-lonely: 0,7671 angry-aggresive: 0,8598 
    Geometric Mean Average Precision: 0,8016
    amazed-suprised: 0,7073 happy-pleased: 0,6567 relaxing-calm: 0,9344 quiet-still: 0,9271 sad-lonely: 0,7671 angry-aggresive: 0,8598 
    Mean Average Interpolated Precision: 0,8104
    amazed-suprised: 0,7385 happy-pleased: 0,6642 relaxing-calm: 0,9226 quiet-still: 0,9137 sad-lonely: 0,7703 angry-aggresive: 0,8534 
    Geometric Mean Average Interpolated Precision: 0,8048
    amazed-suprised: 0,7385 happy-pleased: 0,6642 relaxing-calm: 0,9226 quiet-still: 0,9137 sad-lonely: 0,7703 angry-aggresive: 0,8534 
    Micro-averaged AUC: 0,9226
    Macro-averaged AUC: 0,9013
    amazed-suprised: 0,8858 happy-pleased: 0,7873 relaxing-calm: 0,9424 quiet-still: 0,9749 sad-lonely: 0,9091 angry-aggresive: 0,9082 
    Logarithmic Loss: 2,0458
    


