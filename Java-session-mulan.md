

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


```Java
// Load the train and test partitions from ARFF files
var train = new MultiLabelInstances("emotions-train.arff", "emotions.xml");
var test = new MultiLabelInstances("emotions-test.arff", "emotions.xml")
```


```Java
// Some data exploration
System.out.println(
    "Instances: " + train.getNumInstances() +    
    ", Labels: " + train.getNumLabels() + 
    ", Card: " + train.getCardinality())
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
List<Measure> metrics = Arrays.asList(
    new HammingLoss(), new ExampleBasedAccuracy(), new ExampleBasedFMeasure());
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

```
