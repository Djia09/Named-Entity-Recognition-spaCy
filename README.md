# Named-Entity-Recognition
This is the implementation of a Named-Entity-Recognition model using [SpaCy](https://spacy.io/) trained on CoNLL-2003 dataset. The paper about CoNLL-2003 dataset is given [here](http://aclweb.org/anthology/W03-0419) and the dataset can be download in this site: [https://www.clips.uantwerpen.be/conll2003/ner/](https://www.clips.uantwerpen.be/conll2003/ner/)

## Getting started

### Requirements
* Python 2 or 3
* scikit-learn
* SpaCy

SpaCy installation tutorial can be found here: [https://spacy.io/usage/](https://spacy.io/usage/). 

### Training a model.
```
$ python train_ner_spacy.py 
The following arguments are mandatory:
  -tp		Path to CoNLL-2003 training data

The following arguments are optional:
  -output	Path where the model will be saved. Default: "./Model"
  -n_iter	Number of epochs. Default: 16
```

For example, run:
```
$ python train_ner_spacy.py -n 32 -output ./Model/first_model -tp ./CoNLL-2003/eng.train
```

### Testing a model.
```
$ python test_ner_spacy.py 
The following arguments are mandatory:
  -tp		Path to CoNLL-2003 testing data
  -model	Path to the saved model
```

For example, run: 
```
$ python test_ner_spacy.py -tp ./CoNLL-2003/eng.testa -model ./Model/first_model
```

## Results
The model testing was done with CoNLL-2003 dataset test A and test B for 32 iterations.

| | Test A  | Test B |
| -- | ------------- | ------------- |
| Random accuracy score | 0.83 | 0.82 |
| Accuracy score | 0.97 | 0.95 |
| Precision | 0.88 | 0.80 |
| Recall | 0.86 | 0.80 |
| **F1 score** | **0.87** | **0.80** |

* The random accuracy score is the accuracy where the prediction is a list only composed by 'O'. 
* The accuracy score is the word-by-word score. 
* Precision is the word-by-word precision score for the predictions different from 'O'.
* Recall is the word-by-word recall score for the predictions different from 'O'.
* F1 score is the word-by-word f1 score for the predictions different from 'O': 
F1 score = (2*Precision*Recall) / (Precision + Recall)

For example: 
```
sent = ["I", "go", "to", "Harvard", "University", "in", "the", "USA".]
y_true = ["O", "O", "O", "ORG", "ORG", "O", "O", "LOC"]
y_pred = ["O", "O", "O", "LOC", "O", "O", "O", "LOC"]
```
Random accuracy will be: 5/8 = 0.625
Accuracy score will be: 6/8 = 0.75 
Precision will be: 1/2 = 0.50 (LOC and LOC from y_pred)
Recall will be: 1/3 = 0.33 (ORG, ORG and LOC from y_true)
F1 score will be: 2*1/2*1/3 / (1/2 + 1/3) = 0.40

## Author
David JIA