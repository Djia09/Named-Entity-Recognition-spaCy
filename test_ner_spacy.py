import spacy
import plac
import numpy as np
import time
import re

from conllToSpacy import main
testing, ent_true = main('./CoNLL-2003/eng.testa', test='test')
testing_tokens = [x.split() for x in testing]
print('Length testing sentences: ', len(testing))
print('Length testing tokens: ', len(testing_tokens))

def compute_score(L1, L2):
    correct_answers = 0
    nb_answers = 0
    for i in range(len(L1)):
        if L1[i] != 'O':
            nb_answers += 1
            if L1[i] == L2[i]:
                correct_answers += 1
    print('%d correct out of %d answers.' % (correct_answers, nb_answers))
    return correct_answers/nb_answers

model = './Model/first_full_32_notAdd_UP'

def main():
    # test the saved model
    print("Loading from", model)
    nlp2 = spacy.load(model)
    ent_pred = []
    testing_pred = []
    print("Start predicttion...")
    count = 1
    k = 1

    for text in testing:
        start = time.time()
        doc = nlp2(text)
        entities = ['O'] * len(text.split())
        for ent in doc.ents:
            try:
                entities[text.split().index(ent.text)] = ent.label_
            except IndexError:
                print('Index Error! Ent:', list(ent.text), '. Text:', text)
            except ValueError:
                print('Value Error! Ent:', list(ent.text), '. Text:', text)
        ent_pred.append(entities)
        testing_pred.append([t.text for t in doc])
        print(str(count)+'/'+str(len(testing))+' done in %fs' % (time.time()-start))
        count += 1

    # Check whether there are the same number of sentences, and the same number of words in each sentence. 
    print('Length pred sentences: ', len(testing_pred))
    for i in range(len(testing_tokens)):
        if len(ent_true[i]) != len(ent_pred[i]):
            print("NOT THE SAME LENGTH !")
            print("True Text: ", testing_tokens[i])
            print("Pred Text: ", testing_pred[i])
            print("Entities true: ", ent_true[i])
            print("Entities pred: ", ent_pred[i])

    print('Pred Entity: ', set([x for y in ent_pred for x in y]))
    print('True Entity: ', set([x for y in ent_true for x in y]))
    y_pred = [x for y in ent_pred for x in y]
    y_true = [x for y in ent_true for x in y]

    Precision = compute_score(y_pred, y_true)
    Recall = compute_score(y_true, y_pred)
    F1_score = 2*(Recall * Precision) / (Recall + Precision)
    print('Precision: %0.2f' % (Precision))
    print('Recall: %0.2f' % (Recall))
    print('F1 score: %0.2f' % (F1_score))
        
if __name__ == '__main__':
    plac.call(main) 