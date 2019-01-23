#!/usr/bin/env python
# coding: utf8
"""
For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import time
from conllToSpacy import main
import argparse
import sys

# Parsing argument for command-line.
parser = argparse.ArgumentParser(description="Training an NER model with SpaCy.")
parser.add_argument("-tp", "--train_path", help="Path to CoNLL train dataset.")
parser.add_argument("-output", "--output_path", help="Path where to save the output model.")
parser.add_argument("-n", "--n_iter", help="Number of epochs for training", type=int)
args = parser.parse_args()

if args.train_path:
    TRAIN_DATA = main(args.train_path)
    print("Got train data at", args.train_path)
else:
    print("No training data path given ! Interrupting the script...")
    sys.exit()
if args.output_path:
    output_dir = args.output_path
    print("Model will be saved at", output_dir)
else:
    print("No output path given ! The model will be saved in a folder called ./Model.")
    output_dir = "./Model/first_full_32_notAdd_UP"
if args.n_iter:
    n_iter = args.n_iter
    print("Number of iterations chosen: ", n_iter)
else:
    print("Default number of iterations given: 16")
    n_iter = 16

def main(model=None, output_dir=output_dir, n_iter=n_iter):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            start = time.time()
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(16., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print('Epoch', str(itn+1)+'/'+str(n_iter), '. Losses', losses, ' in %fs' % (time.time()-start))

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

START = time.time()
if __name__ == '__main__':
    main(model=None, output_dir=output_dir, n_iter=n_iter)
print('Done in %fs' % (time.time()-START))
