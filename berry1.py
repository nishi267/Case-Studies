from __future__ import unicode_literals, print_function
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
import random
from pathlib import Path
import pickle
from spacy.util import minibatch, compounding
import os
from os import path, mkdir
import spacy
import time
import numpy as np  
from string import digits
import json 
import yake
import pandas as pd 

print("Directory: ", os.getcwd)
def load_data_spacy(file_path):
    #print("curr directory: ", os.getcwd())
    file=open(file_path, 'r')
    training_data, entities,sentence, unique_labels=[],[],[],[]
    current_annotation=None
    end=0
    for line in file:
        print(line)
        line=line.strip("\n").split(",")
        if len(line) >1:
            label=line[1][0:]
            label_type=line[1][0]
            word=line[0]
            sentence.append(word)
            end+=(len(word)+1)

            if label_type!='I' and current_annotation:
                entities.append((start, end-2-len(word), current_annotation))
                current_annotation=None
            if label_type=='B':
                start=end-len(word)-1
                current_annotation =label
            if label_type=='I':
                current_annotation =label
            if label !='O' and label not in unique_labels:
                unique_labels.append(label)
               # lines with len == 1 are breaks between sentences
        if line[0]=='.':
            if current_annotation:
                entities.append((start, end - 1, current_annotation))
            sentence = " ".join(sentence)
            training_data.append([sentence, {'entities' : entities}])
            # reset the counters and temporary lists
            end = 0            
            entities, sentence = [], []
            current_annotation = None
    file.close()
    return training_data, unique_labels  

TRAIN_DATA, LABELS=load_data_spacy("dataset1.csv")
nlp = spacy.blank('en')
def train_model(train_data):

    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    for _,annotation in train_data:
        for ent in annotation['entities']:
            ner.add_label(ent[2])
  
    # Disable other pipelines in SpaCy to only train NER
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(10):
            random.shuffle(train_data) # shuffle the training data before each iteration
            losses = {}
            index=0
            batches = minibatch(train_data, size = compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                try:

                    nlp.update(          
                    texts,
                    annotations,
                    drop = 0.2,  
                    losses = losses)
                    print(annotations)
                except Exception as e:
                    print("exception", e)

            print("Losses", losses)

train_model(TRAIN_DATA)
nlp.to_disk("Trial")
print("Model Trained")

nlpmodel=spacy.load("Trial")
text="Home page is not launching on Google Chrome"
doc=nlpmodel(text)
for ent in doc.ents:
    print(f'{ent.label_.upper():{90}}-{ent.text}')

  


