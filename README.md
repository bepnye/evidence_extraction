# Evidence Extraction

This repository hosts scripts for formatting data and running baseline models for evidence extraction tasks.

## External Dependencies

* [evidence-inference](https://github.com/jayded/evidence-inference) - Evidence identification/classification data.
* [bert](https://github.com/naver/biobert-pretrained/releases) - A trained BERT model of your choice. BioBERT is linked here.
* [EBM-NLP](https://github.com/bepnye/EBM-NLP) - NER data for PICO elements.

## Getting Started

To run a given task, you must:
1. Process the relevant source data to generate intermediate representations
2. Generate input for the task of choice
3. Train the appropriate model

For example, learn to identify evidence-bearing sentences, you would need to run the following:

```
cd scripts/
python process_evidence_inference
python generate_sentence_classifier_input
cd ../models/sentence_classifier/
./train.sh
```

## Models / Tasks

### Sentence Classification

This family of tasks use the original BERT method for sentence classification (predict based on the [CLS] token).

#### Evidence identification (ev\_binary)

Given a sentence, predict if it contains a conclusion about an ICO frame. Positive examples are all evidence spans from evidence-inference, negative examples are random sentences.

#### Evidence classification (ev\_trinary)

Given an evidence-bearing sentence, predict what the conclusion is (increased, decreased, no sig difference). This is equivalent to the NAACL oracle task presented in the [corpus paper](https://arxiv.org/abs/1904.01606).

### PICO extraction

TODO: add processing for EBM-NLP, provide sequence for training the NER tagger.
