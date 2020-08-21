# Evidence Extraction

This repository hosts scripts for formatting data and running baseline models for evidence extraction tasks on randomized controlled trials (RCTs).

## External Dependencies

* [evidence-inference](https://github.com/jayded/evidence-inference) - Evidence identification/classification data.
* [EBM-NLP](https://github.com/bepnye/EBM-NLP) - NER data for PICO elements.
* [bert](https://github.com/naver/biobert-pretrained/releases) - A trained BERT model of your choice. BioBERT is linked here.

## Overview

Evidence extraction is composed of several pipeline models that are composed to annotate:
* NER spans for the Interventions/Comparators, Participants, and Outcomes
* Evidence-bearing sentences
* Relations indicating which Interventions were compared to each other and what the measured Outcomes were
* Inference of the efficacy of an Intervention with respect to a Comparator and an Outcome

The NER component is trained on EBM-NLP, and the evidence and relations are trained on Evidence Inference.

## Implementation

The codebase reads source data into an intermediate Doc class, and performs all subsequent steps from there. The interface for generating Docs from different data sources are in `scripts/process_{data_source}.py`.

Writing outputs for the various models to consume is in `scripts/writer.py`.

The models can be run either from scripts in the corresponding `models/{model_type}` directory, or run as a python module from `scripts/run_pipeline.py`.
