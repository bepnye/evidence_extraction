import copy
import random
import torch.nn as nn
import argparse
import sys
from load_data import *
from load_CDR_data import load_CDR
from model import *
from utils import *
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report
import numpy as np
ev_inf_label_config = {'E1': 1, 'E2': 2, 'NULL': 3}
cdr_label_config = {'E1': 0, 'E2': 1, 'NULL': 7}

def get_dataset(dataset):
    """ Generate the dataset based on what is requested. """
    generate_data = {'evidence-inference': load_data, 'CDR': load_CDR}.get(dataset)
    return generate_data()

def create_model(dataset, bert_backprop):
    """
    Create a model with the given specifications and return it.

    @param dataset specifies what dataset to use so we know how big our output dim should be.
    @param bert_backprop determines if we are to backprop through BERT.
    @return a model set up with the given specification.
    """
    assert(dataset in set(['evidence-inference', 'CDR']))
    output_dimensions = {'evidence-inference': 4, 'CDR': 2}.get(dataset)
    return BERTVergaPytorch(output_dimensions, bert_backprop = bert_backprop)

def evaluate_model(model, criterion, label_config, test, epoch, batch_size):
    # evaluate on validation set
    model.eval()
    test_outputs = []
    test_labels  = []
    test_loss = 0
    for batch_range in range(0, len(test), batch_size):
        data = test[batch_range: batch_range + batch_size]
        inputs, labels, _ = extract_data(data, label_config)
        if len(labels) == 0: continue

        # run model through validation data
        with torch.no_grad():
            outputs = model(inputs)

        loss = criterion(outputs, torch.tensor(labels).cuda())
        test_loss += loss.item()
        test_outputs.extend(outputs.cpu().numpy()) # or something like this
        test_labels.extend(labels)

    outputs = [np.argmax(x) for x in test_outputs]
    labels  = test_labels
    f1 = f1_score(labels, outputs, average = 'macro')
    if max(labels) > 1 or max(outputs) > 1:
        print(classification_report(labels, outputs))
        outputs = [1 if x != 3 else 0 for x in outputs]
        labels  = [1 if x != 3 else 0 for x in labels]
        bin_f1  = f1_score(labels, outputs, average = 'macro')
    else:
        bin_f1 = 0

    print(classification_report(labels, outputs))
    print("Epoch {}\nF1 score: {}\nBinary F1: {}\nLoss: {}\n".format(epoch, f1, bin_f1, test_loss))
    return f1

def train_model(model, df, parameters):
    """ Take a model and train it with the given data. """
    # get parameters of how to train model
    epochs     = parameters.epochs
    batch_size = parameters.batch_size
    balance_classes = parameters.balance_classes
    learning_rate   = parameters.lr
    label_config = 

    # split data, set up our optimizers
    best_model = None
    max_f1_score = 0 # best f1 score seen thus far
    train, dev, test = split_data(df, parameters.percent_train)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    for epoch in range(epochs):
        model.train()
        # define losses to use later
        training_loss = 0
        dev_loss      = 0
        
        train_data, train_labels, _ = extract_data(train, balance_classes == 'True')
        label_offset = 0
        # single epoch train
        for batch_range in range(0, len(train), batch_size):
            inputs = train_data[batch_range: batch_range + batch_size]

            if len(inputs) == 0: continue
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backwards + optimize
            outputs = model(inputs)
            labels  = train_labels[label_offset: label_offset + len(outputs)]
            loss = criterion(outputs, torch.tensor(labels).cuda())
            loss.backward()
            optimizer.step()

            # add loss
            training_loss += loss.item()

            # next labels
            label_offset += len(outputs)

        ### Print the losses and evaluate on the dev set ###
        print("Epoch {} Training Loss: {}\n".format(epoch, training_loss))
        f1_score = evaluate_model(model, criterion, label_config, dev, epoch, batch_size)

        # update our scores to find the best possible model
        best_model   = copy.deepcopy(model) if max_f1_score < f1_score else best_model
        max_f1_score = max(max_f1_score, f1_score)

    print("Final test run:\n")
    evaluate_model(best_model, criterion, label_config, test, epoch, batch_size)

def main(args=sys.argv[1:]):
    ### Use arg parser to read in data. ###
    parser = argparse.ArgumentParser(description="Parses input specifications of how to run model.")
    parser.add_argument("--dataset", dest='dataset', required=True, help="Which dataset to use.")
    parser.add_argument("--epochs", dest='epochs', type=int, required=True, help="How many epochs to run model for.")
    parser.add_argument("--learning_rate", dest='lr', type=float, required=True, help="What should the learning rate be?")
    parser.add_argument("--batch_size", dest='batch_size', type=int, required=True, help="What should the batch_size be?")
    parser.add_argument("--balance_classes", dest="balance_classes", default=False, help="Should we balance the classes for you?")
    parser.add_argument("--bert_backprop", dest="bert_backprop", default=False, help="Should we backprop through BERT?")
    parser.add_argument("--percent_train", dest="percent_train", type=float, default=1, help="What percent if the training data should we use?")
    args = parser.parse_args()
    print("Running with the given arguments:\n\n{}".format(args))

    ### GET THE DATA ###
    df = get_dataset(args.dataset)

    ### LOAD THE MODEL ###
    model = create_model(dataset=args.dataset, bert_backprop=args.bert_backprop == 'True')

    ### TRAIN ###
    train_model(model, df, args)

if __name__ == '__main__':
    main()
