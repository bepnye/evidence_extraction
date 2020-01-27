import random
import torch.nn as nn
from load_data import * 
from model import * 
import torch.optim as optim 
from sklearn.metrics import f1_score

def find_entity_match(entity_list, str_):
    """ 
    Find which entity matches the str_ description (i.e. compare entity.text to str_).
    If there are no matches, return None, otherwise return the matched entity.
    @param entity_list a list of entities to search in.
    @param str_        a string to match the entities with.
    """
    for e in entity_list:
        if e.text == str_:
            return e

    return None


def label_to_val(label):
    """ Convert the @param label (is type string) to a natural number. """
    return {'INCR': 0, 'DECR': 1, 'SAME': 2, 'NULL': 3}.get(label)

def split_data(df):
    """ Split the data into train, dev, and test. """
    random.shuffle(df)
    train_split = int(len(df) * 0.6)
    dev_split   = int(len(df) * 0.8)
    return df[:train_split], df[train_split:dev_split], df[dev_split:]


def to_segmentation_ids(tokens):
    """ 
    Take @param tokens and iterate through it to get segmentation ids. 
    Convert text at end to correct format (of token values).
    """
    start = 0
    segments = []
    for t in tokens:
        if t == '[SEP]':
            start += 1

        segments.append(start)

    return TOKENIZER.convert_tokens_to_ids(tokens), segments

def extract_data(df):
    """ 
    Extract the data from the classes and reformat it. 
    
    Goal: Create tuples of 3 -> (entity1, entity2, relation).
    
    The current entity map is (string1, string2): string3, 
    where string1 and string2 are string versions of the entities,
    and string 3 is the label. 
    """
    all_data   = []
    labels = []
    for d in df:
        mapping   = d.entity_map[0]
        relations = d.entity_map[1] # this is a dictionary
        doc_data  = []
        for key in relations.keys():
            # find what entity matches us
            entity1, entity2 = find_entity_match(mapping, key[0]), find_entity_match(mapping, key[1])
            if len(entity1.mentions) == 0 or len(entity2.mentions) == 0:
                continue

            doc_data.append((entity1, entity2))
            labels.append(label_to_val(relations[key]))
            assert(not(entity1 is None) and not(entity2 is None)) 
            
        text, segment_ids = to_segmentation_ids(d.tokenized_text)
        all_data.append({'text': text, 'segment_ids': segment_ids, 'relations': doc_data})

    return all_data, labels         

def create_model():
    """ Create a model and return it. """
    return BERTVergaPytorch()

def train_model(model, df, batch_size = 1, epochs = 100):
    """ Take a model and train it with the given data. """
    train, dev, test = split_data(df)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr = 1e-5) 
    for epoch in range(epochs):
        # define losses to use later
        training_loss = 0
        dev_loss      = 0 
        # single epoch train
        for batch_range in range(0, len(train), batch_size): 
            data = train[batch_range: batch_range + batch_size]
            inputs, labels = extract_data(data)
            if len(labels) == 0: continue

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backwards + optimize
            outputs = model(inputs)
            loss = criterion(outputs, torch.tensor(labels).cuda())
            loss.backward()
            optimizer.step()

            # add loss 
            training_loss += loss.item()
        
        # evaluate on validation set
        dev_outputs = []
        for batch_range in range(0, len(dev), batch_size):
            data = dev[batch_range: batch_range + batch_size]
            inputs, labels = extract_data(data) 
            if len(labels) == 0: continue 

            # run model through validation data
            with torch.no_grad():
                outputs = model(inputs)
            
            loss = criterion(outputs, torch.tensor(labels).cuda())
            dev_loss += loss.item()
            dev_outputs.extend(outputs.cpu().numpy()) # or something like this
       
        outputs = [torch.argmax(x).item() for x in outputs]
        f1 = f1_score(labels, outputs, average = 'macro')
        print("Epoch {}\nDev F1 score: {}\nDev Loss: {}\nTraining Loss: {}\n\n".format(epoch, f1, dev_loss, training_loss))

def main(): 
    df    = load_data()
    model = create_model()
    train_model(model, df)

if __name__ == '__main__':
    main()
