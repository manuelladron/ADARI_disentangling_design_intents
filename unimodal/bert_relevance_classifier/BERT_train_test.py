#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import time
import numpy as np
import datetime
import pdb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# In[4]:


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


# In[13]:


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.to('cpu').numpy()
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def acc_prec_recall_f1(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.to('cpu').numpy()
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    acc = accuracy_score(labels_flat, pred_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, pred_flat, average='binary')
    return acc, precision, recall, f1


# In[12]:


def train(model, loader, optimizer, scheduler):
    start_time = time.time()

    running_loss = 0.0
    running_acc = 0.0
    running_prec = 0.0
    running_recall = 0.0
    running_f1 = 0.0 
    
    model.train()

    for step, batch in enumerate(loader):
        b_input_ids = batch[0].to(device) # .long()  # batch x seq_length
        b_input_mask = batch[1].to(device) # .float() # batch x seq_length
        b_labels = batch[2].view((batch[2].shape[0]))
        b_labels = b_labels.to(device) # .long()    # batch x 1 (num_classes, binary)

        model.zero_grad()        

        # The "logits"--the model outputs prior to activation.
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)
        
        running_loss += loss.item()
        acc, prec, recall, f1 = acc_prec_recall_f1(logits, b_labels)
        
        running_acc += acc
        running_prec += prec
        running_recall += recall
        running_f1 += f1
  
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
    running_loss /= len(loader)
    running_acc /= len(loader)
    running_prec /= len(loader)
    running_recall /= len(loader)
    running_f1 /= len(loader)

    print("  Average training loss: {0:.2f}".format(running_loss))
    print("  Average training acc: {0:.2f}".format(running_acc))
    print("  Average training prec: {0:.2f}".format(running_prec))
    print("  Average training recall: {0:.2f}".format(running_recall))
    print("  Average training f1: {0:.2f}".format(running_f1))
    
    print("  Training took: %.2f seconds" % (time.time() - start_time))
    
    return running_loss, running_acc, running_prec, running_recall, running_f1


def test(model, loader):
    with torch.no_grad():
        start_time = time.time()
        model.eval()

        running_loss = 0
        running_acc = 0.0
        running_prec = 0.0
        running_recall = 0.0
        running_f1 = 0.0 
        nb_eval_steps = 0

        for batch in loader:
            b_input_ids = batch[0].to(device) 
            b_input_mask = batch[1].to(device) 
            b_labels = batch[2].to(device) 

            # The "logits" are the output values prior to applying an activation function like the softmax.
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

            running_loss += loss.item()
            acc, prec, recall, f1 = acc_prec_recall_f1(logits, b_labels)

            running_acc += acc
            running_prec += prec
            running_recall += recall
            running_f1 += f1
            
        running_loss /= len(loader)
        running_acc /= len(loader)
        running_prec /= len(loader)
        running_recall /= len(loader)
        running_f1 /= len(loader)

        print("  Validation Loss: {0:.2f}".format(running_loss))
        print("  Validation Accuracy: {0:.2f}".format(running_acc))
        print("  Validation Precision: {0:.2f}".format(running_prec))
        print("  Validation Recall: {0:.2f}".format(running_recall))
        print("  Validation F1: {0:.2f}".format(running_f1))
        
        print("  Validation took: %.2f seconds" % (time.time() - start_time))

    return running_loss, running_acc, running_prec, running_recall, running_f1

