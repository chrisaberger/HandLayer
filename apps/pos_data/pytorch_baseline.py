import nltk
from nltk.corpus import treebank
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse

def prepare_word_sequence(seq, to_ix):
    idxs = [to_ix[w] for w, t in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prepare_tag_sequence(seq, to_ix):
    idxs = [to_ix[t] for w, t in seq]
    return torch.tensor(idxs, dtype=torch.long)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def dump_state(self):
        np.save("h_t.npy", self.hidden[0].data)
        np.save("c_t.npy", self.hidden[1].data)
        np.save("weight_ih_l0.npy", self.lstm.weight_ih_l0.data.numpy())
        np.save("weight_hh_l0.npy", self.lstm.weight_hh_l0.data.numpy())
        np.save("bias_ih_l0.npy", self.lstm.bias_ih_l0.data.numpy())
        np.save("bias_hh_l0.npy", self.lstm.bias_hh_l0.data.numpy())
        np.save("embedding.npy", self.word_embeddings.weight.detach().numpy())
        np.save("linear_weight.npy", self.hidden2tag.weight.detach().numpy())
        np.save("linear_bias.npy", self.hidden2tag.bias.detach().numpy())

    def load_state(self):
        self.hidden = (torch.tensor(np.load("h_t.npy")), torch.tensor(np.load("c_t.npy")))
        self.lstm.weight_ih_l0.data = torch.Tensor(np.load("weight_ih_l0.npy"))
        self.lstm.weight_hh_l0.data = torch.Tensor(np.load("weight_hh_l0.npy"))
        self.lstm.bias_ih_l0.data = torch.Tensor(np.load("bias_ih_l0.npy"))
        self.lstm.bias_hh_l0.data = torch.Tensor(np.load("bias_hh_l0.npy"))
        self.word_embeddings.weight.data = torch.Tensor(np.load("embedding.npy"))
        self.hidden2tag.weight.data = torch.Tensor(np.load("linear_weight.npy"))
        self.hidden2tag.bias.data = torch.Tensor(np.load("linear_bias.npy"))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def training():
    sentences = treebank.tagged_sents(tagset='universal')
    tags = set([
        tag for sentence in sentences 
        for _, tag in sentence
    ])

    words = set([
        w for sentence in sentences
        for w, _ in sentence
    ])

    train_sentences = []
    test_sentences = []

    test_file = open("test_sentences.txt","w") 
    for i, s in enumerate(sentences):
        # holdout 20%
        if i % 5:
            test_sentences.append(s)
            for w, t in s:
                test_file.write(w + "\n")
                test_file.write(t + "\n")
            test_file.write("end_sentence_done_here\n")
        else:
            train_sentences.append(s)
    test_file.close()

    tag_file = open("tag_file.txt","w") 
    tag_to_ix = {}
    for i, t in enumerate(tags):
        tag_file.write(t)
        tag_file.write("\n")
        tag_to_ix[t] = i
    tag_file.close()

    word_file = open("word_file.txt","w")
    word_to_ix = {}
    for i, w in enumerate(words):
        word_file.write(w)
        word_file.write("\n")
        word_to_ix[w] = i
    word_file.close()

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(50):  # again, normally you would NOT do 300 epochs, it is toy data
        total_loss = 0
        acc = 0.0
        for sentence in train_sentences:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_word_sequence(sentence, word_to_ix)
            targets = prepare_tag_sequence(sentence, tag_to_ix)

            #print(sentence_in)
            #sentence_in = torch.tensor([[[7,1,8,2,5]]])
            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()

            total_loss += loss 
            #print(tag_scores)
            prediction = tag_scores.data.numpy().argmax(axis=1)
            acc += 100*np.mean(prediction == targets.numpy())

            optimizer.step()
        
        acc2 = 0
        for sentence in test_sentences:
            model.hidden = model.init_hidden()
            sentence_in = prepare_word_sequence(sentence, word_to_ix)
            targets = prepare_tag_sequence(sentence, tag_to_ix)
            tag_scores = model(sentence_in)
            prediction = tag_scores.data.numpy().argmax(axis=1)
            acc2 += 100*np.mean(prediction == targets.numpy())

        print("loss: ", total_loss.detach().numpy()/len(train_sentences), 
              " train acc: ", acc/len(train_sentences), 
              " test acc: ", acc2/len(test_sentences))

    model.dump_state()

def inference():
    sentences = treebank.tagged_sents(tagset='universal')
    train_sentences = []
    test_sentences = []
    for i, s in enumerate(sentences):
        # holdout 20%
        if i % 5:
            test_sentences.append(s)
        else:
            train_sentences.append(s) 

    print("Num sentences: ",len(test_sentences))

    file = open("tag_file.txt", "r") 
    tag_to_ix = {}
    i = 0
    for line in file:
        tag_to_ix[line[:-1]] = i
        i += 1
    file.close()

    file = open("word_file.txt", "r") 
    word_to_ix = {}
    i = 0
    for line in file:
        word_to_ix[line[:-1]] = i
        i += 1
    file.close()

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    model.load_state()

    hits = 0
    total = 0
    i = 0
    for sentence in test_sentences:
        sentence_in = prepare_word_sequence(sentence, word_to_ix)
        targets = prepare_tag_sequence(sentence, tag_to_ix)
        for i, w in enumerate(sentence_in):
            tag_scores = model(torch.tensor([w]))
            prediction = tag_scores.data.numpy().argmax(axis=1)
            if prediction == targets[i]:
                hits += 1
            total += 1

    print("test accuracy:", hits/total, hits, "/", total)


## Main code here.
parser = argparse.ArgumentParser()
parser.add_argument("--mode", action="store", type=str,
                    choices=["inference", "training"],
                             required=True)
args = parser.parse_args()

torch.manual_seed(1)

print(args)

EMBEDDING_DIM = 32
HIDDEN_DIM = 64

if args.mode == "inference":
    inference()
elif args.mode == "training":
    training()
