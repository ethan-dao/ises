import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#Load data
train_raw = pd.read_csv('/Users/irisgu/VisualStudioCode/ises/data/train_MPRA.txt', delimiter='\t', header=None)
test_raw = pd.read_csv('/Users/irisgu/VisualStudioCode/ises/data/test_MPRA.txt', delimiter='\t', header=None)
train_sol = pd.read_csv('/Users/irisgu/VisualStudioCode/ises/data/trainsolutions.txt', delimiter='\t', header=None)

strand_length = 295

#Preprocess train data
train_scores = np.array(train_raw.iloc[:, 2:297])
raw_dna_strands_train = [list(train_raw[1][i]) for i in range(len(train_raw))]
embedded_dna_strands_train = [
    np.column_stack(
        (np.array(pd.get_dummies(pd.concat([pd.Series(raw_dna_strands_train[i]), pd.Series(["A", "C", "T", "G"])]), dtype="int"))[:-4],
         np.arange(295))
    )
    for i in range(len(train_raw))
]
embedded_dna_strands_train = [
    embedded_dna_strands_train[i] for i in range(len(embedded_dna_strands_train)) if not ("N" in raw_dna_strands_train[i])
]
train_scores = [train_scores[i] for i in range(len(raw_dna_strands_train)) if not ("N" in raw_dna_strands_train[i])]

#Model
class SelfAttentionFeedForward(nn.Module):
    def __init__(self, attention_size, embed_size, hidden_size, hidden_layers, lr):
        super().__init__()
        self.attention_size = attention_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.lr = lr

        #Initialize weight matrices for self-attention
        self.W_Q = nn.Parameter(torch.rand(self.attention_size, self.embed_size))
        self.W_K = nn.Parameter(torch.rand(self.attention_size, self.embed_size))
        self.W_V = nn.Parameter(torch.rand(self.attention_size, self.embed_size))
        self.b = nn.Parameter(torch.rand(295))  # Bias term

        #Feed-forward network layers
        fc1 = nn.Linear(self.attention_size, self.hidden_size)
        relu1 = nn.ReLU()
        self.layers = [fc1, relu1]
        for _ in range(self.hidden_layers - 1):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.hidden_size, 1))

        self.layers = nn.Sequential(*self.layers)

        #Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def forward(self, x):
        #Attention mechanism
        queries = torch.matmul(self.W_Q, x.T)
        keys = torch.matmul(self.W_K, x.T)
        values = torch.matmul(self.W_V, x.T)

        attention = torch.matmul(queries, keys.T)
        weights = F.softmax(attention, dim=0)

        contextualized_embeddings = torch.matmul(weights, values).T

        #Feed-forward layers
        output = self.layers(contextualized_embeddings)

        #Sigmoid activation for binary classification
        return torch.sigmoid(output + self.b)

    def loss(self, predicted, y):
        #Binary cross-entropy loss
        criterion = nn.BCELoss()
        return criterion(predicted, y)

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        pred = self.forward(x)
        loss = self.loss(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, x, y, epochs=10):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(x)):
                xi = torch.Tensor(x[i])
                yi = torch.Tensor(y[i])
                total_loss += self.train_step(xi, yi)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(x):.4f}")

#Initialize and train the model
model = SelfAttentionFeedForward(attention_size=50, embed_size=5, hidden_size=20, hidden_layers=3, lr=1e-5)
model.train(embedded_dna_strands_train, train_scores, epochs=10)

#Test prediction on a sample
pred = model.predict(torch.Tensor(embedded_dna_strands_train[0])).detach().numpy()
print(pred - train_scores[0])  #Print the difference between predicted and actual values

#The output should now reflect predictions based on the training data

#Generate predictions
predictions = []
for i, strand in enumerate(embedded_dna_strands_train):
    logits = model.forward(torch.Tensor(strand)).detach().numpy().flatten()
    predicted_scores = torch.sigmoid(torch.Tensor(logits)).numpy().flatten()  #Apply Sigmoid to logits
    predicted_labels = ["A" if score > 0.5 else "R" for score in predicted_scores]  #0.5 threshold
    sequence_id = f"train{str(i).zfill(4)}"
    nucleotides = raw_dna_strands_train[i]

    for pos, (nucleotide, label) in enumerate(zip(nucleotides, predicted_labels)):
        predictions.append([label, sequence_id, nucleotide])

#Save predictions
predictions_df = pd.DataFrame(predictions, columns=["Activation/Repression", "Sequence_ID", "Nucleotide"])
predictions_df.to_csv("predictions.txt", sep="\t", header=False, index=False)

#Load training solutions
training_solutions = pd.read_csv("/Users/irisgu/VisualStudioCode/ises/data/trainsolutions.txt", delimiter="\t", header=None, names=["Activation/Repression", "Sequence_ID", "Nucleotide"])

#Load predictions
predictions = pd.read_csv("predictions.txt", delimiter="\t", header=None, names=["Activation/Repression", "Sequence_ID", "Nucleotide"])

#Ensure "Nucleotide" columns are strings in both DataFrames
predictions["Nucleotide"] = predictions["Nucleotide"].astype(str)
training_solutions["Nucleotide"] = training_solutions["Nucleotide"].astype(str)

#Merge predictions and training solutions
comparison = pd.merge(predictions, training_solutions, on=["Sequence_ID", "Nucleotide"], how="inner", suffixes=('_pred', '_true'))

#Add a "Match" column to indicate where Prediction matches Truth
comparison["Match"] = comparison["Activation/Repression_pred"] == comparison["Activation/Repression_true"]

#Calculate accuracy
accuracy = comparison["Match"].mean()
print(f"Accuracy: {accuracy:.2%}")

#Save the comparison to a file for inspection
comparison.to_csv("comparison.txt", sep="\t", index=False)