import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Load data
train_raw = pd.read_csv('/Users/irisgu/VisualStudioCode/ises/data/train_MPRA.txt', delimiter='\t', header=None)
test_raw = pd.read_csv('/Users/irisgu/VisualStudioCode/ises/data/test_MPRA.txt', delimiter='\t', header=None)
train_sol = pd.read_csv('/Users/irisgu/VisualStudioCode/ises/data/trainsolutions.txt', delimiter='\t', header=None)

strand_length = 295

# Preprocess train data
# Get our x and y data
train_scores = np.array(train_raw.iloc[:, 2:297]) # Dimensions are 8000 (samples) by 295 (SHARPR scores per nucleotide)
raw_dna_strands_train = [list(train_raw[1][i]) for i in range(len(train_raw))] # List of lists holding DNA strands separated by character. Size 8000 lists each of length 290
embedded_dna_strands_train = [np.column_stack((np.array(pd.get_dummies(pd.concat([pd.Series(raw_dna_strands_train[i]), pd.Series(["A", "C", "T", "G"])]), dtype='int'))[:-4], np.arange(295))) for i in range(len(train_raw))] #One hot encoded dna strands, list of 8000 matrices, each (295,5)
embedded_dna_strands_train = [embedded_dna_strands_train[i] for i in range(len(embedded_dna_strands_train)) if not ("N" in raw_dna_strands_train[i])]
train_scores  = [train_scores[i] for i in range(len(raw_dna_strands_train)) if not ("N" in raw_dna_strands_train[i])]
# Repeat for test data
raw_dna_strands_test = [list(test_raw[1][i]) for i in range(len(test_raw))] # List of lists holding DNA strands separated by character. Size 8000 lists each of length 290
embedded_dna_strands_test = [np.column_stack((np.array(pd.get_dummies(pd.concat([pd.Series(raw_dna_strands_test[i]), pd.Series(["A", "C", "T", "G"])]), dtype='int'))[:-4], np.arange(295))) for i in range(len(test_raw))]
embedded_dna_strands_test = [embedded_dna_strands_test[i] for i in range(len(embedded_dna_strands_test)) if not ("N" in raw_dna_strands_test[i])]

# Add column with unique identifier for each nucleotide (sequence + location)
train_sol[3] = [str(train_sol.iloc[i, 1][5:]).zfill(4) + str(train_sol.iloc[i,2]).zfill(3) for i in range(len(train_sol))]

# Split by activators and repressors
train_sol_act = train_sol[train_sol[0] == 'A'][3]
train_sol_rep = train_sol[train_sol[0] == 'R'][3]

# Preprocess train data
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

# Split data into training and validation sets
embedded_dna_train, embedded_dna_val, scores_train, scores_val = train_test_split(
    embedded_dna_strands_train, train_scores, test_size=0.2, random_state=42
)


# train_scores are logits, need to apply a sigmoid transformation
train_scores = [1 / (1 + np.exp(-np.array(score))) for score in train_scores]

# Model
class DNADataset(Dataset):
    def __init__(self, embedded_dna_strands, train_scores):
        self.x = torch.tensor(embedded_dna_strands, dtype=torch.float32) # Convert x and y to tensors
        self.y = torch.tensor(train_scores, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
class SelfAttentionFeedForward(nn.Module):
    # Initialize hyperparameters and NN matrices
    def __init__(self, attention_size, seq_len, embed_size, hidden_size, hidden_layers, lr, train_len):
        super().__init__()
        self.attention_size = attention_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.train_len = train_len
        self.seq_len = seq_len
        # self.dropout_rate = dropout_rate 

        self.initAttention()
        self.initFFN()

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            amsgrad=True,
        )

    # Initialize our weight matrices as torch objects, allows them to be automatically optimized
    def initAttention(self):
        self.W_Q = nn.Linear(self.embed_size, self.attention_size, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.attention_size, bias=False)
        self.W_V = nn.Linear(self.embed_size, self.attention_size, bias=False)

        
    
    # Initialize Feed Forward layers, based on however many hidden layers we want
    def initFFN(self):

        layers = []

        layers.append(nn.Linear(self.attention_size, self.hidden_size))
        layers.append(nn.ReLU())
        # layers.append(nn.Dropout(self.dropout_rate))
        
        for _ in range(self.hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(self.dropout_rate)) #Add this later on

        layers.append(nn.Linear(self.hidden_size, 1))

        self.layers = nn.ModuleList(layers)
        self.criterion = nn.MSELoss() # Switch to mean squared error instead of simple norm (this is better apparently?)

    def loss(self, predicted, y):
        return torch.norm(predicted - y)

    def forward(self, x):
        # x of size                                               (batch_size, sequence_length, embedding_size)
        if x.shape[-1] != self.embed_size:
            raise ValueError

        queries = self.W_Q(x) #                                   (batch_size, sequence_length, attention_size)
        keys = self.W_K(x) #                                      (batch_size, sequence_length, attention_size)
        values = self.W_V(x) #                                    (batch_size, sequence_length, attention_size)

        # Scale to prevent overflow errors, divide by square root of attention dimension
        scale = torch.sqrt(torch.tensor(self.attention_size, dtype=torch.float32))

        #Compute attention and then normalize
        attention = torch.bmm(queries, keys.transpose(1,2)) / scale #                  (batch_size, seq_len, seq_len)
        weights = torch.nn.functional.softmax(attention, dim=2) # Apply this per sample

        # Use as weights for values
        context = torch.bmm(weights, values) #    (batch_size, attention_size, sequence_length)
        # Run through all FFN layers
        for layer in self.layers:
            context = layer(context)
        # Return prediction with added b term
        return context.squeeze(-1)

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        pred = self(x)
        loss = self.criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0) # This is for stability
        self.optimizer.step()
        return loss.item() # Diagnostic info

    def train(self, dataloader):
        losses = []
        for epoch in range(self.train_len):
            epoch_loss = 0
            for x_batch, y_batch in dataloader:
                loss = self.train_step(x_batch, y_batch)
                epoch_loss += loss
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.train_len}, Loss: {avg_loss:.4f}")
        return losses

dataset = DNADataset(embedded_dna_strands_train, train_scores)

# Create a DataLoader for batching, shuffling, and parallel data loading
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SelfAttentionFeedForward(20, 295, 5, 10, 1, 1e-4, 100) # (attention_size, seq_len, embed_size, hidden_size, hidden_layers, lr, train_len)
model.train(dataloader)

# Test prediction on a sample
pred = model.forward(torch.tensor(np.stack(embedded_dna_strands_test), dtype=torch.float32)).detach().numpy()

# The output should now reflect predictions based on the training data

# Generate predictions
predictions = []

for i, strand in enumerate(embedded_dna_strands_train):
    strand_tensor = torch.Tensor(strand).unsqueeze(0)  # Add batch dimension (1, seq_len, embed_size)
    logits = model.forward(strand_tensor).detach().numpy().flatten()  # Forward pass
    predicted_scores = torch.sigmoid(torch.Tensor(logits)).numpy().flatten()  # Apply sigmoid to logits
    sequence_id = f"train{str(i).zfill(4)}"
    nucleotides = raw_dna_strands_train[i]

    for pos, (nucleotide, score) in enumerate(zip(nucleotides, predicted_scores)):
        predictions.append(["A" if score > 0.5 else "R", score, sequence_id, str(pos).zfill(4)])


# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=["Activation/Repression", "Score", "Sequence_ID", "Nucleotide_Position"])

# Sort predictions by score (descending for activators, ascending for repressors)
top_activating = predictions_df[predictions_df["Activation/Repression"] == "A"].nlargest(100_000, "Score")
top_repressive = predictions_df[predictions_df["Activation/Repression"] == "R"].nsmallest(50_000, "Score")

# Combine the two sets
top_sequences = pd.concat([top_activating, top_repressive])

# Drop the "Score" column and format output
top_sequences = top_sequences[["Activation/Repression", "Sequence_ID", "Nucleotide_Position"]]

# Save to a TSV file
top_sequences.to_csv("top_sequences.tsv", sep="\t", header=False, index=False)

