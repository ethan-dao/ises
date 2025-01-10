import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import math

# 
# 1. Helper Functions
# 
def parse_jaspar_file(filepath):
    """
    Parses a JASPAR file to extract PWMs.
    :param filepath: Path to the JASPAR file.
    :return: Dictionary with TF IDs, names, and PWMs.
    """
    pwm_dict = {}
    with open(filepath, 'r') as file:
        lines = file.readlines()

    current_tf = None
    pwm_data = None
    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if current_tf and pwm_data:
                pwm_dict[current_tf] = {"name": tf_name, "pwm": np.array(pwm_data)}
            parts = line.split("\t")
            current_tf = parts[0][1:]  # e.g., "MA0004.1"
            tf_name = parts[1] if len(parts) > 1 else None
            pwm_data = []
        elif line.startswith(("A", "C", "G", "T")):
            counts = list(map(int, line.split("[")[1].split("]")[0].split()))
            pwm_data.append(counts)
    if current_tf and pwm_data:
        pwm_dict[current_tf] = {"name": tf_name, "pwm": np.array(pwm_data)}
    return pwm_dict

def normalize_pwm(pwm):
    """Normalize a PWM to probabilities."""
    column_sums = pwm.sum(axis=0, keepdims=True)
    column_sums[column_sums == 0] = 1  # To avoid division by zero
    return pwm / column_sums

def pwm_scan(sequence, pwm):
    """
    Scans a DNA sequence using a PWM.
    :param sequence: One-hot encoded DNA sequence (seq_len x 4).
    :param pwm: PWM (4 x pwm_len).
    :return: Binding scores.
    """
    seq_len, _ = sequence.shape
    pwm_len = pwm.shape[1]
    scores = [
        (sequence[i:i + pwm_len] * pwm).sum()
        for i in range(seq_len - pwm_len + 1)
    ]
    scores = np.pad(scores, (0, seq_len - len(scores)), mode="constant")
    return np.array(scores)

# Convert DNA strands to one-hot encoding
def convert_to_one_hot(dna_strands, strand_length):
    one_hot_encoded = []
    for strand in dna_strands:
        one_hot = np.zeros((4, strand_length))
        for i, base in enumerate(strand):
            if base == "A":
                one_hot[0, i] = 1
            elif base == "C":
                one_hot[1, i] = 1
            elif base == "G":
                one_hot[2, i] = 1
            elif base == "T":
                one_hot[3, i] = 1
        one_hot_encoded.append(one_hot)
    return one_hot_encoded

one_hot_dna_strands_train = convert_to_one_hot(raw_dna_strands_train, strand_length)
one_hot_dna_strands_test = convert_to_one_hot(raw_dna_strands_test, strand_length)


def compute_pwm_features_optimized(dna_sequences, pwms):
    """
    Optimized computation of PWM scores for all sequences and PWMs.
    :param dna_sequences: List of one-hot encoded DNA sequences (shape: (4, seq_len)).
    :param pwms: Dictionary of PWMs (shape: (4, pwm_width)).
    :return: List of feature arrays.
    """
    pwm_features = []
    for seq in dna_sequences:
        seq_len = seq.shape[1]
        features = []
        for pwm in pwms.values():
            pwm_array = pwm["pwm"]
            pwm_width = pwm_array.shape[1]
            
            # Generate a rolling window of slices for the sequence
            seq_windowed = np.lib.stride_tricks.sliding_window_view(seq, window_shape=(4, pwm_width), axis=(0, 1))
            
            # Compute dot product between the sequence window and the PWM
            scores = np.einsum('ijk,ij->k', seq_windowed, pwm_array)
            
            # Pad scores to match sequence length
            scores = np.pad(scores, (0, seq_len - len(scores)), mode="constant")
            features.append(scores)
        pwm_features.append(np.stack(features, axis=1))  # Stack scores for all PWMs
    return pwm_features


def combine_features(dna_embeddings, pwm_features):
    """
    Combines DNA embeddings with PWM features.
    :param dna_embeddings: List of DNA embeddings.
    :param pwm_features: List of PWM feature arrays.
    :return: Combined features.
    """
    return [np.hstack([embedding, pwm_feat]) for embedding, pwm_feat in zip(dna_embeddings, pwm_features)]

# 
# 2. Parse JASPAR File
# 
jaspar_filepath = '/Users/irisgu/Downloads/MPRA_Challenge/ises/data/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt'
pwm_dict = parse_jaspar_file(jaspar_filepath)

# Normalize PWMs
for key in pwm_dict:
    pwm_dict[key]["pwm"] = normalize_pwm(pwm_dict[key]["pwm"])

# 
# 3. Preprocess DNA Data
# 
train_raw = pd.read_csv('/Users/irisgu/VisualStudioCode/ises/data/train_MPRA.txt', delimiter='\t', header=None)
test_raw = pd.read_csv('/Users/irisgu/VisualStudioCode/ises/data/test_MPRA.txt', delimiter='\t', header=None)
train_sol = pd.read_csv('/Users/irisgu/VisualStudioCode/ises/data/trainsolutions.txt', delimiter='\t', header=None)

strand_length = 295

# Process train data
train_scores = np.array(train_raw.iloc[:, 2:297])
raw_dna_strands_train = [list(train_raw[1][i]) for i in range(len(train_raw))]
embedded_dna_strands_train = [
    np.column_stack(
        (np.array(pd.get_dummies(pd.concat([pd.Series(raw_dna_strands_train[i]), pd.Series(["A", "C", "T", "G"])]), dtype="int"))[:-4],
         np.arange(strand_length))
    )
    for i in range(len(train_raw))
]
embedded_dna_strands_train = [
    embedded_dna_strands_train[i] for i in range(len(embedded_dna_strands_train)) if not ("N" in raw_dna_strands_train[i])
]
train_scores = [train_scores[i] for i in range(len(raw_dna_strands_train)) if not ("N" in raw_dna_strands_train[i])]

# Process test data
raw_dna_strands_test = [list(test_raw[1][i]) for i in range(len(test_raw))]
embedded_dna_strands_test = [
    np.column_stack(
        (np.array(pd.get_dummies(pd.concat([pd.Series(raw_dna_strands_test[i]), pd.Series(["A", "C", "T", "G"])]), dtype="int"))[:-4],
         np.arange(strand_length))
    )
    for i in range(len(test_raw))
]
embedded_dna_strands_test = [
    embedded_dna_strands_test[i] for i in range(len(embedded_dna_strands_test)) if not ("N" in raw_dna_strands_test[i])
]


# Compute PWM features
pwm_features_train = compute_pwm_features_optimized(one_hot_dna_strands_train, pwm_dict)
pwm_features_train = compute_pwm_features_optimized(one_hot_dna_strands_test, pwm_dict)


# Combine features
combined_train_data = combine_features(embedded_dna_strands_train, pwm_features_train)
combined_test_data = combine_features(embedded_dna_strands_test, pwm_features_test)


# Split data into train and validation sets
embedded_dna_train, embedded_dna_val, scores_train, scores_val = train_test_split(
    combined_train_data, train_scores, test_size=0.2, random_state=42
)

# Apply sigmoid transformation to train scores
train_scores = [1 / (1 + np.exp(-np.array(score))) for score in train_scores]

# 
# 4. Dataset and Model
# 
class CombinedDNADataset(Dataset):
    def __init__(self, combined_features, train_scores):
        self.x = torch.tensor(combined_features, dtype=torch.float32)
        self.y = torch.tensor(train_scores, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class SelfAttentionFeedForward(nn.Module):
    def __init__(self, attention_size, seq_len, embed_size, hidden_size, hidden_layers, lr, train_len):
        super().__init__()
        self.attention_size = attention_size
        self.embed_size = embed_size

        self.W_Q = nn.Linear(embed_size, attention_size, bias=False)
        self.W_K = nn.Linear(embed_size, attention_size, bias=False)
        self.W_V = nn.Linear(embed_size, attention_size, bias=False)

        layers = [nn.Linear(attention_size, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, 1))
        self.layers = nn.Sequential(*layers)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, amsgrad=True)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        queries = self.W_Q(x)
        keys = self.W_K(x)
        values = self.W_V(x)
        scale = math.sqrt(self.attention_size)
        attention = torch.bmm(queries, keys.transpose(1, 2)) / scale
        weights = torch.nn.functional.softmax(attention, dim=2)
        context = torch.bmm(weights, values)
        for layer in self.layers:
            context = layer(context)
        return context.squeeze(-1)

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        pred = self(x)
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

# 
# 5. Training the Model
# 
train_dataset = CombinedDNADataset(embedded_dna_train, scores_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

embed_size = 5 + len(pwm_dict)  # Original embedding size + PWM features
model = SelfAttentionFeedForward(
    attention_size=20, seq_len=strand_length, embed_size=embed_size,
    hidden_size=10, hidden_layers=1, lr=1e-4, train_len=100
)

for epoch in range(10):
    for x_batch, y_batch in train_loader:
        loss = model.train_step(x_batch, y_batch)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")


# The output should now reflect predictions based on the training data

#
# 6. Generate predictions
#
predictions = []

for i, strand in enumerate(embedded_dna_strands_test):
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