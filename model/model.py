import torch
import torch.nn as nn


class SelfAttentionFeedForward(nn.Module):
    #Initialize hyperparameters and NN matrices
    def __init__(self, attention_size, embed_size, hidden_size, hidden_layers, lr, train_len):
        super().__init__()
        self.attention_size = attention_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.train_len = train_len

        self.initAttention()
        self.initFFN()

    #Initialize our weight matrices as torch objects, allows them to be automatically optimized
    def initAttention(self):
        self.W_Q = nn.Parameter(torch.rand(self.attention_size, self.embed_size))
        self.W_K = nn.Parameter(torch.rand(self.attention_size, self.embed_size))
        self.W_V = nn.Parameter(torch.rand(self.attention_size, self.embed_size))
        self.b = nn.Parameter(torch.rand(295)) # This is an addition term, analogous to y-intercept

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            amsgrad=True,
        )
    
    #Initialize Feed Forward layers, based on however many hidden layers we want
    def initFFN(self):
        fc1 = nn.Linear(self.attention_size, self.hidden_size)
        relu1 = nn.ReLU()
        self.layers = [fc1, relu1]
        for i in range(self.hidden_layers - 1):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.hidden_size, 1))


    def loss(self, predicted, y):
        return torch.norm(predicted - y)

    def predict(self, x):
        # Multiply by all embeddings
        queries = torch.matmul(self.W_Q, x.T)
        keys = torch.matmul(self.W_K, x.T)
        values = torch.matmul(self.W_V, x.T)

        #Compute attention and then normalize
        attention = torch.matmul(queries, keys.T)
        weights = torch.nn.functional.softmax(attention, dim=0)

        # Use as weights for values
        contextualized_embeddings = torch.matmul(weights, values).T #This is dim self.attention_size x 295
        # Run through all FFN layers
        for layer in self.layers:
            contextualized_embeddings = layer(contextualized_embeddings)
        # Return prediction with added b term
        return contextualized_embeddings + self.b

    def train_step(self, x, y):
        pred = self.predict(x)
        loss = self.loss(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return None

    def train(self, x ,y):
        for i in range(self.train_len):
            print(i)
            self.train_step(torch.Tensor(x[i]), torch.Tensor(y[i]))

