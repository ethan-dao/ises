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
        #self.dropout_rate = dropout_rate 

        self.initAttention()
        self.initFFN()

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            amsgrad=True,
        )

    #Initialize our weight matrices as torch objects, allows them to be automatically optimized
    def initAttention(self):
        self.W_Q = nn.Linear(self.embed_size, self.attention_size, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.attention_size, bias=False)
        self.W_V = nn.Linear(self.embed_size, self.attention_size, bias=False)

        self.layers 
        self.b = nn.Parameter(torch.rand(295)) # This is an addition term, analogous to y-intercept

        
    
    #Initialize Feed Forward layers, based on however many hidden layers we want
    def initFFN(self):

        layers = []

        layers.append(nn.Linear(self.attention_size, self.hidden_size))
        layers.append(nn.ReLU())
        #layers.append(nn.Dropout(self.dropout_rate))
        
        for _ in range(self.hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
            #layers.append(nn.Dropout(self.dropout_rate)) #Add this later on

        layers.append(nn.Linear(self.hidden_size, 1))

        self.layers = nn.ModuleList(layers)
        self.criterion = nn.MSELoss() # Swithc to mean squared error instead of simple norm (this is better apparently?)

    def loss(self, predicted, y):
        return torch.norm(predicted - y)

    def predict(self, x):
        # Multiply by all embeddings
        queries = self.W_Q(x)
        keys = self.W_K(x)
        values = self.W_V(x)
        #Compute attention and then normalize
        attention = torch.matmul(queries, keys.T)
        weights = torch.nn.functional.softmax(attention, dim=0)

        # Use as weights for values
        context = torch.matmul(weights, values).T #This is dim self.attention_size x 295
        # Run through all FFN layers
        for layer in self.layers:
            context = layer(context)
        # Return prediction with added b term
        return context + self.b

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        pred = self(x)
        loss = self.criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0) # This is for stability
        self.optimizer.step()
        return loss.item() # Diagnostic info

    def train(self, x ,y):
        self.train() # Put model into training mode
        losses = [] # We want to store this info to track training progress

        # Batching implementation
        torch.utils.data.TensorDataset(torch.Tensor(x), torch.Tensor(y))
        for epoch in range(self.train_len):
            print(i)
            self.train_step(torch.Tensor(x[i]), torch.Tensor(y[i]))

