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

    def forward(self, x):
        # x of size                                               (batch_size, sequence_length, embedding_size)
        batch_size, seq_len, emb_size = x.shape
        if emb_size != self.embed_size:
            raise ValueError

        queries = self.W_Q(x) #                                   (batch_size, sequence_length, attention_size)
        keys = self.W_K(x) #                                      (batch_size, sequence_length, attention_size)
        values = self.W_V(x) #                                    (batch_size, sequence_length, attention_size)

        #Compute attention and then normalize
        attention = torch.bmm(queries.transpose(1,2), keys) #                  (batch_size, attention_size, attention_size)
        weights = torch.nn.functional.softmax(attention, dim=2) # Apply this per sample

        # Use as weights for values
        context = torch.bmm(weights, values.transpose(1,2)).transpose(1,2) #    (batch_size, attention_size, sequence_length)
        # Run through all FFN layers
        for layer in self.layers:
            context = layer(context)
        # Return prediction with added b term
        return context

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        pred = self(x)
        loss = self.criterion(pred.squeeze(-1), y)
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
            
