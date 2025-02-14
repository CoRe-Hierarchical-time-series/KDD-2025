

class Config: 
    def __init__(self):
        batch_size = 32
        context_window = 5
        train_split = 0.8
        val_split = 0.1
        hidden_dim = 128

        num_runs = 1
        n_epochs = 1000