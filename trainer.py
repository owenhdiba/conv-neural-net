import torch


class Trainer():
    '''
    Takes pytorch model and runs optimization on training set. Calculates 
    loss and error on validation data from test_loader every "test_period"
    number of iterations. History of losses and errors on training and 
    test dataset are stored as attributes.
    '''

    def __init__(self, model, device, optimizer, loss_fn, test_loss_fn,
                 train_loader, test_loader, epochs, test_period):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.test_loss_fn = test_loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.test_period = test_period
        self.losses = []
        self.errors = []
        self.test_losses = []
        self.test_errors = []
        self.train_size = len(train_loader.dataset)
        self.test_size = len(test_loader.dataset)

    def test(self):
        self.model.eval()
        loss = 0
        error = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss += self.test_loss_fn(outputs, labels).item()
                # get the index of the max log-probability
                pred = outputs.argmax(dim=1, keepdim=True)
                error += pred.ne(labels.view_as(pred)).sum().item()
        return loss / self.test_size, error / self.test_size

    def train_batch(self, inputs, labels):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)

        loss = self.loss_fn(outputs, labels)
        self.losses.append(loss.item())
        loss.backward()

        self.optimizer.step()
        # get the index of the max log-probability
        pred = outputs.argmax(dim=1, keepdim=True)
        error = pred.ne(labels.view_as(pred)).to(torch.float32).mean().item()
        self.errors.append(error)
        return loss

    def train_epoch(self, iteration):
        self.model.train(True)
        epoch_loss = 0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            batch_size = len(inputs)
            batch_loss = self.train_batch(inputs, labels)
            epoch_loss += batch_loss * batch_size
            if iteration % self.test_period == 0:
                test_loss, test_error = self.test()
                self.test_losses.append(test_loss)
                self.test_errors.append(test_error)
                self.model.train(True)
            iteration += 1
        epoch_loss /= self.train_size
        return iteration, epoch_loss

    def train(self):
        iteration = 1
        for epoch in range(1, self.epochs + 1):
            iteration, epoch_loss = self.train_epoch(iteration)
            fstring = f"Epoch {epoch} of {self.epochs}: loss = {epoch_loss:.3f}"
            print(fstring)