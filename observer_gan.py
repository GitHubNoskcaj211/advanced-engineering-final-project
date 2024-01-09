import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import torch_utils

class FeedforwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, should_sigmoid: bool):
        super(FeedforwardNetwork, self).__init__()
        if should_sigmoid:
                self.input_layer = nn.Linear(input_size, hidden_size)
        else:
            self.input_layer = nn.utils.spectral_norm(nn.Linear(input_size, hidden_size))
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            if should_sigmoid:
                self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            else:
                self.hidden_layers.append(nn.utils.spectral_norm(nn.Linear(hidden_size, hidden_size)))
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.randomize()
        self.should_sigmoid = should_sigmoid

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
        x = self.output_layer(x)
        if self.should_sigmoid:
            x = self.sigmoid(x)
        return x
    
    def randomize(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(init_weights)

# Discrimnator
# Give unlabeled and generated (generator forward pass of random sample) samples. Train unlabeled to regress to 1 and generated to regress to 0.
# Understand differences between real and generated samples


# Generator
# Run the discrimnator on the generator of random sample. Train to have the discriminator regress to 1 and observer regress to 1
# (make generations more realistic wrt the discriminator and non-positive wrt the observer)

# Observer
# Give positive and generated (generator forward pass of random sample) samples. Train positive to regress to 0 and generated to regress to 1.
# Understand differences between positive and generated samples

# given N x ... tensor sample n_samples from axis 0
# if there are not enough return the full tensor
def sample_tensor(tensor, n_samples):
    random_indices = torch.randperm(tensor.size(0))[:n_samples]
    sampled_rows = tensor[random_indices]
    return sampled_rows

def random_normal(n_samples, D):
    return torch.randn((n_samples, D))

# All losses are BCE
# unlabeled_data - N x D
# positive_data - N x D
def train(learning_rate, n_epochs, unlabeled_data, positive_data, n_samples_per_epoch, hidden_size_discriminator, hidden_size_generator, hidden_size_observer, n_layers_discriminator, n_layers_generator, n_layers_observer):
    D = unlabeled_data.shape[1]
    discriminator = FeedforwardNetwork(D, hidden_size_discriminator, 1, n_layers_discriminator, True)
    generator = FeedforwardNetwork(D, hidden_size_generator, D, n_layers_generator, True)
    observer = FeedforwardNetwork(D, hidden_size_observer, 1, n_layers_observer, True)

    bce_loss = nn.BCELoss()

    optimizer_discriminator = optim.SGD(discriminator.parameters(), lr=learning_rate)
    optimizer_generator = optim.SGD(generator.parameters(), lr=learning_rate)
    optimizer_observer = optim.SGD(observer.parameters(), lr=learning_rate)

    discriminator_losses = []
    generator_losses = []
    observer_losses = []

    if n_epochs == 0:
        return discriminator, generator, observer, discriminator_losses, generator_losses, observer_losses

    for epoch in tqdm(range(n_epochs), desc='Epoch'):
        n_epochs_reset = 2000
        reset_observer = 0
        reset_discriminator = int(n_epochs_reset / 3)
        reset_generator = int(n_epochs_reset / 3 * 2)
        if epoch < n_epochs - n_epochs_reset:
            if epoch % n_epochs_reset == reset_observer:
                observer.randomize()
            # if epoch % n_epochs_reset == reset_discriminator:
            #     discriminator.randomize()
            # if epoch % n_epochs_reset == reset_generator:
            #     generator.randomize()
        observer_quality = 1 if epoch >= n_epochs - n_epochs_reset else ((epoch - reset_observer) % n_epochs_reset) / n_epochs_reset
        discriminator_quality = 1#1 if epoch >= n_epochs - n_epochs_reset else ((epoch - reset_discriminator) % n_epochs_reset) / n_epochs_reset
        generator_quality = 1#1 if epoch >= n_epochs - n_epochs_reset else ((epoch - reset_generator) % n_epochs_reset) / n_epochs_reset
        unlabeled_sample = sample_tensor(unlabeled_data, n_samples_per_epoch)
        positive_sample = sample_tensor(positive_data, n_samples_per_epoch)
        normal_sample = random_normal(n_samples_per_epoch, D)
        random_sample = generator(normal_sample)

        optimizer_discriminator.zero_grad()
        loss_discriminator = bce_loss(discriminator(unlabeled_sample), torch.full((unlabeled_sample.shape[0], 1), 1, dtype=torch.float32)) + generator_quality * bce_loss(discriminator(random_sample.detach()), torch.full((random_sample.shape[0], 1), 0, dtype=torch.float32))
        loss_discriminator.backward()
        optimizer_discriminator.step()

        optimizer_observer.zero_grad()        
        loss_observer = bce_loss(observer(positive_sample), torch.full((positive_sample.shape[0], 1), 1, dtype=torch.float32)) + generator_quality * bce_loss(observer(random_sample.detach()), torch.full((random_sample.shape[0], 1), 0, dtype=torch.float32))
        loss_observer.backward()
        optimizer_observer.step()

        optimizer_generator.zero_grad()
        loss_generator = discriminator_quality * bce_loss(discriminator(random_sample), torch.full((random_sample.shape[0], 1), 1, dtype=torch.float32)) + observer_quality * bce_loss(observer(random_sample), torch.full((random_sample.shape[0], 1), 0, dtype=torch.float32))
        loss_generator.backward()
        optimizer_generator.step()

        discriminator_losses.append(loss_discriminator.item())
        generator_losses.append(loss_observer.item())
        observer_losses.append(loss_generator.item())

    
    return discriminator, generator, observer, discriminator_losses, generator_losses, observer_losses

def train_wgan(learning_rate, n_epochs, unlabeled_data, positive_data, n_samples_per_epoch, hidden_size_discriminator, hidden_size_generator, hidden_size_observer, n_layers_discriminator, n_layers_generator, n_layers_observer):
    D = unlabeled_data.shape[1]
    discriminator = FeedforwardNetwork(D, hidden_size_discriminator, 1, n_layers_discriminator, False)
    generator = FeedforwardNetwork(D, hidden_size_generator, D, n_layers_generator, True)
    observer = FeedforwardNetwork(D, hidden_size_observer, 1, n_layers_observer, False)

    optimizer_discriminator = optim.SGD(discriminator.parameters(), lr=learning_rate)
    optimizer_generator = optim.SGD(generator.parameters(), lr=learning_rate)
    optimizer_observer = optim.SGD(observer.parameters(), lr=learning_rate)

    discriminator_losses = []
    generator_losses = []
    observer_losses = []

    C = 1

    for epoch in tqdm(range(n_epochs), desc='Epoch'):
        n_epochs_reset = 3000
        reset_observer = 0
        reset_discriminator = int(n_epochs_reset / 3)
        reset_generator = int(n_epochs_reset / 3 * 2)
        if epoch < n_epochs - n_epochs_reset:
            if epoch % n_epochs_reset == reset_observer:
                observer.randomize()
            if epoch % n_epochs_reset == reset_discriminator:
                discriminator.randomize()
            if epoch % n_epochs_reset == reset_generator:
                generator.randomize()
        observer_quality = ((epoch - reset_observer) % n_epochs_reset) / n_epochs_reset
        generator_quality = ((epoch - reset_generator) % n_epochs_reset) / n_epochs_reset
        discriminator_quality = ((epoch - reset_discriminator) % n_epochs_reset) / n_epochs_reset
        unlabeled_sample = sample_tensor(unlabeled_data, n_samples_per_epoch)
        positive_sample = sample_tensor(positive_data, n_samples_per_epoch)
        normal_sample = random_normal(n_samples_per_epoch, D)
        random_sample = generator(normal_sample)

        optimizer_discriminator.zero_grad()
        loss_discriminator = -torch.mean(discriminator(unlabeled_sample)) + generator_quality * torch.mean(discriminator(random_sample.detach()))
        loss_discriminator.backward()
        optimizer_discriminator.step()

        for p in discriminator.parameters():
            p.data.clamp_(-C, C)

        optimizer_observer.zero_grad()
        loss_observer = -torch.mean(observer(positive_sample)) + generator_quality * torch.mean(observer(random_sample.detach()))
        loss_observer.backward()
        optimizer_observer.step()

        for p in observer.parameters():
            p.data.clamp_(-C, C)

        optimizer_generator.zero_grad()
        loss_generator = -discriminator_quality * torch.mean(discriminator(random_sample)) + observer_quality * torch.mean(observer(random_sample))
        loss_generator.backward()
        optimizer_generator.step()

        discriminator_losses.append(loss_discriminator.item())
        generator_losses.append(loss_observer.item())
        observer_losses.append(loss_generator.item())

    
    return discriminator, generator, observer, discriminator_losses, generator_losses, observer_losses

def save_models(discriminator, generator, observer, file):
    torch.save(discriminator.state_dict(), f'models/{file}_discriminator.pth')
    torch.save(generator.state_dict(), f'models/{file}_generator.pth')
    torch.save(observer.state_dict(), f'models/{file}_observer.pth')

def load_models(discriminator, generator, observer, file):
    discriminator.load_state_dict(torch.load(f'models/{file}_discriminator.pth'))
    generator.load_state_dict(torch.load(f'models/{file}_generator.pth'))
    observer.load_state_dict(torch.load(f'models/{file}_observer.pth'))

def random_model(input):
    return np.random.random(input.shape[0])