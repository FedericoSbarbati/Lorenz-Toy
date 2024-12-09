import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import random_split
from sklearn.metrics import mean_absolute_error
import os
import math


class EmbeddedDataset(Dataset):
    def __init__(self, embedding_y1, embedding_y2_1, embedding_y2_2):
        """
        Dataset per embeddings temporali.

        Parametri:
        - embedding_y1: Array numpy o torch tensor per y1.
        - embedding_y2_1: Array numpy o torch tensor per y2_1.
        - embedding_y2_2: Array numpy o torch tensor per y2_2.
        """
        # Assicurati che i dati siano convertiti in torch tensor float32
        self.embedding_y1 = torch.tensor(embedding_y1, dtype=torch.float32)
        self.embedding_y2_1 = torch.tensor(embedding_y2_1, dtype=torch.float32)
        self.embedding_y2_2 = torch.tensor(embedding_y2_2, dtype=torch.float32)

    def __len__(self):
        # La lunghezza del dataset è quella di embedding_y1
        return len(self.embedding_y1)

    def __getitem__(self, idx):
        """
        Restituisce:
        - input: embedding_y1 (input della rete)
        - target: concatenazione di embedding_y2_1 e embedding_y2_2
        """
        input_data = self.embedding_y1[idx]
        target_data = torch.cat((self.embedding_y2_1[idx], self.embedding_y2_2[idx]), dim=0)
        return input_data, target_data

    
# DEFINITION OF THE VAE MODEL
class VAE(torch.nn.Module):
    def __init__(self, encoder_layers, decoder_layers):
        """
        Costruisce un VAE con encoder, spazio latente (mu e log_var), e decoder.

        Parameters:
            encoder_layers (list): Lista delle dimensioni dei layer dell'encoder.
            decoder_layers (list): Lista delle dimensioni dei layer del decoder.
        """
        super().__init__()
        
        # Encoder
        self.encoder_layers = []
        for i in range(len(encoder_layers) - 1):
            self.encoder_layers.append(torch.nn.Linear(encoder_layers[i], encoder_layers[i + 1]))
            if i < len(encoder_layers) - 2:  # Attivazione solo per i layer nascosti
                self.encoder_layers.append(torch.nn.LeakyReLU())
        self.encoder = torch.nn.Sequential(*self.encoder_layers)
        
        # Latent space: mean and log variance
        latent_dim = encoder_layers[-1]
        self.fc_mu = torch.nn.Linear(latent_dim, latent_dim)
        self.fc_log_var = torch.nn.Linear(latent_dim, latent_dim)
        
        # Decoder
        self.decoder_layers = []
        for i in range(len(decoder_layers) - 1):
            output_dim = decoder_layers[i + 1]
            self.decoder_layers.append(torch.nn.Linear(decoder_layers[i], output_dim))
            if i < len(decoder_layers) - 2:  # Attivazione solo per i layer nascosti
                self.decoder_layers.append(torch.nn.LeakyReLU())
        self.decoder_layers.append(torch.nn.Sigmoid())  #Attivazione per l'ultimo layer
        self.decoder = torch.nn.Sequential(*self.decoder_layers)


    def forward(self, x):
        """
        Esegue il passaggio forward del modello.

        Parameters:
            x (torch.Tensor): Input.

        Returns:
            decoded (torch.Tensor): Output ricostruito.
            mu (torch.Tensor): Media dello spazio latente.
            log_var (torch.Tensor): Log-varianza dello spazio latente.
        """
        # Encoder
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        
        # Latent space
        z = self.reparameterize(mu, log_var)
        
        # Decoder
        decoded = self.decoder(z)
        return decoded, mu, log_var




    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return decoded, mu, log_var
    




# AUXILIARY FUNCTIONS FOR TRAINING AND EVALUATION
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, verbose=False):
        """
        Implementa Early Stopping per interrompere il training.
        
        Parametri:
        - patience: numero di epoche senza miglioramenti dopo cui fermare il training.
        - delta: minimo miglioramento richiesto per considerare una variazione significativa.
        - verbose: se True, stampa messaggi sullo stato dell'early stopping.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.stopped_epoch = 0

    def __call__(self, val_loss, epoch):
        """
        Controlla se fermare il training.
        
        Parametri:
        - val_loss: perdita di validazione dell'epoca corrente.
        - epoch: epoca corrente.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience} epochs without improvement.")
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch

def get_scheduler(optimizer, scheduler_config):
    """
    Crea uno scheduler in base alla configurazione.

    Parametri:
    - optimizer: ottimizzatore associato.
    - scheduler_config: configurazione dello scheduler (dizionario con "type" e "params").

    Ritorna:
    - Istanza dello scheduler.
    """
    if scheduler_config is None:
        raise ValueError("Scheduler configuration is missing.")

    scheduler_type = scheduler_config.get("type")
    scheduler_params = scheduler_config.get("params", {})

    if scheduler_type == "StepLR":
        return lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_type == "ExponentialLR":
        return lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
    elif scheduler_type == "ReduceLROnPlateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")




def get_beta(epoch, warmup_epochs, method="constant", beta_value=1.0):
    """
    Calcola il valore di beta in base al metodo specificato.

    Parametri:
    - epoch: epoca corrente.
    - warmup_epochs: numero di epoche di warm-up.
    - method: metodo per calcolare beta (constant, sigmoid, linear).
    - beta_value: valore costante di beta (usato per il metodo "constant").
    
    Ritorna:
    - Valore di beta.
    """
    if method == "constant":
        return beta_value
    elif method == "sigmoid":
        return 1 / (1 + math.exp(-0.1 * (epoch - warmup_epochs // 2)))
    elif method == "linear":
        return min(1.0, epoch / warmup_epochs)
    else:
        raise ValueError(f"Metodo di beta sconosciuto: {method}")
    
def plot_results(train_losses, val_losses, recon_losses, kld_losses, beta_values):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot KLD loss vs reconstruction loss
    axes[0].plot(kld_losses, label='KLD Loss', color='blue')
    axes[0].plot(recon_losses, label='Reconstruction Loss', color='red')
    axes[0].set_title('KLD Loss vs Reconstruction Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot training loss and validation loss
    axes[1].plot(train_losses, label='Training Loss', color='green')
    axes[1].plot(val_losses, label='Validation Loss', color='orange')
    axes[1].set_title('Training Loss vs Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    # Plot beta values
    axes[2].plot(beta_values, label='Beta', color='purple')
    axes[2].set_title('Beta Values over Epochs')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Beta')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def loss_function_vae(recon_x, x, mu, log_var, beta):
    # Ricostruzione loss
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    # Kullback-Leibler divergence
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kld_loss, recon_loss, kld_loss

def reconstruct_z2fromz1(epochs, train_loader, val_loader, model, optimizer, scheduler, scheduler_config, kl_annealing_epochs, beta_method="constant", beta_value=1.0, early_stopping_params=None):
    # Early stopping
    early_stopping = EarlyStopping(**early_stopping_params) if early_stopping_params else None

    # Liste per memorizzare le perdite
    train_losses = []
    val_losses = []
    recon_losses = []
    kld_losses = []
    beta_values = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        kld_loss_epoch = 0
        recon_loss_epoch = 0

        # Calcola il valore di beta
        beta = get_beta(epoch, kl_annealing_epochs, method=beta_method, beta_value=beta_value)
        beta_values.append(beta)

        for batch in train_loader:
            y1, y2 = batch  # Decomponi input (z1) e target (z2)
            y1 = y1.float()
            y2 = y2.float()
            optimizer.zero_grad()

            # Calcola l'output della rete
            recon_y2, mu, log_var = model(y1)

            # Calcola la perdita basata su z2 (y)
            loss, recon_loss, kld_loss = loss_function_vae(recon_y2, y2, mu, log_var, beta)

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            kld_loss_epoch += kld_loss.item()
            recon_loss_epoch += recon_loss.item()

        epoch_loss /= len(train_loader.dataset)
        kld_loss_epoch /= len(train_loader.dataset)
        recon_loss_epoch /= len(train_loader.dataset)

        train_losses.append(epoch_loss)
        recon_losses.append(recon_loss_epoch)
        kld_losses.append(kld_loss_epoch)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                y1, y2 = batch  # Decomponi input (z1) e target (z2)
                y1 = y1.float()
                y2 = y2.float()
                recon_y2, mu, log_var = model(y1)
                loss, _, _  = loss_function_vae(recon_y2, y2, mu, log_var, beta)
                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if scheduler_config["type"] == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Controllo Early Stopping
        if early_stopping:
            early_stopping(val_loss, epoch)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Stampa la perdita media dell'epoca
        print(f'Epoch [{epoch+1}/{epochs}], Rec Loss: {recon_loss_epoch:.4f}, KLD Loss: {kld_loss_epoch:.4f}, '
              f'Beta: {beta:.4f}, Training Loss: {epoch_loss:.4f}')

    # Plot dei risultati
    plot_results(train_losses, val_losses, recon_losses, kld_losses, beta_values)

    return train_losses, val_losses, recon_losses, kld_losses, beta_values, early_stopping.stopped_epoch



# METHODS TO SAVE AND LOAD OPTIMIZER STATE

def save_model(model, optimizer, train_data, val_data, epoch, stopped_epoch, encoder_layers, decoder_layers, train_losses, val_losses, recon_losses, kld_losses, beta_values, output_folder="Models", model_name="autoencoder_model.pth"):
    """
    Salva il modello, lo stato dell'ottimizzatore e i dati di training/validazione.

    Parametri:
    - model: modello PyTorch da salvare.
    - optimizer: ottimizzatore associato al modello.
    - train_data: dati di training.
    - val_data: dati di validazione.
    - epoch: epoca corrente al momento del salvataggio.
    - stopped_epoch: epoca in cui l'early stopping ha fermato il training.
    - layer_dims: dimensioni dei layer del modello.
    - train_losses, val_losses, recon_losses, kld_losses, entropies, beta_values: metriche del training.
    - output_folder: cartella di destinazione per il salvataggio.
    - model_name: nome del file del modello.
    """
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, model_name)

    torch.save({
        'epoch': epoch,
        'stopped_epoch': stopped_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_data': train_data,
        'val_data': val_data,
        'encoder_layers': encoder_layers,
        'decoder_layers': decoder_layers,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'recon_losses': recon_losses,
        'kld_losses': kld_losses,
        'beta_values': beta_values
    }, file_path)

    print(f"Model saved to {file_path}")

def load_model(file_path):
    """
    Carica un modello salvato con tutti i parametri e i dati associati.

    Parametri:
    - file_path: percorso al file salvato.

    Ritorna:
    - Tuple contenente il modello, l'ottimizzatore, i dati e i parametri salvati.
    """
    checkpoint = torch.load(file_path)
    encoder_layers = checkpoint['encoder_layers']
    decoder_layers = checkpoint['decoder_layers']
    
    # Ricrea l'istanza del modello
    model = VAE(encoder_layers, decoder_layers)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Ricrea l'ottimizzatore
    optimizer = torch.optim.Adam(model.parameters())  # Cambia se usi un altro ottimizzatore
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Carica altri dati
    train_data = checkpoint['train_data']
    val_data = checkpoint['val_data']
    epoch = checkpoint['epoch']
    stopped_epoch = checkpoint.get('stopped_epoch', None)
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    recon_losses = checkpoint['recon_losses']
    kld_losses = checkpoint['kld_losses']
    beta_values = checkpoint['beta_values']

    return (model, optimizer, train_data, val_data, epoch, stopped_epoch,
            train_losses, val_losses, recon_losses, kld_losses, beta_values)

import torch
import numpy as np

def run_model_and_collect_outputs(model, dataloader, device):
    """
    Esegue il modello su un DataLoader e restituisce gli output, mean e logvar, 
    e i dati originali in forma di array numpy.

    Parametri:
    - model: Modello PyTorch da eseguire
    - dataloader: DataLoader con i dati di input
    - device: Dispositivo (CPU o GPU)

    Ritorna:
    - true_data: Array numpy con i dati originali
    - predicted_data: Array numpy con gli output del modello
    - mean_vectors: Array numpy con i vettori mean
    - logvar_vectors: Array numpy con i vettori logvar
    """
    model.eval()  # Modalità eval
    true_data = []
    predicted_data = []
    mean_vectors = []
    logvar_vectors = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            
            # Inferenza del modello
            outputs, mean, logvar = model(inputs)  # Supponendo che il modello restituisca anche mean e logvar
            
            # Aggiungi i risultati a liste
            true_data.append(inputs.cpu().numpy())
            predicted_data.append(outputs.cpu().numpy())
            mean_vectors.append(mean.cpu().numpy())
            logvar_vectors.append(logvar.cpu().numpy())

    # Concatena tutti i batch in array numpy
    true_data = np.concatenate(true_data, axis=0)
    predicted_data = np.concatenate(predicted_data, axis=0)
    mean_vectors = np.concatenate(mean_vectors, axis=0)
    logvar_vectors = np.concatenate(logvar_vectors, axis=0)

    return true_data, predicted_data, mean_vectors, logvar_vectors


