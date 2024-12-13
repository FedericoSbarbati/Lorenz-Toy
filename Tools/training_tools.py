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




def get_beta(epoch, warmup_epochs, method="constant", beta_value=1.0, decay_start=100, decay_epochs=50):
    """
    Calcola il valore di beta in base al metodo specificato.

    Parametri:
    - epoch: epoca corrente.
    - warmup_epochs: numero di epoche di warm-up.
    - method: metodo per calcolare beta (constant, sigmoid, linear, linear_decay, exponential_decay).
    - beta_value: valore costante di beta (usato per il metodo "constant").
    - decay_start: epoca in cui iniziare la diminuzione di beta.
    - decay_epochs: numero di epoche su cui effettuare la diminuzione (lineare/esponenziale).
    
    Ritorna:
    - Valore di beta.
    """
    if method == "constant":
        return beta_value

    elif method == "sigmoid":
        return beta_value / (1 + math.exp(-0.1 * (epoch - warmup_epochs // 2)))

    elif method == "linear":
        return min(1.0, epoch / warmup_epochs)

    elif method == "linear_decay":
        # Warm-up fino a warmup_epochs, poi diminuzione lineare
        if epoch <= warmup_epochs:
            return min(1.0, epoch / warmup_epochs)
        elif epoch > decay_start:
            return max(0.0, 1.0 - (epoch - decay_start) / decay_epochs)
        else:
            return 1.0

    elif method == "exponential_decay":
        # Warm-up fino a warmup_epochs, poi diminuzione esponenziale
        if epoch <= warmup_epochs:
            return min(1.0, epoch / warmup_epochs)
        elif epoch > decay_start:
            decay_rate = decay_epochs / 5  # Fattore di scala per il decadimento
            return max(0.0, 1.0 * math.exp(-float(epoch - decay_start) / decay_rate))
        else:
            return 1.0

    else:
        raise ValueError(f"Metodo di beta sconosciuto: {method}")

    
def plot_results(train_losses, val_losses, recon_losses, kld_losses,effective_kld_losses, beta_values, gradient_history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot KLD loss vs reconstruction loss
    axes[0].plot(effective_kld_losses, label=f'KLD Loss * beta ', color='blue')
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


    plt.figure(figsize=(10, 6))
    plt.plot(kld_losses, label="KLD Loss")
    plt.xlabel("Epoch")
    plt.ylabel("KLD Loss")
    plt.title("KLD Loss Evolution During Training")
    plt.legend()
    plt.grid()
    plt.show()

    # Carica i gradienti salvati
    epochs = range(len(gradient_history["encoder"]))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, gradient_history["encoder"], label="Encoder Gradient Norm")
    plt.plot(epochs, gradient_history["decoder"], label="Decoder Gradient Norm")
    plt.plot(epochs, gradient_history["latent"], label="Latent Gradient Norm")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm Evolution During Training")
    plt.legend()
    plt.grid()
    plt.show()

def plot_Decoder_results(train_losses, val_losses, gradient_history):
    # Plot training loss and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='green')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    # Carica i gradienti salvati
    epochs = range(len(gradient_history["decoder"]))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, gradient_history["decoder"], label="Decoder Gradient Norm")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm Evolution During Training")
    plt.legend()
    plt.grid()
    plt.show()


def loss_function_vae(recon_x, x, mu, log_var, beta):
    # Ricostruzione loss
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    # Kullback-Leibler divergence
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kld_loss, recon_loss, kld_loss


def visualize_latent_space_with_pca(model, dataloader, device='cpu', n_components=2):
    model.eval()
    model.to(device)
    
    latent_mu = []
    true_labels = []

    # Estrazione dello spazio latente
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            _, mu, _ = model(inputs)  # Ottieni solo μ dallo spazio latente
            latent_mu.append(mu.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    # Concatena correttamente tutti i batch
    latent_mu = np.vstack(latent_mu)
    true_labels = np.concatenate(true_labels)[:len(latent_mu)]  # Allinea la dimensione a latent_mu

    # Applica la PCA per ridurre lo spazio latente
    pca = PCA(n_components=n_components)
    latent_pca = pca.fit_transform(latent_mu)

    # Visualizza il risultato
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latent_pca[:, 0], latent_pca[:, 1], 
                               c=true_labels[:len(latent_pca)], cmap='viridis', s=10)
        plt.colorbar(scatter, label='Class Labels')
        plt.title('Latent Space Visualization with PCA (2D)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(latent_pca[:, 0], latent_pca[:, 1], latent_pca[:, 2],
                              c=true_labels[:len(latent_pca)], cmap='viridis', s=10)
        fig.colorbar(scatter, label='Class Labels')
        ax.set_title('Latent Space Visualization with PCA (3D)')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.show()
    else:
        print(f"PCA visualization not supported for {n_components} dimensions.")

        

# Salvataggio e caricamento del modello
def save_encoder(encoder, decoder, optimizer, train_data, val_data, epoch, stopped_epoch, encoder_layers, decoder_layers, 
               train_losses, val_losses, recon_losses, kld_losses, effective_kld_losses, beta_values, gradient_history, 
               output_folder="Models", model_name="vae_model.pth"):
    """
    Salva encoder, decoder, lo stato dell'ottimizzatore e i dati di training/validazione in un unico file.

    Parametri:
    - encoder: rete encoder da salvare.
    - decoder: rete decoder da salvare.
    - optimizer: ottimizzatore associato.
    - train_data: dati di training.
    - val_data: dati di validazione.
    - epoch: epoca corrente al momento del salvataggio.
    - stopped_epoch: epoca in cui l'early stopping ha fermato il training.
    - encoder_layers, decoder_layers: dimensioni dei layer di encoder e decoder.
    - train_losses, val_losses, recon_losses, kld_losses, beta_values: metriche del training.
    - gradient_history: storico dei gradienti.
    - output_folder: cartella di destinazione.
    - model_name: nome del file di salvataggio.
    """
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, model_name)

    torch.save({
        'epoch': epoch,
        'stopped_epoch': stopped_epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_data': train_data,
        'val_data': val_data,
        'encoder_layers': encoder_layers,
        'decoder_layers': decoder_layers,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'recon_losses': recon_losses,
        'kld_losses': kld_losses,
        'effective_kld_losses': effective_kld_losses,
        'beta_values': beta_values,
        'gradient_history': gradient_history
    }, file_path)

    print(f"Model saved to {file_path}")


def load_encoder_model(path):
    checkpoint = torch.load(path)
    return {
        'epoch': checkpoint['epoch'],
        'stopped_epoch': checkpoint['stopped_epoch'],
        'encoder_state_dict': checkpoint['encoder_state_dict'],
        'decoder_state_dict': checkpoint['decoder_state_dict'],
        'optimizer_state_dict': checkpoint['optimizer_state_dict'],
        'train_data': checkpoint['train_data'],
        'val_data': checkpoint['val_data'],
        'encoder_layers': checkpoint['encoder_layers'],
        'decoder_layers': checkpoint['decoder_layers'],
        'train_losses': checkpoint['train_losses'],
        'val_losses': checkpoint['val_losses'],
        'recon_losses': checkpoint['recon_losses'],
        'kld_losses': checkpoint['kld_losses'],
        'effective_kld_losses': checkpoint['effective_kld_losses'],
        'beta_values': checkpoint['beta_values'],
        'gradient_history': checkpoint['gradient_history']
    }

def load_encoder(path, encoder_class):
    """
    Carica l'encoder e le informazioni del training da un file .pth.
    
    Parametri:
    - path: percorso del file .pth salvato.
    - encoder_class: classe da usare per ricostruire l'encoder.
    
    Ritorna:
    - encoder: istanza dell'encoder con i parametri caricati.
    - training_info: dizionario contenente le variabili del training e dei dati.
    """
    # Carica il checkpoint dal file
    checkpoint = torch.load(path)

    # Ricostruisci l'encoder
    encoder_layers = checkpoint['encoder_layers']
    encoder = encoder_class(encoder_layers)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])

    # Raccogli le informazioni del training
    training_info = {
        'epoch': checkpoint['epoch'],
        'stopped_epoch': checkpoint['stopped_epoch'],
        'train_data': checkpoint['train_data'],
        'val_data': checkpoint['val_data'],
        'encoder_layers': checkpoint['encoder_layers'],
        'decoder_layers': checkpoint['decoder_layers'],  # Incluso per compatibilità
        'train_losses': checkpoint['train_losses'],
        'val_losses': checkpoint['val_losses'],
        'recon_losses': checkpoint['recon_losses'],
        'kld_losses': checkpoint['kld_losses'],
        'effective_kld_losses': checkpoint['effective_kld_losses'],
        'beta_values': checkpoint['beta_values'],
        'gradient_history': checkpoint['gradient_history'],
        'optimizer_state_dict': checkpoint['optimizer_state_dict']
    }

    print(f"Encoder and training info loaded from {path}")
    return encoder, training_info




def save_decoder(decoder, train_losses, val_losses, train_data, val_data, epoch, stopped_epoch, decoder_layers, gradient_history, lr_evolution, output_folder="Models", decoder_name="decoder_model.pth"):
    """
    Salva il decoder e le informazioni del training in un file .pth.
    
    Parametri:
    - decoder: modello del decoder da salvare.
    - train_losses: lista delle perdite di training.
    - val_losses: lista delle perdite di validazione.
    - stopped_epoch: epoca in cui l'early stopping ha fermato il training.
    - gradient_history: storico dei gradienti per ogni epoca.
    - lr_evolution: evoluzione del learning rate.
    - output_folder: cartella in cui salvare il file.
    - decoder_name: nome del file di salvataggio.
    """
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, decoder_name)

    torch.save({
        'decoder_state_dict': decoder.state_dict(),
        'decoder_layers': decoder_layers,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_data': train_data,
        'val_data': val_data,
        'epoch': epoch,
        'stopped_epoch': stopped_epoch,
        'gradient_history': gradient_history,
        'lr_evolution': lr_evolution
    }, file_path)

    print(f"Decoder and training info saved to {file_path}")

def load_decoder(file_path, decoder_class):
    """
    Carica il decoder e le informazioni del training da un file .pth.
    
    Parametri:
    - file_path: percorso del file .pth salvato.
    - decoder_class: classe da usare per ricostruire il decoder.
    
    Ritorna:
    - decoder: istanza del decoder con i parametri caricati.
    - training_info: dizionario contenente le variabili del training e dei dati.
    """
    # Carica il checkpoint dal file
    checkpoint = torch.load(file_path)

    # Ricostruisci il decoder
    decoder_layers = checkpoint['decoder_layers']
    decoder_layers_reversed = decoder_layers[::-1]
    decoder = decoder_class(decoder_layers_reversed)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # Raccogli le informazioni del training
    training_info = {
        'train_losses': checkpoint['train_losses'],
        'val_losses': checkpoint['val_losses'],
        'train_data': checkpoint['train_data'],
        'val_data': checkpoint['val_data'],
        'epoch': checkpoint['epoch'],
        'stopped_epoch': checkpoint['stopped_epoch'],
        'gradient_history': checkpoint['gradient_history'],
        'lr_evolution': checkpoint['lr_evolution']
    }

    print(f"Decoder and training info loaded from {file_path}")
    return decoder, training_info




# Function to run a VAE from a DataLoader and collect outputs
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



