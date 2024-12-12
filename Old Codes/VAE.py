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

from Tools.training_tools import*

def reconstruct_z2fromz1(epochs, train_loader, val_loader, model, optimizer, scheduler, scheduler_config, kl_annealing_epochs, decay_start, decay_epoch, beta_method="constant", beta_value=1.0, early_stopping_params=None):
    # Early stopping
    early_stopping = EarlyStopping(**early_stopping_params) if early_stopping_params else None

    # Liste per memorizzare le perdite
    train_losses = []
    val_losses = []
    recon_losses = []
    kld_losses = []
    beta_values = []
    lr_evolution = []
    # Per salvare i gradienti medi per epoch
    gradient_history = {
        "encoder": [],
        "decoder": [],
        "latent": []
    }


    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        kld_loss_epoch = 0
        recon_loss_epoch = 0

        # Calcola il valore di beta
        beta = get_beta(epoch, kl_annealing_epochs, method=beta_method, beta_value=beta_value, decay_start=decay_start, decay_epochs=decay_epoch)
        beta_values.append(beta)

        for batch in train_loader:
            y1, y2 = batch  # Decomponi input (y1) e target (y2)
            y1 = y1.float()
            y2 = y2.float()
            optimizer.zero_grad()

            # Calcola l'output della rete
            recon_y2, mu, log_var = model(y1)

            # Calcola la perdita basata su z2 (y)
            loss, recon_loss, kld_loss = loss_function_vae(recon_y2, y2, mu, log_var, beta)

            loss.backward()

             # Variabili temporanee per sommare i gradienti per batch
            encoder_grad_total = 0
            decoder_grad_total = 0
            latent_grad_total = 0
            num_batches = 0

            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Nel ciclo batch del training
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if "encoder" in name:
                        encoder_grad_total += grad_norm
                    elif "decoder" in name:
                        decoder_grad_total += grad_norm
                    elif "fc_mu" in name or "fc_log_var" in name: 
                        latent_grad_total += grad_norm
            num_batches += 1

            # Accesso ai learning rate
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']

            lr_evolution.append(current_lr)

            # Alla fine dell'epoch, calcola la media
            gradient_history["encoder"].append(encoder_grad_total / num_batches)
            gradient_history["decoder"].append(decoder_grad_total / num_batches)
            gradient_history["latent"].append(latent_grad_total / num_batches)

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
              f'Beta: {beta:.4f}, Training Loss: {epoch_loss:.4f}, lr = {current_lr:.6f}')

    # Plot dei risultati
    plot_results(train_losses, val_losses, recon_losses, kld_losses, beta_values, gradient_history)

    return train_losses, val_losses, recon_losses, kld_losses, beta_values, early_stopping.stopped_epoch, gradient_history, lr_evolution



# METHODS TO SAVE AND LOAD OPTIMIZER STATE

def save_vae_model(model, optimizer, train_data, val_data, epoch, stopped_epoch, encoder_layers, decoder_layers, train_losses, val_losses, recon_losses, kld_losses, beta_values, gradient_history, output_folder="Models", model_name="autoencoder_model.pth"):
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
        'beta_values': beta_values,
        'gradient_history' : gradient_history
    }, file_path)

    print(f"Model saved to {file_path}")

def load_vae_model(file_path):
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
    gradient_history = checkpoint['gradient_history']

    return (model, optimizer, train_data, val_data, epoch, stopped_epoch,
            train_losses, val_losses, recon_losses, kld_losses, beta_values,gradient_history)


