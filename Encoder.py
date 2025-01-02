import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torch.optim.lr_scheduler as lr_scheduler


from Tools.training_tools import*

# Creazione del dataset per allenare l'encoder
class EmbeddedDataset(Dataset):
    def __init__(self, projected_hankel):
        """
        Dataset per allenare l'encoder. Ogni colonna della matrice proiettata è un esempio.

        Parameters:
        - projected_hankel (numpy.ndarray): Matrice proiettata di dimensione (r, p-d+1),
                                             dove r è il numero di polinomi di Legendre.
        """
        # Converti le colonne della matrice in righe
        self.data = torch.tensor(projected_hankel.T, dtype=torch.float32)

    def __len__(self):
        # Numero di esempi (colonne originali)
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Restituisci il campione e il target
        return self.data[idx], self.data[idx]



    

# Creazione del dataset per allenare il decoder con due insiemi di embedding temporali
class EmbeddedDatasetforDecoder(Dataset):
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

    


# Crezione delle classi per il modello VAE
class Encoder(nn.Module):
    def __init__(self, layer_dims):
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(len(layer_dims) - 2):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            self.layers.append(nn.LeakyReLU())
        self.encoder = nn.Sequential(*self.layers)
        
        # Strati per la media e la log-varianza
        self.fc_mu = nn.Linear(layer_dims[-2], layer_dims[-1])
        self.fc_log_var = nn.Linear(layer_dims[-2], layer_dims[-1])

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        return mu, log_var
    
class Decoder(nn.Module):
    def __init__(self, layer_dims):
        super(Decoder, self).__init__()
        self.layers = []
        for i in range(len(layer_dims) - 1, 0, -1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i - 1]))
            if i - 1 > 0:  # Strati intermedi
                self.layers.append(nn.LeakyReLU())
            else:  # Ultimo strato
                self.layers.append(nn.Sigmoid())  # Output normalizzati
        self.decoder = nn.Sequential(*self.layers)

    def forward(self, z):
        return self.decoder(z)

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        """
        Inizializza il modello VAE con encoder e decoder forniti come input.

        Parametri:
        - encoder: istanza di una classe Encoder
        - decoder: istanza di una classe Decoder
        """
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, log_var):
        """
        Applica il trucco della ri-parametrizzazione per campionare dallo spazio latente.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Propagazione forward: codifica, campiona e decodifica.

        Parametri:
        - x: input del modello.

        Ritorna:
        - reconstructed: output ricostruito.
        - mu: media del vettore latente.
        - log_var: log-varianza del vettore latente.
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decoder(z)
        return reconstructed, mu, log_var

    


# Training del VAE per ricostruzione di Y1: allenamento dell'encoder
def trainEncoder(epochs, train_loader, val_loader, model, optimizer, scheduler, scheduler_config, kl_annealing_epochs, decay_start, decay_epoch, beta_method="constant", beta_value=1.0, early_stopping_params=None):
    # Early stopping
    early_stopping = EarlyStopping(**early_stopping_params) if early_stopping_params else None

    # Liste per memorizzare le perdite
    train_losses = []
    val_losses = []
    recon_losses = []
    kld_losses = []
    effective_kld_losses = []
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
        real_kld_loss = 0

        # Calcola il valore di beta
        beta = get_beta(epoch, kl_annealing_epochs, method=beta_method, beta_value=beta_value, decay_start=decay_start, decay_epochs=decay_epoch)
        beta_values.append(beta)

        for batch in train_loader:
            input, target = batch  # Decomponi input (y1) e target (y1) per ricostruzione
            input = input.float()
            target = target.float()
            optimizer.zero_grad()

            # Calcola l'output della rete
            recon, mu, log_var = model(input)

            # Calcola la perdita basata su z2 (y)
            loss, recon_loss, kld_loss = loss_function_vae(recon, target, mu, log_var, beta)

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
                    if name in ["encoder.encoder.0.weight", "encoder.encoder.0.bias", "encoder.encoder.2.weight", "encoder.encoder.2.bias"]:
                        encoder_grad_total += grad_norm
                    elif name in ["decoder.decoder.0.weight", "decoder.decoder.0.bias", "decoder.decoder.2.weight", "decoder.decoder.2.bias", "decoder.decoder.4.weight", "decoder.decoder.4.bias"]:
                        decoder_grad_total += grad_norm
                    elif name in ["encoder.fc_mu.weight", "encoder.fc_mu.bias", "encoder.fc_log_var.weight", "encoder.fc_log_var.bias"]:
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
            real_kld_loss += (beta*kld_loss.item())

        epoch_loss /= len(train_loader.dataset)
        kld_loss_epoch /= len(train_loader.dataset)
        recon_loss_epoch /= len(train_loader.dataset)
        real_kld_loss /= len(train_loader.dataset)

        train_losses.append(epoch_loss)
        recon_losses.append(recon_loss_epoch)
        kld_losses.append(kld_loss_epoch)
        effective_kld_losses.append(real_kld_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input, target = batch  # Decomponi input (z1) e target (z2)
                input = input.float()
                target = target.float()
                recon, mu, log_var = model(input)
                loss, _, _  = loss_function_vae(recon, target, mu, log_var, beta)
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
              f'Beta: {beta:.4f}, KLD*B = {real_kld_loss:.4f} Training Loss: {epoch_loss:.4f}, lr = {current_lr:.6f}')

    # Plot dei risultati
    plot_results(train_losses, val_losses, recon_losses, kld_losses, effective_kld_losses, beta_values, gradient_history)

    return train_losses, val_losses, recon_losses, kld_losses, effective_kld_losses, beta_values, early_stopping.stopped_epoch, gradient_history, lr_evolution


# Training del VAE per ricostruzione di Y2 con Encoder già allenato
def reconstruct_y2fromy1(epochs, train_loader, val_loader, model, optimizer, scheduler, scheduler_config, early_stopping_params=None):
    # Gradienti dell'Encoder e dello spazio latente sono già congelati nel Notebook principale
    # Early stopping
    early_stopping = EarlyStopping(**early_stopping_params) if early_stopping_params else None

    # Liste per memorizzare le perdite
    train_losses = []
    val_losses = []
    # Durante il ciclo di training
    decoder_gradients = {
        "decoder": []
    }
    lr_evolution = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            y1, y2 = batch  # Decomponi input (z1) e target (z2)
            y1 = y1.float()
            y2 = y2.float()
            optimizer.zero_grad()

            # Variabili temporanee per sommare i gradienti per batch
            decoder_grad_total = 0
            num_batches = 0

            # Calcola l'output della rete
            recon_y2, _, _ = model(y1)

            # Calcola la perdita basata su z2 (y)
            recon_loss = torch.nn.functional.mse_loss(recon_y2, y2, reduction='mean')
            recon_loss.backward()


            # Clip decoder gradient norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Nel ciclo batch del training
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if name in ["decoder.decoder.0.weight", "decoder.decoder.0.bias", "decoder.decoder.2.weight", "decoder.decoder.2.bias", "decoder.decoder.4.weight", "decoder.decoder.4.bias"]:
                        decoder_grad_total += grad_norm

            num_batches += 1

            # Accesso ai learning rate
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']

            lr_evolution.append(current_lr)

            # Alla fine dell'epoch, calcola la media
            decoder_gradients["decoder"].append(decoder_grad_total / num_batches)

            # Aggiorna solo i parametri del decoder
            optimizer.step()

            epoch_loss += recon_loss.item()

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                y1, y2 = batch  # Decomponi input (y1) e target (y2)
                y1 = y1.float()
                y2 = y2.float()
                recon_y2, _, _ = model(y1)
                recon_loss = torch.nn.functional.mse_loss(recon_y2, y2, reduction='mean')
                val_loss += recon_loss.item()

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
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Training Loss: {epoch_loss:.6f}, lr = {current_lr:.6f}')

    # Plot dei risultati DA AGGIORNARE
    plot_Decoder_results(train_losses, val_losses, decoder_gradients)

    return train_losses, val_losses, early_stopping.stopped_epoch, decoder_gradients, lr_evolution





  