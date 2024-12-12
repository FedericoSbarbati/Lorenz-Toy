import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

def reconstruct_signal_from_embedding(embedding, embedding_dim, delay):
    """
    Ricostruisce il segnale originale da un embedding ritardato.

    Parametri:
    - embedding: Array numpy con l'embedding (n_samples, embedding_dim).
    - embedding_dim: Dimensione dell'embedding (int).
    - delay: Ritardo tra campioni nell'embedding (int).

    Ritorna:
    - signal_reconstructed: Array numpy 1D con il segnale ricostruito.
    """
    n_samples = embedding.shape[0] + (embedding_dim - 1) * delay
    signal_reconstructed = np.zeros(n_samples)

    # Conta quante volte ogni valore contribuisce alla ricostruzione
    weight = np.zeros(n_samples)

    for i in range(embedding.shape[0]):
        for j in range(embedding_dim):
            idx = i + j * delay
            signal_reconstructed[idx] += embedding[i, j]
            weight[idx] += 1

    # Media i valori sovrapposti
    signal_reconstructed /= np.maximum(weight, 1)  # Evita la divisione per zero

    return signal_reconstructed


def plot_r2_histogram(true_data, predicted_data):
    # Convert tensors to numpy arrays
    inputs_np = true_data
    reconstructed_np = predicted_data
    
    # Calculate R² score manually
    ss_res = np.sum((inputs_np - reconstructed_np) ** 2, axis=1)
    ss_tot = np.sum((inputs_np - np.mean(inputs_np, axis=0)) ** 2, axis=1)
    r2_scores = 1 - (ss_res / ss_tot)

    print(f"r2_score entries: {len(r2_scores)}")

    # Get indices of the 3 samples with the worst R²
    worst_r2_indices = np.argsort(r2_scores)[:3]

    return worst_r2_indices, r2_scores


def calculate_errors_and_plot_hist(true_data, predicted_data):
    # Convert tensors to numpy arrays
    inputs_np = true_data
    reconstructed_np = predicted_data

    # Calculate MSE
    mse_errors = np.mean((inputs_np - reconstructed_np) ** 2, axis=1)
    # Calculate Max SE
    max_se_errors = np.max((inputs_np - reconstructed_np) ** 2, axis=1)

    print(f"Entries: {len(mse_errors)}")

    # Get indices of the 3 samples with the worst MAXse
    worst_max_se_indices = np.argsort(max_se_errors)[-3:]
    # Get indices of the 3 samples with the best MSE
    best_mse_indices = np.argsort(mse_errors)[:3]

    plt.figure(figsize=(12, 6))
    
    # Plot MSE histogram
    plt.subplot(1, 2, 1)
    plt.hist(mse_errors, bins=50, alpha=0.6, label='MSE')
    plt.title(f'MSE Histogram\nEstimate: {mse_errors.mean():.4f} ± {mse_errors.std():.4f}')
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot Max SE histogram
    plt.subplot(1, 2, 2)
    plt.hist(max_se_errors, bins=50, alpha=0.6, label='Max SE')
    plt.title(f'Max SE Histogram\nEstimate: {max_se_errors.mean():.4f} ± {max_se_errors.std():.4f}')
    plt.xlabel('Max SE')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    return worst_max_se_indices, best_mse_indices, mse_errors, max_se_errors


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
    print("Varianza spiegata:", pca.explained_variance_ratio_)

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


def plot_signal_and_pcs(original_signal, embedding, n_pcs=3, n_delays=5, delay_step=15):
    """
    Plotta il segnale originale e le sue componenti principali calcolate tramite PCA 
    su una matrice time-delay embedding.

    Parametri:
    - original_signal: Array numpy con il segnale temporale originale (1D).
    - n_pcs: Numero di componenti principali da calcolare.
    - n_delays: Numero di ritardi da includere nell'embedding.
    - delay_step: Passo temporale tra i ritardi.
    """
    # PCA sull'embedding
    pca = PCA(n_components=n_pcs)
    pcs_embedding = pca.fit_transform(embedding)

    # Tempo (asse x)
    time = np.arange(embedding.shape[0])
    for i in range(n_pcs):
        plt.figure(figsize=(12, 6))
        plt.plot(original_signal, label="Segnale Originale", color="purple", alpha=0.5)
        plt.plot(time, pcs_embedding[:, i], label=f"PC{i + 1}", linestyle="--")
        plt.xlabel("Tempo")
        plt.ylabel("Ampiezza")
        plt.title(f"Segnale Originale e PC{i + 1} (Time-Delay Embedding)")
        plt.legend()
        plt.grid(False)
        plt.show()

    return pcs_embedding

import pandas as pd
import os
# Funzione per salvare i dati di analisi in un file CSV
def save_analysis_data(model_name, latent_dim, r2_mean1, r2_std1, mse_mean1, mse_std1, maxse_mean1, maxse_std1, r2_mean2, r2_std2, mse_mean2, mse_std2, maxse_mean2, maxse_std2, r2_z1, r2_z2, r2_Z1Z3, r2_t_Z1, r2_t_Z3, file_path):
    # Crea un dizionario con i dati da salvare
    data = {
        "model_name": model_name,
        "latent_dim": latent_dim,
        "r2_mean1": r2_mean1,
        "r2_std1": r2_std1,
        "mse_mean1": mse_mean1,
        "mse_std1": mse_std1,
        "maxse_mean1": maxse_mean1,
        "maxse_std1": maxse_std1,
        "r2_mean2": r2_mean2,
        "r2_std2": r2_std2,
        "mse_mean2": mse_mean2,
        "mse_std2": mse_std2,
        "maxse_mean2": maxse_mean2,
        "maxse_std2": maxse_std2,
        "r2_z1": r2_z1,
        "r2_z2": r2_z2,
        "r2_Z1Z3": r2_Z1Z3,
        "r2_t_Z1" : r2_t_Z1,
        "r2_t_Z3" : r2_t_Z3
    }
    
    # Controlla se il file esiste già
    if os.path.exists(file_path):
        # Se esiste, carica il dataframe e aggiungi la nuova riga
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    else:
        # Se non esiste, crea un nuovo dataframe
        df = pd.DataFrame([data])
    
    # Salva il dataframe in un file CSV
    df.to_csv(file_path, index=False)
    print(f"Analysis data saved to {file_path}")



# Funzione per leggere tutti i file di analisi nella cartella "Analysis Data"
def read_all_analysis_data(analysis_folder="Analysis Data"):
    all_analysis_data = []
    if os.path.exists(analysis_folder):
        for file_name in os.listdir(analysis_folder):
            if file_name.endswith(".csv"):
                file_path = os.path.join(analysis_folder, file_name)
                df = pd.read_csv(file_path)
                all_analysis_data.append(df)
    if all_analysis_data:
        all_data_df = pd.concat(all_analysis_data, ignore_index=True)
        return all_data_df
    else:
        print("No analysis files found in the folder.")
        return None
    

# Funzione per plottare i risultati di analisi
def plot_analysis_results(analysis_data_df):
    if analysis_data_df is not None:
        # Plot R²
        plt.figure(figsize=(10, 6))
        plt.scatter(analysis_data_df["latent_dim"], analysis_data_df["r2_mean"], c=analysis_data_df["latent_dim"], cmap='viridis', label="R²")
        plt.errorbar(analysis_data_df["latent_dim"], analysis_data_df["r2_mean"], yerr=analysis_data_df["r2_std"], fmt='none', ecolor='black')
        plt.xlabel("Latent Dimension")
        plt.ylabel("R² Score")
        plt.title("R² Score vs Latent Dimension")
        plt.grid(True)
        plt.legend()
        plt.colorbar(label='Latent Dimension')
        plt.show()
        
        # Plot MSE
        plt.figure(figsize=(10, 6))
        plt.scatter(analysis_data_df["latent_dim"], analysis_data_df["mse_mean"], c=analysis_data_df["latent_dim"], cmap='viridis', label="MSE")
        plt.errorbar(analysis_data_df["latent_dim"], analysis_data_df["mse_mean"], yerr=analysis_data_df["mse_std"], fmt='none', ecolor='black')
        plt.xlabel("Latent Dimension")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.title("MSE vs Latent Dimension")
        plt.grid(True)
        plt.legend()
        plt.colorbar(label='Latent Dimension')
        plt.show()
        
        # Plot Max SE
        plt.figure(figsize=(10, 6))
        plt.scatter(analysis_data_df["latent_dim"], analysis_data_df["maxse_mean"], c=analysis_data_df["latent_dim"], cmap='viridis', label="Max SE")
        plt.errorbar(analysis_data_df["latent_dim"], analysis_data_df["maxse_mean"], yerr=analysis_data_df["maxse_std"], fmt='none', ecolor='black')
        plt.xlabel("Latent Dimension")
        plt.ylabel("Maximum Squared Error (Max SE)")
        plt.title("Max SE vs Latent Dimension")
        plt.grid(True)
        plt.legend()
        plt.colorbar(label='Latent Dimension')
        plt.show()
    else:
        print("No analysis data available to plot.")
