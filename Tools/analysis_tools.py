import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from Tools.data_utils import extract_params_from_filename
from sklearn.decomposition import PCA
import pandas as pd
import os
import json


#Funzioni per analizzare i risultati di training
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

import os
import pandas as pd

def save_decoder_analysis_data(
    model_name, latent_dim, r2_mean1, r2_std1, mse_mean1, mse_std1, maxse_mean1, maxse_std1, 
    r2_mean2, r2_std2, mse_mean2, mse_std2, maxse_mean2, maxse_std2, 
    r2_z1, r2_z2, r2_Z1Z3, r2_t_Z1, r2_t_Z3, ind1 ,ind2 ,ind3 , file_path
):
    """
    Salva i dati di analisi in un file CSV appiattendo i dati annidati per leggibilità in Excel.
    """
    # Crea un dizionario appiattito con i dati da salvare
    data = {
        "Model Name": model_name,
        "Latent Dim": latent_dim,
        "Z1 R2 Mean": r2_mean1,
        "Z1 R2 Std": r2_std1,
        "Z1 MSE Mean": mse_mean1,
        "Z1 MSE Std": mse_std1,
        "Z1 MaxSE Mean": maxse_mean1,
        "Z1 MaxSE Std": maxse_std1,
        "Z3 R2 Mean": r2_mean2,
        "Z3 R2 Std": r2_std2,
        "Z3 MSE Mean": mse_mean2,
        "Z3 MSE Std": mse_std2,
        "Z3 MaxSE Mean": maxse_mean2,
        "Z3 MaxSE Std": maxse_std2,
        "PCA R2 Z1 PC": r2_z1,
        "Varianzce Explained Z1": ind1,
        "PCA R2 Z3 PC": r2_z2,
        "Varianzce Explained Z3": ind2,
        "PCA R2 (Z1+Z3) PC": r2_Z1Z3,
        "Variance Explained (Z1+Z3)": ind3,
        "PCA R2 Time Signal Z1": r2_t_Z1,
        "PCA R2 Time Signal Z3": r2_t_Z3
    }
    
    # Crea un nuovo dataframe con i dati
    df = pd.DataFrame([data])
    
    # Sovrascrive il file CSV esistente
    df.to_csv(file_path, index=False)
    print(f"Analysis data saved to {file_path}")


def save_encoder_analysis_data(
    model_name, latent_dim, r2_mean, r2_std, mse_mean, mse_std, maxse_mean, maxse_std, 
    r2_z2, r2_t_Z2, ind1, file_path ):
    """
    Salva i dati di analisi in un file CSV appiattendo i dati annidati per leggibilità in Excel.
    """
    # Crea un dizionario appiattito con i dati da salvare
    data = {
        "Model Name": model_name,
        "Latent Dim": latent_dim,
        "Z2 R2 Mean": r2_mean,
        "Z2 R2 Std": r2_std,
        "Z2 MSE Mean": mse_mean,
        "Z2 MSE Std": mse_std,
        "Z2 MaxSE Mean": maxse_mean,
        "Z2 MaxSE Std": maxse_std,
        "PCA R2 Z2 PC": r2_z2,
        "Varianzce Explained Z2": ind1,
        "PCA R2 Time Signal Z2": r2_t_Z2
    }
    
    # Crea un nuovo dataframe con i dati
    df = pd.DataFrame([data])
    
    # Sovrascrive il file CSV esistente
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


def latent_variables(vae_model, mean_vectors, logvar_vectors, dataloader, device='cpu'):
    # Assumi che il modello e i dati siano già caricati
    vae_model.eval()  # Imposta il modello in modalità eval

    # Usa mean_vectors e logvar_vectors già calcolati
    all_mu = torch.tensor(mean_vectors)
    all_sigma = torch.exp(0.5 * torch.tensor(logvar_vectors))
    
    ratios = abs(all_sigma / all_mu)

    n_dimensions = ratios.shape[1]
    n_cols = 2  # Numero di colonne
    n_rows = (n_dimensions + n_cols - 1) // n_cols  # Calcola righe necessarie
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()  # Appiattisci l'array degli assi per accedere in un ciclo

    for i in range(n_dimensions):  # Per ogni dimensione latente
        ax = axes[i]
        data = ratios[:, i].numpy()
        
        ax.hist(data, bins=50, alpha=0.7, color=f'C{i}', label=f'Dimensione {i+1}')
        ax.axvline(np.mean(data), color='red', linestyle='--', label='Media')
        ax.set_xlabel('Rapporto (sigma / mean)')
        ax.set_ylabel('Frequenza')
        ax.set_title(f'Dimensione {i+1}')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, min(100, np.max(data)))  # Riduci dinamica se necessario

    # Nascondi gli assi inutilizzati
    for j in range(n_dimensions, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def pca_r2_analysis_embeddings(true_embeddings, reconstructed_embeddings, title1 = "titolo", title2 = "titolo", n_components=4):
    """
    Esegue PCA sui vettori di embedding originali e ricostruiti, confronta fino a n_components
    e calcola R² per ciascuna componente principale.

    Parametri:
    - true_embeddings: Array 2D (n_samples, embedding_dim) dei vettori originali.
    - reconstructed_embeddings: Array 2D (n_samples, embedding_dim) dei vettori ricostruiti.
    - n_components: Numero di componenti principali da analizzare.

    Output:
    - R² per ogni componente principale.
    - Varianza spiegata per ogni componente principale.
    - Grafico della varianza spiegata per ciascuna PC.
    """
    # Assicurati che i dati siano 2D
    if true_embeddings.ndim != 2 or reconstructed_embeddings.ndim != 2:
        raise ValueError("I dati di input devono essere array 2D con shape (n_samples, embedding_dim).")
    
    # PCA sui dati originali
    pca = PCA(n_components=n_components)
    pca.fit(true_embeddings)  # Fitta il PCA ai dati originali

    # Proiezione dei dati originali e ricostruiti nello spazio delle PC
    true_projected = pca.transform(true_embeddings)
    reconstructed_projected = pca.transform(reconstructed_embeddings)

    # Funzione per calcolare R² a mano
    def calculate_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
        return 1 - ss_res / ss_tot

    # Calcolo di R² per ciascuna componente principale
    r2_per_component = [
        calculate_r2(true_projected[:, i], reconstructed_projected[:, i])
        for i in range(true_projected.shape[1])
    ]

    # Ottenere la varianza spiegata da ciascuna componente principale (DEI DATI ORIGINALI)
    explained_variance = pca.explained_variance_ratio_

    # Plot dei risultati
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: R² per componente principale
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(r2_per_component) + 1), r2_per_component, marker='o', label="R² per PC")
    plt.xlabel("Componente Principale (PC)")
    plt.ylabel("R²")
    plt.title(title1)
    plt.grid(True)
    plt.legend()

    # Subplot 2: Varianza spiegata
    plt.subplot(1, 2, 2)
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label="Varianza Spiegata")
    plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', color='red', label="Cumulativa")
    plt.xlabel("Componente Principale (PC)")
    plt.ylabel("Varianza Spiegata")
    plt.title(title2)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    return r2_per_component, explained_variance




# Funzioni per caricare i dati di analisi da file CSV
def load_encoder_analysis_data(folder_path):
    """
    Carica i file CSV dalla cartella specificata e restituisce un unico DataFrame.
    
    Parametri:
    - folder_path: percorso della cartella contenente i file CSV degli encoder.
    
    Ritorna:
    - DataFrame con tutti i dati uniti.
    """
    all_data = []
    loaded_files = set()  # Set per tenere traccia dei file già caricati
    
    # Itera su tutti i file nella cartella
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            if file_path not in loaded_files:  # Controlla se il file è già stato caricato
                print(f"Caricando dati da: {file_path}")
                df = pd.read_csv(file_path)
                all_data.append(df)
                loaded_files.add(file_path)  # Aggiungi il file al set dei file caricati
    
    # Combina tutti i DataFrame in uno solo
    combined_data = pd.concat(all_data, ignore_index=True)
    # Rimuove i duplicati
    combined_data = combined_data.drop_duplicates()

    print(f"Totale modelli caricati: {len(combined_data)}")
    return combined_data

def load_decoder_analysis_data(folder_path):
    """
    Carica i file CSV dalla cartella specificata e restituisce un unico DataFrame.
    
    Parametri:
    - folder_path: percorso della cartella contenente i file CSV dei decoder.
    
    Ritorna:
    - DataFrame con tutti i dati uniti.
    """
    all_data = []
    
    # Itera su tutti i file nella cartella
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Caricando dati da: {file_path}")
            df = pd.read_csv(file_path)
            all_data.append(df)
    
    # Combina tutti i DataFrame in uno solo
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Totale modelli caricati: {len(combined_data)}")
    return combined_data


# Funzioni per analizzare il decoder

def plot_encoder_metric(data, metric, title, ylabel):
    """
    Crea un plot di una metrica (es. R2 Mean) per ogni encoder, utilizzando il nome del modello sull'asse X.
    
    Parametri:
    - data: DataFrame contenente i dati degli encoder.
    - metric: Nome della colonna nel DataFrame che rappresenta la metrica da plottare.
    - title: Titolo del grafico.
    - ylabel: Etichetta dell'asse Y.
    """
    # Converti i nomi dei modelli in stringhe per evitare errori di ordinamento
    data["Model Name"] = data["Model Name"].astype(str)
    
    # Ordina i modelli alfabeticamente per chiarezza (opzionale)
    data = data.sort_values(by="Model Name")

    # Estrai i nomi dei modelli e i valori della metrica
    model_names = data["Model Name"]
    metric_values = data[metric]
    
    # Crea il plot
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, metric_values, color="skyblue", edgecolor="black")
    plt.title(title)
    plt.xlabel("Model Name")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def plot_z_metrics(data, r2_col, mse_col, maxse_col, title):
    """
    Crea un plot con barre sovrapposte per R2, MSE e MaxSE per ogni encoder.
    I valori sono normalizzati per gestire scale diverse.
    
    Parametri:
    - data: DataFrame con i dati degli encoder.
    - r2_col: Nome della colonna per R2.
    - mse_col: Nome della colonna per MSE.
    - maxse_col: Nome della colonna per MaxSE.
    - title: Titolo del grafico.
    """
    # Crea una copia del DataFrame per evitare SettingWithCopyWarning
    data = data.copy()
    
    # Normalizza i dati per portare le metriche tra 0 e 1
    data["R2 Normalized"] = data[r2_col]  # Normalizzazione opzionale
    data["MSE Normalized"] = (data[mse_col] - data[mse_col].min()) / (data[mse_col].max() - data[mse_col].min())
    data["MaxSE Normalized"] = (data[maxse_col] - data[maxse_col].min()) / (data[maxse_col].max() - data[maxse_col].min())

    # Etichette sull'asse X
    model_names = data["Model Name"].astype(str)
    x = np.arange(len(model_names))
    
    # Larghezza delle barre
    bar_width = 0.3
    # Crea il plot
    plt.figure(figsize=(12, 6))
    
    # Aggiungi le barre per ciascuna metrica
    plt.bar(x - bar_width, data["R2 Normalized"], bar_width, alpha=0.7, label="R2", color="blue")
    plt.bar(x, data["MSE Normalized"], bar_width, alpha=0.7, label="MSE (Normalized)", color="orange")
    plt.bar(x + bar_width, data["MaxSE Normalized"], bar_width, alpha=0.7, label="MaxSE (Normalized)", color="green")
    
    # Personalizza il grafico
    plt.title(title)
    plt.xlabel("Model Name")
    plt.ylabel("Normalized Metric Value")
    plt.xticks(x, model_names, rotation=45, ha="right", fontsize=10)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_r2_z1_z3(data, r2_z1_col, r2_z3_col, title):
    """
    Crea un plot con barre sovrapposte per R² di Z1 e Z3 per ogni modello.
    
    Parametri:
    - data: DataFrame con i dati degli encoder.
    - r2_z1_col: Nome della colonna per R² di Z1.
    - r2_z3_col: Nome della colonna per R² di Z3.
    - title: Titolo del grafico.
    """
    # Crea una copia del DataFrame per evitare SettingWithCopyWarning
    data = data.copy()
    
    # Etichette sull'asse X
    model_names = data["Model Name"].astype(str)
    x = np.arange(len(model_names))
    
    # Larghezza delle barre
    bar_width = 0.3
    
    # Crea il plot
    plt.figure(figsize=(12, 6))
    
    # Aggiungi le barre per ciascun R²
    plt.bar(x - bar_width / 2, data[r2_z1_col], bar_width, alpha=0.7, label="R² Z1", color="blue")
    plt.bar(x + bar_width / 2, data[r2_z3_col], bar_width, alpha=0.7, label="R² Z3", color="green")
    
    # Personalizza il grafico
    plt.title(title)
    plt.xlabel("Model Name")
    plt.ylabel("R² Value")
    plt.xticks(x, model_names, rotation=45, ha="right", fontsize=10)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


import re

def preprocess_array_string(array_string):
    """
    Preprocessa una stringa di array per correggere formati irregolari,
    rimuovere prefissi 'np.float32', aggiungere virgole dove mancano
    e convertire in lista di float.
    
    Parametri:
    - array_string: stringa contenente i dati numerici
    
    Ritorna:
    - Lista di float.
    """
    try:
        # 1. Rimuove 'np.float32' e altre parentesi esterne
        clean_string = re.sub(r"np\.float32\((.*?)\)", r"\1", array_string)
        
        # 2. Rimuove newline e spazi multipli
        clean_string = re.sub(r"\s+", " ", clean_string.strip())
        
        # 3. Sostituisce spazi con virgole (separazione dei valori)
        clean_string = clean_string.replace(" ", ", ")
        
        # 4. Rimuove parentesi quadre esterne se presenti
        clean_string = clean_string.strip("[]")
        
        # 5. Converte in lista di float
        return [float(value) for value in clean_string.split(",") if value.strip()]
    
    except Exception as e:
        print(f"Errore nel preprocessamento della stringa: {array_string}")
        print(f"Dettagli dell'errore: {e}")
        return []




def plot_pca_r2_and_variance(data, r2_col, variance_col, title):
    """
    Crea un plot con linee collegate per ogni modello:
    - X: Varianza spiegata per ciascun componente principale.
    - Y: R2 per ciascun componente principale.
    
    Ogni modello ha un colore diverso.
    
    Parametri:
    - data: DataFrame con i dati degli encoder.
    - r2_col: Nome della colonna contenente i R2 per i PC.
    - variance_col: Nome della colonna contenente la varianza spiegata per i PC.
    - title: Titolo del grafico.
    """
    plt.figure(figsize=(12, 6))
    
    for index, row in data.iterrows():
        model_name = row["Model Name"]
        
        # Recupera R2 e varianza spiegata, correggendo eventuali problemi di formato
        r2_values = np.array(eval(row[r2_col]))
        variance_values = np.array(preprocess_array_string(row[variance_col]))
        
        # Plotta i valori per il modello
        plt.plot(
            variance_values, r2_values, marker='o', label=model_name, alpha=0.8
        )

    
    # Personalizza il grafico
    plt.title(title)
    plt.xlabel("Explained Variance (%)")
    plt.ylabel("PCA R2")
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.legend(title="Models", loc="best", fontsize="small")
    plt.tight_layout()
    plt.show()

def preprocess_array_data(data, r2_col, variance_col):
    """
    Preprocessa i dati nelle colonne PCA R2 e Variance Explained.
    
    Parametri:
    - data: DataFrame con i dati.
    - r2_col: Nome della colonna con R2 delle componenti principali.
    - variance_col: Nome della colonna con la varianza spiegata.
    
    Ritorna:
    - Dizionario con i dati processati per plotting.
    """
    processed_data = {"models": [], "r2_values": [], "variance_values": []}

    for index, row in data.iterrows():
        model_name = row["Model Name"]

        # Trasforma i dati da stringa a lista numerica
        r2_values = np.array(preprocess_array_string(row[r2_col]))
        variance_values = np.array(preprocess_array_string(row[variance_col]))

        processed_data["models"].append(model_name)
        processed_data["r2_values"].append(r2_values)
        processed_data["variance_values"].append(variance_values)
    
    return processed_data

def plot_r2_pca(data, r2_col , variance_col, title):

    processed_data = preprocess_array_data(data, r2_col, variance_col)
    
    # Plot per R2
    plt.figure(figsize=(12, 6))
    for model_name, r2_values in zip(processed_data["models"], processed_data["r2_values"]):
        components = np.arange(1, len(r2_values) + 1)
        plt.plot(components, r2_values, marker='o', linestyle='-', label=model_name, alpha=0.8)
    
    plt.title(title)
    plt.xlabel("Principal Component Number")
    plt.ylabel("R² Value")
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.legend(title="Models", loc="best", fontsize="small")
    plt.tight_layout()
    plt.show()

def plot_variance_pca(data, r2_col , variance_col, title):
    processed_data = preprocess_array_data(data, r2_col, variance_col)
    # Plot per Varianza spiegata
    plt.figure(figsize=(12, 6))
    for model_name, variance_values in zip(processed_data["models"], processed_data["variance_values"]):
        components = np.arange(1, len(variance_values) + 1)
        plt.plot(components, variance_values, marker='x', linestyle='--', label=model_name, alpha=0.8)
    
    
    plt.title(title)
    plt.xlabel("Principal Component Number")
    plt.ylabel("Explained Variance")
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.legend(title="Models", loc="best", fontsize="small")
    plt.tight_layout()
    plt.show()



def plot_r2_and_variance_separate(data, r2_col, variance_col):
    """
    Crea due plot separati per R2 e Varianza spiegata dalle componenti principali.
    
    Parametri:
    - data: DataFrame con i dati.
    - r2_col: Nome della colonna con R2.
    - variance_col: Nome della colonna con la varianza spiegata.
    """
    processed_data = preprocess_array_data(data, r2_col, variance_col)
    
    # Plot per R2
    plt.figure(figsize=(12, 6))
    for model_name, r2_values in zip(processed_data["models"], processed_data["r2_values"]):
        components = np.arange(1, len(r2_values) + 1)
        plt.plot(components, r2_values, marker='o', linestyle='-', label=model_name, alpha=0.8)
    
    plt.title("PCA R² by Principal Component")
    plt.xlabel("Principal Component Number")
    plt.ylabel("R² Value")
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.legend(title="Models", loc="best", fontsize="small")
    plt.tight_layout()
    plt.show()

    # Plot per Varianza spiegata
    plt.figure(figsize=(12, 6))
    for model_name, variance_values in zip(processed_data["models"], processed_data["variance_values"]):
        components = np.arange(1, len(variance_values) + 1)
        plt.plot(components, variance_values, marker='x', linestyle='--', label=model_name, alpha=0.8)
    
    
    plt.title("Explained Variance by Principal Component")
    plt.xlabel("Principal Component Number")
    plt.ylabel("Explained Variance")
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.legend(title="Models", loc="best", fontsize="small")
    plt.tight_layout()
    plt.show()

def order_dataset_by_model(data):
    # Extract model information
    data[['Model', 'A', 'B']] = data['Model Name'].str.extract(r'(\d+D),(\d+),(\d+)')
    # Convert A and B to integers
    data['A'] = data['A'].astype(int)
    data['B'] = data['B'].astype(int)
    # Sort by B first, then by A
    data = data.sort_values(by=['B', 'A'])
    # Drop the temporary columns
    data = data.drop(columns=['Model', 'A', 'B'])
    return data

def enrich_analysis_with_config(analysis_data, config_file):
    """
    Unisce i dati di analisi con i parametri di configurazione dei modelli e
    aggiunge parametri estratti dai nomi dei file di training.

    Parameters:
        analysis_data (pd.DataFrame): Dati di analisi, deve contenere la colonna 'Model Name'.
        config_file (str): Percorso del file JSON con le configurazioni dei modelli.

    Returns:
        pd.DataFrame: Dati arricchiti con i parametri di configurazione e quelli estratti dai file.
    """
    # Carica il database delle configurazioni
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    # Converte il database in un DataFrame per facilitarne la manipolazione
    config_df = pd.DataFrame.from_dict(config_data, orient='index').reset_index()
    config_df.rename(columns={'index': 'Model Name'}, inplace=True)
    
    # Unisce i dati di analisi con le configurazioni usando 'Model Name'
    enriched_data = pd.merge(analysis_data, config_df, on='Model Name', how='left')
    
    # Estrae i parametri dal nome del file e li aggiunge come colonne
    extracted_params = []
    for model_name in enriched_data['dataset']:
        params = extract_params_from_filename(model_name)
        if params:
            extracted_params.append(params)
        else:
            # Aggiungi valori nulli se i parametri non possono essere estratti
            extracted_params.append({"tau": None, "embedding_dim": None, "r": None, "alpha": None, "beta": None})
    
    # Converte i parametri estratti in un DataFrame
    params_df = pd.DataFrame(extracted_params)
    
    # Aggiunge i parametri estratti ai dati arricchiti
    enriched_data = pd.concat([enriched_data, params_df], axis=1)
    
    return enriched_data

def filter_by_config_params(data, filters, yes):
    """
    Filtra i modelli nel dataset in base a più parametri di configurazione o a intervalli.

    Parameters:
        data (pd.DataFrame): Dataset contenente i dati arricchiti con le configurazioni.
        filters (list): Lista di filtri. Ogni filtro è un dizionario con:
                        - "param_name" (str): Nome del parametro di configurazione.
                        - "filter_type" (str): Tipo di filtro ('equals' o 'range').
                        - "filter_value": Valore del filtro. Per 'equals' è un valore singolo,
                          per 'range' è una tupla (min, max).

    Returns:
        pd.DataFrame: Dataset filtrato.
    """
    if yes == False:
        return data
    else:
        filtered_data = data.copy()

        for filter_item in filters:
            param_name = filter_item["param_name"]
            filter_type = filter_item["filter_type"]
            filter_value = filter_item["filter_value"]

            if param_name == "NO":
                continue
            
            elif param_name not in filtered_data.columns:
                raise ValueError(f"Il parametro '{param_name}' non esiste nel dataset.")
            
            if filter_type == "equals":
                filtered_data = filtered_data[filtered_data[param_name] == filter_value]
            
            elif filter_type == "range":
                min_value, max_value = filter_value
                filtered_data = filtered_data[
                    (filtered_data[param_name] >= min_value) & 
                    (filtered_data[param_name] <= max_value)
                ]
            else:
                raise ValueError(f"Tipo di filtro '{filter_type}' non supportato.")
        
        return filtered_data
    
def sort_by_alpha_and_beta(data, alpha_col, beta_col):
    """
    Ordina i dati rispetto ai parametri alpha e beta.
    Per ciascun valore di alpha (in ordine crescente), ordina i beta in ordine crescente.

    Parameters:
        data (pd.DataFrame): Dataset da ordinare.
        alpha_col (str): Nome della colonna per i valori di alpha.
        beta_col (str): Nome della colonna per i valori di beta.

    Returns:
        pd.DataFrame: Dataset ordinato.
    """
    if alpha_col not in data.columns or beta_col not in data.columns:
        raise ValueError(f"Le colonne '{alpha_col}' o '{beta_col}' non esistono nel dataset.")
    
    # Ordina prima per alpha (crescente) e poi per beta (crescente)
    sorted_data = data.sort_values(by=[alpha_col, beta_col], ascending=[True, True])
    return sorted_data










