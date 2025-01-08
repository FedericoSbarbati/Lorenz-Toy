from sklearn.neighbors import NearestNeighbors
from scipy.special import legendre
from numpy.linalg import pinv
import numpy as np
import math

# Funzione per simulare il sistema Dinamico
def lorenz_stochastic(n_steps, dt, alpha, initial_state=[1.0, 1.0, 1.0]):
    """
    Simula un attrattore di Lorenz stocastico.
    
    Parameters:
        n_steps: Numero di passi temporali da simulare
        dt: Passo temporale
        alpha: Intensità del rumore stocastico
        initial_state: Stato iniziale [z1, z2, z3]
        
    Returns:
        trajectory: Array (n_steps, 3) contenente z1, z2, z3
    """
    # Stato iniziale
    z = np.zeros((n_steps, 3))
    z[0, :] = initial_state  # Imposta lo stato iniziale
    
    # Integrazione con Euler-Maruyama
    for i in range(1, n_steps):
        z1, z2, z3 = z[i-1, :]
        # Termini deterministici
        dz1 = 10 * (z2 - z1)
        dz2 = z1 * (28 - z3) - z2
        dz3 = z1 * z2 - (8 / 3) * z3
        # Rumore stocastico
        dW = np.random.normal(0, np.sqrt(dt), size=3)
        # Aggiornamento dello stato
        z[i, 0] = z1 + dz1 * dt + alpha * dW[0]
        z[i, 1] = z2 + dz2 * dt + alpha * dW[1]
        z[i, 2] = z3 + dz3 * dt + alpha * dW[2]
    
    return z


# Funzioni per lavorare con embedding temporali
def create_time_delay_embedding(data, delay=1, embedding_dim=10):
    """
    Crea un embedding a ritardo temporale da una serie temporale.
    
    Parameters:
        data: Array con le serie temporali (es: z2)
        delay: Ritardo tra campioni
        embedding_dim: Numero di ritardi (dimensione embedding)
        
    Returns:
        embedding: Array (n_samples, embedding_dim) con l'embedding a ritardo temporale
    """
    n_samples = len(data) - (embedding_dim - 1) * delay
    embedding = np.zeros((n_samples, embedding_dim))
    
    for i in range(n_samples):
        embedding[i, :] = data[i:i + embedding_dim * delay:delay]
    
    return embedding

def mutual_information(data, delay, n_bins):
    """
    Calcola la mutual information data una serie temporale e un ritardo.
    
    Parametri:
    - data: array di dati della serie temporale
    - delay: ritardo per calcolare la mutual information
    - n_bins: numero di bin per la discretizzazione dei dati
    
    Ritorna:
    - I: valore della mutual information per il ritardo specificato
    """
    I = 0
    xmax = np.max(data)
    xmin = np.min(data)
    size_bin = (xmax - xmin) / n_bins
    
    # Dati con ritardo
    delay_data = data[delay:]
    short_data = data[:-delay]
    
    # Dizionari per probabilità marginali e congiunte
    prob_in_bin = {}
    condition_bin = {}
    condition_delay_bin = {}
    
    # Calcolo delle probabilità marginali
    for h in range(n_bins):
        condition_bin[h] = (short_data >= (xmin + h * size_bin)) & (short_data < (xmin + (h + 1) * size_bin))
        prob_in_bin[h] = np.sum(condition_bin[h]) / len(short_data)
    
    # Calcolo delle probabilità congiunte
    for h in range(n_bins):
        for k in range(n_bins):
            condition_delay_bin[k] = (delay_data >= (xmin + k * size_bin)) & (delay_data < (xmin + (k + 1) * size_bin))
            joint_prob = np.sum(condition_bin[h] & condition_delay_bin[k]) / len(short_data)
            
            # Evita logaritmi di probabilità zero
            if joint_prob > 0 and prob_in_bin[h] > 0 and prob_in_bin[k] > 0:
                I += joint_prob * math.log(joint_prob / (prob_in_bin[h] * prob_in_bin[k]))
    
    return I

def false_nearest_neighbors(data, delay, embedding_dimension, threshold=10):
    """
    Calcola la frazione di falsi vicini in modo ottimizzato.
    """
    embedded_data = create_time_delay_embedding(data, delay, embedding_dimension)
    nbrs = NearestNeighbors(n_neighbors=2).fit(embedded_data)
    distances, indices = nbrs.kneighbors(embedded_data)

    false_neighbors_count = 0
    for i in range(len(embedded_data)):
        if i + embedding_dimension * delay < len(data) and indices[i, 1] + embedding_dimension * delay < len(data):
            distance_increased = abs(
                data[i + embedding_dimension * delay] - data[indices[i, 1] + embedding_dimension * delay]
            )
            ratio = distance_increased / distances[i, 1]
            if ratio > threshold:
                false_neighbors_count += 1

    return false_neighbors_count / len(embedded_data)


# Dimensionality reduction tools

def create_Hankel_matrix(embedded_data, tau):
    """
    Creazione della matrice di Hankel a partire dai dati di embedding temporale.
    Le colonne sono i vettori di embedding e le colonne consecutive rappresentano
    l'evoluzione temporale del sistema per un intervallo di tempo tau*timestep.
    """
    d = embedded_data.shape[1]  # Dimensione dei vettori d-dimensionali
    n_vectors = (len(embedded_data) - 1) // tau + 1  # Numero di colonne nella matrice di Hankel

    hankel_matrix = np.zeros((d, n_vectors))

    for i in range(n_vectors):
        hankel_matrix[:, i] = embedded_data[i * tau]

    return hankel_matrix

def legendre_basis(d, r):
    """
    Creazione della matrice dei primi r polinomi di Legendre su d punti equispaziati nel dominio [-1, 1].
    """
    x = np.linspace(-1, 1, d)  # Punti equispaziati su [-1, 1]
    P = np.zeros((d, r))
    for k in range(r):
        P[:, k] = legendre(k)(x)
    return P


def normalize_columns(matrix):
    """
    Normalizza ogni colonna della matrice nell'intervallo [0, 1].
    
    Parameters:
    - matrix (numpy.ndarray): Matrice da normalizzare (dimensione r x (p-d+1)).
    
    Returns:
    - normalized_matrix (numpy.ndarray): Matrice normalizzata nell'intervallo [0, 1].
    - min_vals (numpy.ndarray): Minimo di ogni colonna (dimensione 1 x (p-d+1)).
    - max_vals (numpy.ndarray): Massimo di ogni colonna (dimensione 1 x (p-d+1)).
    """
    # Calcola minimo e massimo per ogni colonna
    min_vals = matrix.min(axis=0, keepdims=True)
    max_vals = matrix.max(axis=0, keepdims=True)
    
    # Normalizza ogni colonna
    normalized_matrix = (matrix - min_vals) / (max_vals - min_vals)
    
    return normalized_matrix, min_vals, max_vals

def denormalize_columns(normalized_matrix, min_vals, max_vals):
    """
    Denormalizza una matrice utilizzando i valori di minimo e massimo salvati.
    
    Parameters:
    - normalized_matrix (numpy.ndarray): Matrice normalizzata.
    - min_vals (numpy.ndarray): Minimo di ogni colonna.
    - max_vals (numpy.ndarray): Massimo di ogni colonna.
    
    Returns:
    - denormalized_matrix (numpy.ndarray): Matrice denormalizzata.
    """
    return normalized_matrix * (max_vals - min_vals) + min_vals

import numpy as np

def diagonal_mean(hankel_matrix):
    """
    Ripristina una serie temporale dalla matrice di Hankel originale
    utilizzando la media lungo le diagonali.

    Parametri:
        hankel_matrix (np.ndarray): Matrice di Hankel originale di dimensioni (d x (p-d+1)).

    Ritorna:
        np.ndarray: Serie temporale ricostruita.
    """
    # Ottieni le dimensioni della matrice di Hankel
    d, n = hankel_matrix.shape

    # Prepara un array per contenere la serie temporale ricostruita
    serie_length = d + n - 1  # Lunghezza totale della serie temporale
    serie_temporale = np.zeros(serie_length)
    conteggi = np.zeros(serie_length)  # Per calcolare la media lungo le diagonali

    # Somma i valori lungo le diagonali
    for i in range(d):
        for j in range(n):
            serie_temporale[i + j] += hankel_matrix[i, j]
            conteggi[i + j] += 1

    # Calcola la media per ciascun punto temporale
    serie_temporale /= conteggi

    return serie_temporale

def reconstruct_time_series(hankel_matrix):
    """
    Ricostruisce la serie temporale originale da una matrice di Hankel costruita
    con il formalismo rilassato.

    Args:
        hankel_matrix (np.ndarray): Matrice di Hankel (formalismo rilassato),
                                    shape (d, m), dove:
                                    - d è la dimensione del vettore di ritardo,
                                    - m è il numero di colonne.

    Returns:
        np.ndarray: Serie temporale originale ricostruita.
    """
    # Ottieni le dimensioni della matrice
    d, m = hankel_matrix.shape
    
    # Inizializza la serie temporale
    time_series_length = d + m - 1  # Lunghezza totale della serie temporale
    time_series = np.zeros(time_series_length)
    
    # Ricostruisci la serie temporale
    for i in range(d):  # Scorri sulle righe della matrice
        for j in range(m):  # Scorri sulle colonne della matrice
            time_index = i + j  # Posizione nella serie temporale
            time_series[time_index] += hankel_matrix[i, j]
    
    # Contatore per normalizzare la sovrapposizione
    overlap_count = np.zeros(time_series_length)
    for i in range(d):
        for j in range(m):
            time_index = i + j
            overlap_count[time_index] += 1
    
    # Normalizza la serie temporale per gestire la sovrapposizione
    time_series /= overlap_count
    
    return time_series






