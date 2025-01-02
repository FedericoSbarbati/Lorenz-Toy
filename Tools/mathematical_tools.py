from sklearn.neighbors import NearestNeighbors
from scipy.special import legendre
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

def project_on_legendre(hankel_matrix, P):
    """
    Proiezione della matrice di Hankel sui polinomi di Legendre.

    Parameters:
    hankel_matrix: np.ndarray
        Matrice di Hankel.
    P: np.ndarray
        Matrice dei polinomi di Legendre (d x r).

    Returns:
    projected_data: np.ndarray
        Matrice dei dati proiettati nello spazio a dimensione ridotta.
    """
    return hankel_matrix.T @ P  # Proiezione sui polinomi di Legendre