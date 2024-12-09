import numpy as np
import math
import json
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os
import re

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




#   DEFINIZIONE DELLA CLASSE SimulationData

class SimulationData:
    def __init__(self, n_steps=None, dt=None, alpha=None, beta = None, tau=None, embedding_dim=None):
        # Parametri simulazione
        self.n_steps = n_steps
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        
        # Parametri embedding
        self.tau = tau
        self.embedding_dim = embedding_dim
        
        # Dati simulazione
        self.trajectory = None

        # Osservabili y1 e y2
        self.y1 = None
        self.y2 = None

        self.y1_norm = None
        self.y2_norm = None

        self.y1_embedding = None
        self.y2_embedding_1 = None
        self.y2_embedding_2 = None


    def run_simulation(self, lorenz_function):
        """
        Esegue la simulazione utilizzando una funzione specifica per generare il sistema di Lorenz.
        
        Parametri:
        - lorenz_function: funzione che genera la traiettoria di Lorenz
        """
        self.trajectory = lorenz_function(self.n_steps, self.dt, self.alpha)
        self.z1 = self.trajectory[:, 0]
        self.z2 = self.trajectory[:, 1]
        self.z3 = self.trajectory[:, 2]
    
    def normalize_data(self):
        """
        Normalizza le serie temporali y1 e y2.
        """
        if self.y1 is None or self.y2 is None:
            raise ValueError("Gli osservabili y1 e y2 non sono stati generati!")

        # Normalizza y1
        self.y1_norm = self.normalize_series(self.y1)
        
        # Inizializza y2_norm come array vuoto della stessa forma di y2
        self.y2_norm = np.zeros_like(self.y2)  # <--- Inizializzazione necessaria
        
        # Normalizza le colonne di y2
        self.y2_norm[:, 0] = self.normalize_series(self.y2[:, 0])
        self.y2_norm[:, 1] = self.normalize_series(self.y2[:, 1])



    def generate_filename(self):
        """
        Genera un nome di file basato sui parametri della simulazione.
        Formato: (tau)dim(embedding_dim)_noise(alpha).json
        - tau e embedding_dim: interi
        - alpha: numero decimale ridotto al minimo necessario (ad esempio 0.1 e non 0.1000)
        
        Ritorna:
        - Stringa con il nome del file generato.
        """
        alpha_str = f"{self.alpha:.15g}"  # Riduce il numero di cifre a quelle necessarie
        beta_str= f"{self.beta:.15g}"
        return f"delay{int(self.tau)}dim{int(self.embedding_dim)}_noise{alpha_str},{beta_str}.json"
    
    def generate_observables(self):
        """
        Genera gli osservabili y1 e y2 date le traiettorie del sistema di Lorenz e un parametro beta.
        """
        # Controlla che le traiettorie siano state generate
        if self.z1 is None or self.z2 is None or self.z3 is None:
            raise ValueError("Le traiettorie non sono state generate. Esegui prima run_simulation().")
        
        # Genera y1 e y2
        noise_y1 = np.random.normal(0, self.beta, size=self.z2.shape)
        self.y1 = self.z2 + noise_y1
        
        noise_y2 = np.random.normal(0, self.beta, size=(self.z1.shape[0], 2))  # Rumore 2D
        self.y2 = np.stack([self.z1, self.z3], axis=1) + noise_y2


    @staticmethod
    def normalize_series(data):
        """
        Normalizza una serie temporale nell'intervallo [0, 1].
        """
        data_min = np.min(data)
        data_max = np.max(data)
        return (data - data_min) / (data_max - data_min)
    
    def create_embeddings(self):
        """
        Crea gli embedding a ritardo temporale per z1, z2, z3.
        CAMBIA QUA PER NORMALIZZARE O MENO
        """
        self.y1_embedding = self.create_time_delay_embedding(self.y1, self.tau, self.embedding_dim)
        # Supponiamo che y2 sia una matrice con shape (n_steps, n_dimensions)
        y2_component_1 = self.y2[:, 0]
        y2_component_2 = self.y2[:, 1]

        self.y2_embedding_1 = create_time_delay_embedding(y2_component_1, self.tau, self.embedding_dim)
        self.y2_embedding_2 = create_time_delay_embedding(y2_component_2, self.tau, self.embedding_dim)

    
    @staticmethod
    def create_time_delay_embedding(data, delay, dimension):
        """
        Crea un embedding a ritardo temporale per una serie temporale.
        """
        return np.array([
            data[i: i + delay * dimension: delay]
            for i in range(len(data) - delay * (dimension - 1))
        ])
    
    def save_to_file(self, file_path):
        """
        Salva i dati della simulazione in un file JSON o NumPy.
        
        Parametri:
        - file_path: percorso del file in cui salvare i dati
        """
        data = {
            "n_steps": self.n_steps,
            "dt": self.dt,
            "alpha": self.alpha,
            "beta": self.beta,
            "tau": self.tau,
            "embedding_dim": self.embedding_dim,
            "trajectory": self.trajectory.tolist(),
            "y1": self.y1.tolist(),
            "y2": self.y2.tolist(),
            "y1_norm": self.y1_norm.tolist(),
            "y2_norm": self.y2_norm.tolist(),
            "y1_embedding": self.y1_embedding.tolist() if self.y1_embedding is not None else None,
            "y2_embedding_1": self.y2_embedding_1.tolist() if self.y2_embedding_1 is not None else None,
            "y2_embedding_2": self.y2_embedding_2.tolist() if self.y2_embedding_2 is not None else None
        }
        with open(file_path, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load_from_params(cls, tau, embedding_dim, alpha, beta, input_path):
        """
        Carica un'istanza della classe SimulationData usando i parametri specificati
        per generare il nome del file.

        Parametri:
        - tau: ritardo temporale (int)
        - embedding_dim: dimensione dell'embedding (int)
        - alpha: intensità del rumore (float)

        Ritorna:
        - Istanza della classe SimulationData con i dati caricati
        """
        # Genera il nome del file basandosi sui parametri
        alpha_str = f"{alpha:.15g}"  # Riduce il numero di cifre inutili
        beta_str = f"{beta:.15g}"
        file_name = f"delay{tau}dim{embedding_dim}_noise{alpha_str},{beta_str}.json"
        file_path = os.path.join(input_path, file_name)
        
        # Carica i dati dal file generato
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Crea un'istanza della classe con i parametri salvati
        instance = cls(
            n_steps=data["n_steps"],
            dt=data["dt"],
            alpha=data["alpha"],
            beta=data["beta"],
            tau=data["tau"],
            embedding_dim=data["embedding_dim"],
        )
        
        # Carica i dati simulati
        instance.trajectory = np.array(data["trajectory"])
        instance.y1 = np.array(data["y1"])
        instance.y2 = np.array(data["y2"])
        instance.y1_norm = np.array(data["y1_norm"])
        instance.y2_norm = np.array(data["y2_norm"])
        instance.y1_embedding = np.array(data["y1_embedding"])
        instance.y2_embedding_1 = np.array(data["y2_embedding_1"])
        instance.y2_embedding_2 = np.array(data["y2_embedding_2"])

        
        return instance
    
    
def generate_input_filename(tau, embedding_dim, alpha, beta):
    """
    Genera un nome di file basato sui parametri della simulazione.
    Formato: (tau)dim(embedding_dim)_noise(alpha).json
    - tau e embedding_dim: interi
    - alpha: numero decimale ridotto al minimo necessario (ad esempio 0.1 e non 0.1000)
    
    Ritorna:
    - Stringa con il nome del file generato.
    """
    alpha_str = f"{alpha:.15g}"  # Riduce il numero di cifre a quelle necessarie
    beta_str = f"{beta:.15g}"
    return f"delay{int(tau)}dim{int(embedding_dim)}_noise{alpha_str},{beta_str}"


def extract_params_from_filename(filename):
    """
    Estrae i parametri tau, embedding_dim, alpha e beta dal nome del file.

    Parametri:
    - filename: Nome del file (es. "delay10dim20_noise0.1,0.2.json")

    Ritorna:
    - Un dizionario con i parametri estratti: {tau, embedding_dim, alpha, beta}
    """
    # Utilizzo di espressioni regolari per estrarre i parametri
    pattern = r"delay(\d+)dim(\d+)_noise([\d\.eE\-]+),([\d\.eE\-]+)"
    match = re.search(pattern, filename)

    if not match:
        raise ValueError(f"Il nome del file '{filename}' non segue il formato atteso!")

    # Estraggo i parametri e li converto nei loro tipi corretti
    tau = int(match.group(1))
    embedding_dim = int(match.group(2))
    alpha = float(match.group(3))
    beta = float(match.group(4))

    return {"tau": tau, "embedding_dim": embedding_dim, "alpha": alpha, "beta": beta}





