import matplotlib.pyplot as plt
from Tools.mathematical_tools import*
import os
import re
import json


#   DEFINIZIONE DELLA CLASSE SimulationData
class SimulationData:
    def __init__(self, n_steps=None, dt=None, alpha=None, beta =None, tau=None, embedding_dim=None, legendre_dim=None):
        # Parametri simulazione
        self.n_steps = n_steps
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        
        # Embedding parameters and legendre basis dimension
        self.tau = tau
        self.embedding_dim = embedding_dim
        self.r = legendre_dim
        
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

        # Hankel matrix of the Observables
        self.H1 = None
        self.H2_1 = None
        self.H2_2 = None

        self.P = None   # Legendre Basis

        # Projected Hankel Matrices

        self.projected_H1 = None
        self.projected_H2_1 = None
        self.projected_H2_2 = None


    # Simulation methods
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
        self.y1_embedding = self.create_time_delay_embedding(self.y1_norm, self.tau, self.embedding_dim)
        # Supponiamo che y2 sia una matrice con shape (n_steps, n_dimensions)
        y2_component_1 = self.y2_norm[:, 0]
        y2_component_2 = self.y2_norm[:, 1]

        self.y2_embedding_1 = create_time_delay_embedding(y2_component_1, self.tau, self.embedding_dim)
        self.y2_embedding_2 = create_time_delay_embedding(y2_component_2, self.tau, self.embedding_dim)

    def create_Hankel_Matrices(self):

        '''
        Generate Hankel Matrices for the two observables y1 and y2.
        For y2 we compute two Matrices H2_1 and H2_2, corresponding to the two scalar components of y2.
        '''
        self.H1 = create_Hankel_matrix(self.y1_embedding, self.tau)
        self.H2_1 = create_Hankel_matrix(self.y2_embedding_1, self.tau)
        self.H2_2 = create_Hankel_matrix(self.y2_embedding_2, self.tau)

    def project_on_legendre(self):
        
        '''
        Project the Hankel Matrices on the first r Legendre Polynomials.
        '''
        self.P = legendre_basis(self.embedding_dim, self.r)

        self.projected_H1 = project_on_legendre(self.H1, self.P)
        self.projected_H2_1 = project_on_legendre(self.H2_1, self.P)
        self.projected_H2_2 = project_on_legendre(self.H2_2, self.P)

        self.projected_H1 = self.projected_H1.T
        self.projected_H2_1 = self.projected_H2_1.T
        self.projected_H2_2 = self.projected_H2_2.T


    
    @staticmethod
    def create_time_delay_embedding(data, delay, dimension):
        """
        Crea un embedding a ritardo temporale per una serie temporale.
        """
        return np.array([
            data[i: i + delay * dimension: delay]
            for i in range(len(data) - delay * (dimension - 1))
        ])
    
    # Methods to save and load data
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
        return f"delay{int(self.tau)}dim{int(self.embedding_dim)},{self.r}_noise{alpha_str},{beta_str}.json"
    
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
            "r": self.r,
            "trajectory": self.trajectory.tolist(),
            "y1": self.y1.tolist(),
            "y2": self.y2.tolist(),
            "y1_norm": self.y1_norm.tolist(),
            "y2_norm": self.y2_norm.tolist(),
            "y1_embedding": self.y1_embedding.tolist() if self.y1_embedding is not None else None,
            "y2_embedding_1": self.y2_embedding_1.tolist() if self.y2_embedding_1 is not None else None,
            "y2_embedding_2": self.y2_embedding_2.tolist() if self.y2_embedding_2 is not None else None,
            "H1": self.H1.tolist() if self.H1 is not None else None,
            "H2_1": self.H2_1.tolist() if self.H2_1 is not None else None,
            "H2_2": self.H2_2.tolist() if self.H2_2 is not None else None,
            "P": self.P.tolist() if self.P is not None else None,
            "projected_H1": self.projected_H1.tolist() if self.projected_H1 is not None else None,
            "projected_H2_1": self.projected_H2_1.tolist() if self.projected_H2_1 is not None else None,
            "projected_H2_2": self.projected_H2_2.tolist() if self.projected_H2_2 is not None else None,
        }
        with open(file_path, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load_from_params(cls, tau, embedding_dim, alpha, beta, r,input_path):
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
        file_name = f"delay{tau}dim{embedding_dim},{r}_noise{alpha_str},{beta_str}.json"
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
            legendre_dim=data["r"]
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
        instance.H1 = np.array(data["H1"])
        instance.H2_1 = np.array(data["H2_1"])
        instance.H2_2 = np.array(data["H2_2"])
        instance.P = np.array(data["P"])
        instance.projected_H1 = np.array(data["projected_H1"])
        instance.projected_H2_1 = np.array(data["projected_H2_1"])
        instance.projected_H2_2 = np.array(data["projected_H2_2"])

        return instance
    
    
def generate_input_filename(tau, embedding_dim, alpha, beta , r):
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
    return f"delay{int(tau)}dim{int(embedding_dim)},{r}_noise{alpha_str},{beta_str}"


def extract_params_from_filename(filename):
    """
    Estrae i parametri tau, embedding_dim, alpha e beta dal nome del file.

    Parametri:
    - filename: Nome del file (es. "delay10dim20_noise0.1,0.2.json")

    Ritorna:
    - Un dizionario con i parametri estratti: {tau, embedding_dim, alpha, beta}
    """
    # Utilizzo di espressioni regolari per estrarre i parametri
    pattern = r"delay(\d+)dim(\d+),(\d+)_noise([\d\.eE\-]+),([\d\.eE\-]+)"
    match = re.search(pattern, filename)

    if not match:
        raise ValueError(f"Il nome del file '{filename}' non segue il formato atteso!")

    # Estraggo i parametri e li converto nei loro tipi corretti
    tau = int(match.group(1))
    embedding_dim = int(match.group(2))
    r = int(match.group(3))
    alpha = float(match.group(4))
    beta = float(match.group(5))

    return {"tau": tau, "embedding_dim": embedding_dim, "r": r, "alpha": alpha, "beta": beta}





