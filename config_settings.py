"""
Filename: config_settings.py
Descrizione: File di configurazione globale del sistema AVI. Contiene tutte le soglie,
             i pesi e i parametri configurabili per adattare il sistema a diversi tipi di oggetti.
"""

# Dimensioni standard per l'elaborazione delle immagini (larghezza x altezza in pixel)
TARGET_WIDTH = 700
TARGET_HEIGHT = 1000

# Soglia minima di similarità combinata per considerare un match valido nel database
SIMILARITY_THRESHOLD = 0.35

# Percorsi dei file del modello di rilevamento difetti (SVM)
MODEL_PATH_TEMPLATE = "{object_type}_svm_model.pkl"
SCALER_PATH_TEMPLATE = "{object_type}_svm_scaler.pkl"

# Default legacy (manteniamo per retrocompatibilita')
MODEL_PATH = "shoe_svm_model.pkl"
SCALER_PATH = "shoe_svm_scaler.pkl"


def get_model_paths(object_type):
    """Restituisce i percorsi del modello SVM per tipo oggetto."""
    safe_type = (object_type or "shoe").lower()
    return (
        MODEL_PATH_TEMPLATE.format(object_type=safe_type),
        SCALER_PATH_TEMPLATE.format(object_type=safe_type),
    )

# === PARAMETRI PER IL MATCHING VISUAL (Compare.py) ===
# Parametri per l'estrazione delle feature orientate ai bordi
ORB_NFEATURES = 500
# Lowe's ratio test (0.7 - 0.8 è ottimale per rimuovere falsi positivi nel descriptor matching)
ORB_RATIO_THRESHOLD = 0.75

# === PESI PER IL CALCOLO DELLA SIMILARITÀ COMBINATA (Match.py) ===
WEIGHT_SIMILARITY = 0.4   # Peso per la similarità strutturale (SSIM)
WEIGHT_HISTOGRAM = 0.4    # Peso per l'istogramma del colore (HSV)
WEIGHT_ORB = 0.1          # Peso per il matching di feature (ORB)
WEIGHT_SHAPE = 0.1        # Peso per l'analisi della forma (Hu Moments)

# === SOGLIE DI PENALIZZAZIONE STRUTTURALE ===
# Se l'oggetto ha forma o geometrie incompatibili, il punteggio viene abbattuto
SHAPE_PENALTY_THRESHOLD = 0.35
ORB_PENALTY_THRESHOLD = 0.20
PENALTY_FACTOR = 0.5      # Moltiplicatore applicato in caso di fallimento del test strutturale

# === PARAMETRI STIMA DIMENSIONALE (Utils.py) ===
# Fattore di calibrazione simulato (Pixel/mm) a distanza telecamera fissa (Z costante)
PIXEL_PER_MM = 5.3

# === PARAMETRI DI TRAINING SVM (svm_trainer.py) ===
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
PARAM_GRID = {
    'C': [0.1, 1.0, 10.0, 100.0],
    'gamma': ['scale', 'auto', 0.01, 0.001],
    'kernel': ['rbf']
}
