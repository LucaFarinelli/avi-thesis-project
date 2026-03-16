import cv2
import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import Config
import Utils
import Homography

# Percorso in cui salvare il modello addestrato
MODEL_PATH = "shoe_svm_model.pkl"

def train_svm_model():
    print("Avvio addestramento SVM per rilevamento difetti...")
    
    # 1. Recupera i dati dal database
    data = Config.get_all_labeled_data()
    
    if not data or len(data) < 2:
        print("[ERRORE] Dati insufficienti per l'addestramento. Servono almeno 2 campioni.")
        return False
    
    features = []
    labels = []
    
    print(f"Trovati {len(data)} campioni nel database.")
    
    # 2. Estrazione feature per ogni record
    for path, label in data:
        if not os.path.exists(path):
            print(f"[AVVISO] File non trovato: {path}. Salto.")
            continue
            
        # Carica e processa l'immagine
        img = cv2.imread(path)
        if img is None:
            continue
            
        # Normalizzazione (come nel Main)
        img = Homography.ensure_standard_size(img, (700, 1000))
        # Estrazione combinata Texture(LBP) + Colore(HSV)
        mask = Utils.get_shoe_mask(img)
        combined_features = Utils.extract_svm_features(img, mask_rembg=mask)
        
        features.append(combined_features)
        labels.append(label)
        print(f".", end="", flush=True)
        
    print("\nEstrazione completata.")
    
    X = np.array(features)
    y = np.array(labels)
    
    # Verifica che ci siano almeno due classi distinte (0 e 1)
    if len(np.unique(y)) < 2:
        print("[ERRORE] L'addestramento richiede campioni di entrambe le classi (buoni e difettosi).")
        print(f"Classi trovate nel DB: {np.unique(y)}")
        return False

    # 3. Addestramento SVM
    # Usiamo un kernel lineare o RBF. C=1 è un buon punto di partenza.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='rbf', probability=True, C=1.0)
    model.fit(X_train, y_train)
    
    # 4. Valutazione e Stampe
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuratezza modello: {accuracy:.2%}")
    print("\nReport di classificazione:")
    print(classification_report(y_test, y_pred))
    
    # --- GENERAZIONE GRAFICI PER LA TESI ---
    os.makedirs('debug1', exist_ok=True)
    
    # 1. Matrice di Confusione
    print("Generazione Matrice di Confusione...")
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=['Conforme', 'Difettosa'], cmap=plt.cm.Blues)
    disp.ax_.set_title('Matrice di Confusione SVM')
    plt.savefig('debug1/thesis_01_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot Istogrammi Medi (Conforme vs Difettosa)
    print("Generazione Confronto Istogrammi LBP...")
    X_conforme = X[y == 0]
    X_difettosa = X[y == 1]
    
    if len(X_conforme) > 0 and len(X_difettosa) > 0:
        mean_conforme = np.mean(X_conforme, axis=0)
        mean_difettosa = np.mean(X_difettosa, axis=0)
        
        plt.figure(figsize=(10, 5))
        plt.plot(mean_conforme, label='Media Conforme', color='green', alpha=0.7)
        plt.plot(mean_difettosa, label='Media Difettosa', color='red', alpha=0.7)
        plt.title('Confronto Firma SVM Unificata (Texture LBP + Colore HSV)')
        plt.xlabel('Dimensione Vettore (LBP Bin: 0-255 | HSV Bin: 256-317)')
        plt.ylabel('Valore feature normalizzato')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('debug1/thesis_02_lbp_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Plot Scatter 2D con PCA (Rappresentazione Vettoriale per la Tesi)
    print("Generazione Spazio Vettoriale 2D (PCA)...")
    if len(X) >= 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=100)
        
        # Aggiungiamo legenda manualmente
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Conforme', markerfacecolor=plt.cm.coolwarm(0.0), markersize=10),
                           Line2D([0], [0], marker='o', color='w', label='Difettosa', markerfacecolor=plt.cm.coolwarm(1.0), markersize=10)]
        plt.legend(handles=legend_elements, loc="best")
        
        plt.title('Rappresentazione Spazio SVM (PCA-ridotto a 2D)')
        plt.xlabel(f'Componente Principale 1 ({pca.explained_variance_ratio_[0]:.1%} inv)')
        plt.ylabel(f'Componente Principale 2 ({pca.explained_variance_ratio_[1]:.1%} inv)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig('debug1/thesis_03_svm_pca_space.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 5. Salvataggio Modello
    joblib.dump(model, MODEL_PATH)
    print(f"\n[OK] Modello salvato in: {MODEL_PATH}")
    print("[OK] Grafici per la tesi generati in 'debug1/':")
    print("  - thesis_01_confusion_matrix.png")
    print("  - thesis_02_lbp_comparison.png")
    print("  - thesis_03_svm_pca_space.png")
    return True

if __name__ == "__main__":
    train_svm_model()
