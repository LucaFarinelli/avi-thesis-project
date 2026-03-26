import cv2
import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from rembg import remove

import Config
import Utils
import Homography
import config_settings as cfg

# Percorsi dei modelli dal file di configurazione
MODEL_PATH = cfg.MODEL_PATH
SCALER_PATH = cfg.SCALER_PATH


def augment_image(img, mask):
    """
    Data Augmentation: genera varianti dell'immagine e della sua mask per aumentare il dataset.
    Il flip flippa ENTRAMBI (img + mask) per mantenere la coerenza geometrica delle feature.
    Le variazioni di luminosità non alterano la geometria, quindi la mask rimane la stessa.
    Ritorna una lista di tuple (immagine_augmentata, mask_augmentata).
    """
    augmented = [(img, mask)]  # Originale sempre inclusa
    
    # 1. Flip orizzontale - flippa ANCHE la mask per coerenza geometrica
    img_flip = cv2.flip(img, 1)
    mask_flip = cv2.flip(mask, 1)
    augmented.append((img_flip, mask_flip))
    
    # 2. Variazione luminosità (+20%) - la forma non cambia, mask invariata
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=15)
    augmented.append((bright, mask))
    
    # 3. Variazione luminosità (-20%) - la forma non cambia, mask invariata
    dark = cv2.convertScaleAbs(img, alpha=0.8, beta=-15)
    augmented.append((dark, mask))
    
    return augmented


def train_svm_model():
    print("Avvio addestramento SVM per rilevamento difetti...")
    print("(con Data Augmentation e ottimizzazione parametri)\n")
    
    # 1. Recupera i dati dal database
    data = Config.get_all_labeled_data()
    
    if not data or len(data) < 2:
        print("[ERRORE] Dati insufficienti per l'addestramento. Servono almeno 2 campioni.")
        return False
    
    features = []
    labels = []
    
    valid_data = [(p, l) for p, l in data if os.path.exists(p)]
    skipped = len(data) - len(valid_data)
    
    if skipped > 0:
        print(f"[AVVISO] {skipped} file non trovati, verranno saltati.")
    
    print(f"Campioni validi: {len(valid_data)} (Augmentation x4 = {len(valid_data) * 4} totali)")
    
    # 2. Estrazione feature con Data Augmentation
    for i, (path, label) in enumerate(valid_data):
        img = cv2.imread(path)
        if img is None:
            continue
            
        # Normalizzazione (come nel Main)
        img = Homography.ensure_standard_size(img, (cfg.TARGET_WIDTH, cfg.TARGET_HEIGHT))
        
        # Rimozione sfondo con rembg (identico a Main.py)
        img_no_bg = remove(img)
        mask = img_no_bg[:, :, 3]  # Alpha channel come maschera
        
        # Data Augmentation: genera 4 coppie (img, mask) con geometria consistente
        augmented_pairs = augment_image(img, mask)
        
        for aug_img, aug_mask in augmented_pairs:
            combined_features = Utils.extract_svm_features(aug_img, mask_rembg=aug_mask)
            features.append(combined_features)
            labels.append(label)
        
        print(f"  [{i+1}/{len(valid_data)}] {os.path.basename(path)} (label={'CONFORME' if label==0 else 'DIFETTOSO'})")
        
    print(f"\nEstrazione completata. Feature totali: {len(features)}")
    
    X = np.array(features)
    y = np.array(labels)
    
    # Distribuzione classi
    unique, counts = np.unique(y, return_counts=True)
    print(f"Distribuzione classi (dopo augmentation):")
    for cls, cnt in zip(unique, counts):
        print(f"  Classe {int(cls)} ({'CONFORME' if cls==0 else 'DIFETTOSO'}): {cnt} campioni")
    
    # Verifica che ci siano almeno due classi distinte (0 e 1)
    if len(np.unique(y)) < 2:
        print("[ERRORE] L'addestramento richiede campioni di entrambe le classi (buoni e difettosi).")
        return False

    # 3. Normalizzazione feature (StandardScaler)
    # Fondamentale perché le feature geometriche hanno scale diverse dagli istogrammi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Split dati
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_SEED, stratify=y
    )
    
    # 5. GridSearchCV: trova automaticamente i parametri migliori
    print("\nRicerca parametri ottimali (GridSearchCV)...")
    param_grid = cfg.PARAM_GRID
    
    cv_folds = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_SEED)
    grid_search = GridSearchCV(
        SVC(probability=True, class_weight='balanced'),
        param_grid, cv=cv_folds, scoring='f1_weighted', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    model = grid_search.best_estimator_
    print(f"Parametri migliori trovati: {grid_search.best_params_}")
    print(f"Score migliore (CV): {grid_search.best_score_:.2%}")
    
    # 6. Valutazione sul test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuratezza sul test set: {accuracy:.2%}")
    print("\nReport di classificazione:")
    print(classification_report(y_test, y_pred, target_names=['Conforme', 'Difettosa']))
    
    # --- GENERAZIONE GRAFICI PER LA TESI ---
    os.makedirs('debug1', exist_ok=True)
    
    # 1. Matrice di Confusione
    print("Generazione Matrice di Confusione...")
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, 
        display_labels=['Conforme', 'Difettosa'], cmap=plt.cm.Blues
    )
    disp.ax_.set_title('Matrice di Confusione SVM')
    plt.savefig('debug1/thesis_01_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot Istogrammi Medi (Conforme vs Difettosa)
    print("Generazione Confronto Feature...")
    X_conforme = X[y == 0]
    X_difettosa = X[y == 1]
    
    if len(X_conforme) > 0 and len(X_difettosa) > 0:
        mean_conforme = np.mean(X_conforme, axis=0)
        mean_difettosa = np.mean(X_difettosa, axis=0)
        
        plt.figure(figsize=(10, 5))
        plt.plot(mean_conforme, label='Media Conforme', color='green', alpha=0.7)
        plt.plot(mean_difettosa, label='Media Difettosa', color='red', alpha=0.7)
        plt.title('Confronto Firma SVM Unificata (LBP + HSV + Geometria)')
        plt.xlabel('Dimensione Vettore (LBP: 0-255 | HSV: 256-317 | Geo: 318-323)')
        plt.ylabel('Valore feature normalizzato')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('debug1/thesis_02_lbp_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Plot Scatter 2D con PCA
    print("Generazione Spazio Vettoriale 2D (PCA)...")
    if len(X_scaled) >= 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=100, alpha=0.6)
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Conforme', markerfacecolor=plt.cm.coolwarm(0.0), markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Difettosa', markerfacecolor=plt.cm.coolwarm(1.0), markersize=10)
        ]
        plt.legend(handles=legend_elements, loc="best")
        
        plt.title('Rappresentazione Spazio SVM (PCA-ridotto a 2D)')
        plt.xlabel(f'Componente Principale 1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        plt.ylabel(f'Componente Principale 2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig('debug1/thesis_03_svm_pca_space.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 7. Salvataggio Modello e Scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n[OK] Modello salvato in: {MODEL_PATH}")
    print(f"[OK] Scaler salvato in: {SCALER_PATH}")
    print("[OK] Grafici per la tesi generati in 'debug1/':")
    print("  - thesis_01_confusion_matrix.png")
    print("  - thesis_02_lbp_comparison.png")
    print("  - thesis_03_svm_pca_space.png")
    return True

if __name__ == "__main__":
    train_svm_model()
