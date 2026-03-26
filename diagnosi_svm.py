"""
Script di diagnosi rapida per lo SVM.
Esegui con: python3 diagnosi_svm.py
Verifica:
  1. Distribuzione etichette nel DB (0=conforme, 1=difettosa)
  2. Lunghezza del vettore feature estratto (deve essere 324)
  3. Esistenza dei file del modello
"""
import os
import cv2
import numpy as np
from rembg import remove
import Config
import Utils
import Homography
import config_settings as cfg

print("=" * 60)
print("DIAGNOSI SVM - DetectShoe AVI")
print("=" * 60)

# 1. Distribuzione etichette
print("\n[1] Distribuzione etichette nel database:")
data = Config.get_all_labeled_data()
conformi = [(p, l) for p, l in data if l == 0]
difettosi = [(p, l) for p, l in data if l == 1]
mancanti = [(p, l) for p, l in data if not os.path.exists(p)]
print(f"  Conformi  (label=0): {len(conformi)}")
print(f"  Difettosi (label=1): {len(difettosi)}")
print(f"  File mancanti:       {len(mancanti)}")
if mancanti:
    for p, l in mancanti[:5]:
        print(f"    MANCANTE: {p}")

# 2. Feature vector length
print("\n[2] Verifica lunghezza vettore feature (attesa: 324):")
sample_pairs = [(p, l) for p, l in data if os.path.exists(p)]
if sample_pairs:
    test_path, test_label = sample_pairs[0]
    img = cv2.imread(test_path)
    if img is not None:
        img = Homography.ensure_standard_size(img, (cfg.TARGET_WIDTH, cfg.TARGET_HEIGHT))
        img_no_bg = remove(img)
        mask = img_no_bg[:, :, 3]
        features = Utils.extract_svm_features(img, mask_rembg=mask)
        print(f"  Lunghezza vettore: {len(features)} ({'OK' if len(features) == 324 else 'ERRORE - INCOERENTE!'})")
        print(f"  Min: {features.min():.4f}, Max: {features.max():.4f}, Mean: {features.mean():.4f}")
    else:
        print("  Impossibile leggere il file di test.")
else:
    print("  Nessun file valido trovato.")

# 3. Modello e scaler
print("\n[3] File modello:")
print(f"  {cfg.MODEL_PATH}: {'TROVATO' if os.path.exists(cfg.MODEL_PATH) else 'NON TROVATO - rilanciare svm_trainer.py!'}")
print(f"  {cfg.SCALER_PATH}: {'TROVATO' if os.path.exists(cfg.SCALER_PATH) else 'NON TROVATO - rilanciare svm_trainer.py!'}")

print("\n" + "=" * 60)
print("DIAGNOSI COMPLETATA")
print("=" * 60)
