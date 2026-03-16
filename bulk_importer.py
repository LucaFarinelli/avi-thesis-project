import cv2
import numpy as np
import os
import Config
import Homography
import Utils
from rembg import remove

def bulk_import():
    print("\n" + "=" * 50)
    print("SISTEMA DI IMPORTAZIONE MASSIVA DATASET")
    print("=" * 50)
    
    shoe_name = input("\nInserisci il nome del modello per queste scarpe (es. 'Nike Dunk'): ").strip()
    if not shoe_name:
        print("[ERRORE] Nome modello necessario.")
        return

    input_dir = "input"
    if not os.path.exists(input_dir):
        print(f"[ERRORE] Cartella '{input_dir}' non trovata.")
        return

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    valid_files = [f for f in files if f.startswith(('scarpa', 'difetto'))]

    if not valid_files:
        print(f"[AVVISO] Nessun file che inizia con 'scarpa' o 'difetto' trovato in '{input_dir}'.")
        return

    print(f"\nTrovati {len(valid_files)} file da importare.")
    confirm = input("Vuoi procedere? (s/n): ").lower()
    if confirm != 's':
        return

    # Gestione cartella dbimage
    db_image_dir = "dbimage"
    if not os.path.exists(db_image_dir):
        os.makedirs(db_image_dir)
    else:
        # Pulisci cartella all'inizio di ogni bulk import
        for f in os.listdir(db_image_dir):
            os.remove(os.path.join(db_image_dir, f))
    
    imported_count = 0
    for filename in valid_files:
        path = os.path.join(input_dir, filename)
        
        # Determina etichetta
        label = 1 if filename.startswith('difetto') else 0
        label_str = "DIFETTOSA" if label == 1 else "CONFORME"
        
        print(f"\nElaborazione: {filename} ({label_str})...")
        
        # Carica immagine
        img = cv2.imread(path)
        if img is None:
            print(f"  [ERRORE] Impossibile caricare {filename}")
            continue
            
        # 1. Normalizzazione dimensioni (700x1000 come nel Main)
        img = Homography.ensure_standard_size(img, (700, 1000))
        
        # 1b. Rimozione Sfondo tramite rembg
        print(f"  [INFO] Rimozione sfondo in corso...")
        img_no_bg = remove(img) 
        
        # Salva l'immagine processata in dbimage per controllo utente
        db_save_path = os.path.join(db_image_dir, f"proc_{filename}")
        cv2.imwrite(db_save_path, img_no_bg)
        
        # rembg restituisce RGBA, estraiamo la maschera dall'alpha channel
        mask_rembg = img_no_bg[:, :, 3]
        
        # 2. Estrazione maschera rembg (già calcolata)
        mask_rembg = img_no_bg[:, :, 3]
        
        # Creazione immagine "High Detail" (Scarpa originale + Bordi interni su sfondo nero)
        shoe_only = cv2.bitwise_and(img, img, mask=mask_rembg)
        gray_shoe = cv2.cvtColor(shoe_only, cv2.COLOR_BGR2GRAY)
        internal_edges = cv2.Canny(gray_shoe, 50, 150)
        
        reference_img = shoe_only.copy()
        reference_img[internal_edges > 0] = [255, 255, 255] # Overlay bordi bianchi
        
        # 3. Salvataggio riferimento scarpa pulita
        next_id = Config.get_next_id()
        reference_path = f"images/scarpa_contorni_{next_id}.jpg"
        cv2.imwrite(reference_path, reference_img)
        
        # 4. Salvataggio nel database
        new_id = Config.save_to_database(shoe_name, path, reference_path, label=label)
        print(f"  [OK] Salvata con ID: {new_id}")
        imported_count += 1

    print("\n" + "=" * 50)
    print(f"IMPORTAZIONE COMPLETATA: {imported_count} record aggiunti.")
    print("ORA PUOI LANCIARE 'python3 svm_trainer.py' PER AGGIORNARE IL MODELLO.")
    print("=" * 50)

if __name__ == "__main__":
    bulk_import()
