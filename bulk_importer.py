import cv2
import numpy as np
import os
import Config
import Homography
import Utils
from rembg import remove
import config_settings as cfg

# Prefissi file da importare (modifica qui per altri oggetti in futuro)
ACCEPTED_PREFIXES = ("can", "difetto")
# Prefissi che identificano la classe difettosa
DEFECT_PREFIXES = ("difetto",)

def bulk_import():
    print("\n" + "=" * 50)
    print("SISTEMA DI IMPORTAZIONE MASSIVA DATASET")
    print("=" * 50)
    
    object_type = input("\nInserisci il tipo oggetto (es. 'shoe', 'bottle'): ").strip().lower()
    if not object_type:
        print("[ERRORE] Tipo oggetto necessario.")
        return

    object_name = input("Inserisci il nome del modello (es. 'Nike Dunk'): ").strip()
    if not object_name:
        print("[ERRORE] Nome modello necessario.")
        return

    input_dir = "input"
    if not os.path.exists(input_dir):
        print(f"[ERRORE] Cartella '{input_dir}' non trovata.")
        return

    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    valid_files = [
        f for f in files if f.lower().startswith(ACCEPTED_PREFIXES)
    ]

    if not valid_files:
        print(
            f"[AVVISO] Nessun file immagine con prefissi {ACCEPTED_PREFIXES} trovato in '{input_dir}'."
        )
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
        label = 1 if filename.lower().startswith(DEFECT_PREFIXES) else 0
        label_str = "DIFETTOSA" if label == 1 else "CONFORME"
        
        print(f"\nElaborazione: {filename} ({label_str})...")
        
        # Carica immagine
        img = cv2.imread(path)
        if img is None:
            print(f"  [ERRORE] Impossibile caricare {filename}")
            continue
            
        # 1. Normalizzazione dimensioni centralizzata
        img = Homography.ensure_standard_size(img, (cfg.TARGET_WIDTH, cfg.TARGET_HEIGHT))
        
        # 1b. Rimozione Sfondo tramite rembg
        print(f"  [INFO] Rimozione sfondo in corso...")
        img_no_bg = remove(img) 
        
        # Salva l'immagine processata in dbimage per controllo utente
        db_save_path = os.path.join(db_image_dir, f"proc_{filename}")
        cv2.imwrite(db_save_path, img_no_bg)
        
        # rembg restituisce RGBA, estraiamo la maschera dall'alpha channel
        mask_rembg = img_no_bg[:, :, 3]
        
        # Creazione immagine "High Detail" centralizzata in Utils
        reference_img = Utils.generate_high_detail_reference(img, mask_rembg)
        
        # 3. Salvataggio riferimento scarpa pulita
        next_id = Config.get_next_id()
        reference_path = f"images/{object_type}_contorni_{next_id}.jpg"
        cv2.imwrite(reference_path, reference_img)
        
        # 4. Salvataggio nel database
        new_id = Config.save_to_database(
            object_name, path, reference_path, label=label, object_type=object_type
        )
        print(f"  [OK] Salvata con ID: {new_id}")
        imported_count += 1

    print("\n" + "=" * 50)
    print(f"IMPORTAZIONE COMPLETATA: {imported_count} record aggiunti.")
    print("ORA PUOI LANCIARE 'python3 svm_trainer.py' PER AGGIORNARE IL MODELLO.")
    print("=" * 50)

if __name__ == "__main__":
    bulk_import()
