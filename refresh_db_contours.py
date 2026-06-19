import cv2
import os
import Config
import Utils
import Homography
from rembg import remove
import config_settings as cfg

def refresh_all_contours():
    """
    Rigenera l'immagine di riferimento strutturale per tutti i record nel database con le nuove logiche.
    Molto utile se si cambiano i parametri di Canny o le dimensioni target nel config, senza dover
    reimportare tutte le immagini daccapo.
    """
    print("\n" + "=" * 50)
    print("SISTEMA DI RIGENERAZIONE CONTORNI DATABASE")
    print("=" * 50)
    
    conn = Config._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, object_type, path_originale FROM dataset")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("[AVVISO] Nessun record trovato nel database.")
        return
        
    print(f"Inizio aggiornamento di {len(rows)} record...\n")
    
    success_count = 0
    
    for object_id, object_type, orig_path in rows:
        object_type = object_type or "object"
        print(f"Aggiornamento ID {object_id} ({orig_path})...")
        if not os.path.exists(orig_path):
            print(f"  [ERRORE] File originale non trovato. Salto.")
            continue
        
        # Carica e normalizza
        img = cv2.imread(orig_path)
        if img is None:
            print(f"  [ERRORE] Impossibile leggere il file.")
            continue
        
        # 1. Normalizza con config
        img = Homography.ensure_standard_size(img, (cfg.TARGET_WIDTH, cfg.TARGET_HEIGHT))
        
        # 2. Rimozione sfondo
        img_no_bg = remove(img)
        mask_rembg = img_no_bg[:, :, 3]
        
        # 3. Generazione High Detail centralizzata
        reference_img = Utils.generate_high_detail_reference(img, mask_rembg)
        
        # 4. Salva il nuovo riferimento
        new_contour_path = f"images/{object_type}_contorni_{object_id}.jpg"
        cv2.imwrite(new_contour_path, reference_img)
        
        # 5. Aggiorna nome e path nel database per sicurezza
        conn = Config._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE dataset SET path_contorno = %s WHERE id = %s",
            (new_contour_path, object_id),
        )
        conn.commit()
        conn.close()
        
        print(f"  [OK] Contorno aggiornato con successo.")
        success_count += 1
        
    print("\n" + "=" * 50)
    print(f"COMPLETATO: Aggiornati con successo {success_count}/{len(rows)} record.")
    print("=" * 50)

if __name__ == "__main__":
    refresh_all_contours()
