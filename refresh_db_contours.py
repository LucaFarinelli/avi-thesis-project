import cv2
import numpy as np
import os
import mysql.connector
from rembg import remove
import Config
import Homography
import Utils

def refresh_contours():
    print("\n" + "=" * 50)
    print("UTILITY DI REFRESH CONTORNI DATABASE (AI-BASED)")
    print("=" * 50)
    print("\nQuesto script rigenererà tutti i contorni nel database usando rembg.")
    print("Verranno creati contorni puliti su sfondo nero per un matching coerente.")
    
    confirm = input("\nVuoi procedere? (s/n): ").lower()
    if confirm != 's':
        return

    conn = mysql.connector.connect(**Config.DB_CONFIG)
    cursor = conn.cursor()

    # Recupera tutti i record
    query = "SELECT idScarpa, nome_scarpa, path_originale, path_contorno FROM dataset"
    cursor.execute(query)
    shoes = cursor.fetchall()
    
    print(f"\nTrovati {len(shoes)} record da aggiornare.\n")

    TARGET_WIDTH = 700
    TARGET_HEIGHT = 1000

    updated_count = 0
    for shoe in shoes:
        id_scarpa, nome, path_orig, path_cont = shoe
        
        print(f"Aggiornamento ID {id_scarpa}: {nome}...")
        
        if not os.path.exists(path_orig):
            print(f"  [ERRORE] File originale non trovato: {path_orig}")
            continue

        # Carica immagine originale
        img = cv2.imread(path_orig)
        if img is None:
            print(f"  [ERRORE] Impossibile caricare {path_orig}")
            continue

        # 1. Normalizzazione (stesse dimensioni di Main e Bulk Importer)
        img = Homography.ensure_standard_size(img, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # 2. Rimozione sfondo AI
        try:
            img_no_bg = remove(img)
            mask = img_no_bg[:, :, 3] # Canale Alpha
            
            # 3. Creazione immagine "High Detail" (Scarpa originale + Bordi interni su sfondo nero)
            # Questo permette un matching basato sui colori e dettagli strutturali (lacci, loghi).
            shoe_only = cv2.bitwise_and(img, img, mask=mask)
            gray_shoe = cv2.cvtColor(shoe_only, cv2.COLOR_BGR2GRAY)
            internal_edges = cv2.Canny(gray_shoe, 50, 150)
            
            clean_shoe_img = shoe_only.copy()
            clean_shoe_img[internal_edges > 0] = [255, 255, 255] # Overlay bordi bianchi
            
            # 5. Sovrascrittura file contorno esistente
            # Se il path nel DB è nullo, ne creiamo uno nuovo
            if not path_cont:
                path_cont = f"images/scarpa_contorni_{id_scarpa}.jpg"
                cursor.execute("UPDATE dataset SET path_contorno = %s WHERE idScarpa = %s", (path_cont, id_scarpa))
            
            cv2.imwrite(path_cont, clean_shoe_img)
            print(f"  [OK] Riferimento aggiornato (Full Detail) in: {path_cont}")
            updated_count += 1
            
        except Exception as e:
            print(f"  [ERRORE] Durante l'elaborazione AI: {e}")

    conn.commit()
    conn.close()

    print("\n" + "=" * 50)
    print(f"REFRESH COMPLETATO: {updated_count} record aggiornati.")
    print("=" * 50)

if __name__ == "__main__":
    refresh_contours()
