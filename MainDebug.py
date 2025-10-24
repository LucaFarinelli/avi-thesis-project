import cv2 as cv
import sys
import os
import numpy as np
import Config
import Match

def resize_keep_aspect(img, target_size, extra_padding=(0, 0)):
    """
    Ridimensiona mantenendo l'aspetto e aggiunge padding nero per raggiungere target_size.
    Evita stretching. Permette di aggiungere padding extra sopra e sotto.
    """
    h, w = img.shape[:2] # estraggo altezza e  larghezza  dall'immmagine
    target_h, target_w = target_size #estraggi altezza e larghezza dalla finestra target
    scale = min(target_w / w, target_h / h) #calcolo fattore in scala per ridimensionare  immagine
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv.resize(img, (new_w, new_h)) #ridimension immagine in base alle nuove misure calcolate

    # Padding per centrare
    pad_h = target_h - new_h #calcolo padding per dim_immagine = dim_target
    pad_w = target_w - new_w
    # divido padding verticale in due parti per centrare immagine
    top = (pad_h // 2) + extra_padding[0]  # Aggiungi padding extra sopra
    bottom = pad_h - (pad_h // 2) + extra_padding[1]  # Aggiungi padding extra sotto
    #divido padding orizzontale in 2 parti per centrare img
    left = pad_w // 2
    right = pad_w - left
    padded = cv.copyMakeBorder(
        resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0]
    )# aggiungo padding calcolato alla mia immagine ridimensionata (con colore nero)
    print(
        f"Dimensioni originali: {img.shape}, scalate: {new_w}x{new_h}, finali: {target_size}, padding extra: {extra_padding}"
    )
    return padded

def normalize_shoe_image(img):
    """
    Raddrizza la scarpa se inclinata, senza ritaglio pesante.
    Mantiene orientamento verticale e ridimensiona mantenendo aspetto.
    """
    img_copy = img.copy()

    gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY) #converto colore in grigio
    blur = cv.GaussianBlur(gray, (5, 5), 0) #sfocatura per ridurrre rumore e dettagli inutili
    edges = cv.Canny(blur, 50, 150) # alg rilevamento bordi
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Se non trova contorni, ridimensiona mantenendo aspetto
        return resize_keep_aspect(img_copy, (700, 1000)) 

    largest_contour = max(contours, key=cv.contourArea) #trova contorno più largo tra quelli trvoati
    rect = cv.minAreaRect(largest_contour) #calcolo rettangolo di area minima per contener contorno 
    center, (width, height), angle = rect #estraggo dimensioni dal rettangolo calcoclato

    # angolo detectato
    print(f"Debug: width={width}, height={height}, angle={angle}")

    # Normalizza l'angolo in modo che la scarpa resti verticale
    if width < height:
        angle = -angle
    else:
        angle = -(angle + 90) #aggiungo offset di 90 gradi

    # Necessario per evitare rotazione di 180°
    if angle < -90:
        angle += 180
    elif angle > 90:
        angle -= 180

    # Ignora rotazioni se l'angolo è vicino a 0 (entro una soglia)
    if abs(angle) < 10:  # Soglia di ±10 gradi
        rotated = img_copy  
    else:
        (h, w) = img_copy.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)  #matrice di rotazione
        rotated = cv.warpAffine(img_copy, M, (w, h))  # applico trasformazione all'immagine

    # Ridimensiona mantenendo aspetto, senza stretching
    final_img = resize_keep_aspect(rotated, (700, 1000))
    return final_img

def calculate_shoe_size(img_normalized, reference_cm=29.7):  # Es. foglio A4 verticale
    # Assumi riferimento visibile; in pratica, rilevalo con un altro contorno o marker
    gray = cv.cvtColor(img_normalized, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 2:
        return "Riferimento non trovato"
    
    # Supponi il contorno più grande = suola, secondo = riferimento
    contours_sorted = sorted(contours, key=cv.contourArea, reverse=True)
    suola_contour = contours_sorted[0]
    ref_contour = contours_sorted[1]  # Migliora: filtra per forma (es. rettangolo per A4)
    
    suola_rect = cv.minAreaRect(suola_contour)
    _, (suola_w, suola_h), _ = suola_rect
    suola_pixel = max(suola_w, suola_h)
    
    ref_rect = cv.minAreaRect(ref_contour)
    _, (ref_w, ref_h), _ = ref_rect
    ref_pixel = max(ref_w, ref_h)  # Per A4, usa l'altezza
    
    scale = reference_cm / ref_pixel
    suola_cm = suola_pixel * scale
    
    # Converti in taglia EU (formula approssimativa; usa tabella reale)
    taglia_eu = round((suola_cm + 1.5) / 0.6667)  # Basata su standard Mondopoint
    
    return f"Lunghezza suola: {suola_cm:.2f} cm, Taglia stimata EU: {taglia_eu}"

cv.setUseOptimized(True)
shoes_path = input("Inserire nome scarpa da cercare: ")

path = f"images/{shoes_path}.jpg"
img = cv.imread(path)

# Funzione per pulire il database da path errati
# print("🧹 Pulizia database...")
# Config.clean_invalid_paths()
# print("✅ Pulizia completata")

if img is not None:
    # Normalizza l'immagine originale
    img_normalized = normalize_shoe_image(img)

    # Nel main, dopo img_normalized:
    print(calculate_shoe_size(img_normalized))

    # Elabora l'immagine normalizzata per trovare i contorni
    gray = cv.cvtColor(
        img_normalized, cv.COLOR_BGR2GRAY
    )  # Usa img_normalized per coerenza
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Crea copia per disegnare i contorni
    img_with_contours = img_normalized.copy()
    cv.drawContours(
        img_with_contours, contours, -1, (0, 255, 0), 2
    )  

    # Salva l'immagine con contorni temporaneamente per il confronto
    temp_contour_path = "temp/scarpa_contorni_temp.jpg"
    os.makedirs("temp", exist_ok=True)
    cv.imwrite(temp_contour_path, img_with_contours)

    last_shoe = Config.get_last_inserted_shoe()

    print("Cercando match nel database...")
    best_match, all_results = Match.find_best_match(temp_contour_path, "contorno")

    # Definisci soglia di similarità (es. 0.85 = 85% di similarità)
    SIMILARITY_THRESHOLD = 0.85

    if best_match and best_match["combined"] >= SIMILARITY_THRESHOLD:
        print(f"\n------------ MATCH TROVATO! -----------------")
        print(f"ID Scarpa: {best_match['id']}")
        print(f"Nome: {best_match['nome']}")
        print(f"Similarity Score: {best_match['similarity']:.4f}")
        print(f"Histogram Score: {best_match['histogram']:.4f}")
        print(
            f"Score Combinato: {best_match['combined']:.4f}"
        )

        # Mostra confronto visivo
        input_img = cv.imread(temp_contour_path)
        match_img = cv.imread(best_match["path_contorno"])

        if match_img is not None:
            fixed_size = (700, 1000)  # (width, height) – verticale standard
            extra_padding = (50, 50)  # Padding extra top/bottom

            # Ridimensiona mantenendo l'aspetto e aggiungi padding extra
            input_resized = resize_keep_aspect(input_img, fixed_size, extra_padding)
            match_resized = resize_keep_aspect(match_img, fixed_size, extra_padding)

            # Concatenazione orizzontale delle immagini
            comparison = np.hstack([input_resized, match_resized])

            # Mostra la finestra con le immagini modificate
            cv.imshow("Input vs Best Match (scaled)", comparison)
            cv.waitKey(0)
            cv.destroyAllWindows()

    else:
        print(
            f"\n❌ Nessun match significativo trovato (soglia: {SIMILARITY_THRESHOLD})"
        )
        print("💾 Creando nuovo record nel database...")

        # Salva l'immagine con contorni permanentemente
        permanent_contour_path = f"images/scarpa_contorni_{Config.get_next_id()}.jpg"
        cv.imwrite(
            permanent_contour_path, img_with_contours
        )  # Salva la versione normalizzata con contorni

        # Salva nel database
        shoe_name = input("Inserisci il nome della scarpa: ") or "Scarpa Sconosciuta"
        new_id = Config.save_to_database(shoe_name, path, permanent_contour_path)
        print(f"Nuova scarpa salvata con ID: {new_id}")

    # Pulisci il file temporaneo
    if os.path.exists(temp_contour_path):
        os.remove(temp_contour_path)

else:
    print("❌ Impossibile caricare l'immagine")
    sys.exit()