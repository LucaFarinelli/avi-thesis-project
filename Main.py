import cv2 as cv
import sys
import os
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.plugin=false' #OpenCv cerca  plugin QT Wayland ma sistema sta usando X11
import numpy as np
import Config
import Match
import Utils  # Nuovo: importa funzioni utili per calcolo taglia

def resize_keep_aspect(img, target_size, extra_padding=(0, 0)):
    '''
        Funzione che ridimensiona l'immagine di input in base ad una taglia specifica 
    '''
    h, w = img.shape[:2]                                                        #estrae altezza h e larghezza w
    target_h, target_w = target_size                                            #estrae dalla dimensione target relativamente h e w
    scale = min(target_w / w, target_h / h)                                     #calcola fattore di ridimensionamento come minimo dei rapporti tra w e h per garantire ridimensionato
    new_w = int(w * scale)                                                      #trovo la nuova w scalando quella originale e trasformando in intero
    new_h = int(h * scale)                                                      #trovo la nuova h scalando quella originale e trasformando in intero
    resized = cv.resize(img, (new_w, new_h))                                    #ridefinisco l'immagine basandomi sulle nuove dimensioni calcolate

    pad_h = target_h - new_h                                                    #calcola il padding totoale in altezza
    pad_w = target_w - new_w                                                    #calcola il padding totoale in lunghezza
    top = (pad_h // 2) + extra_padding[0]                                       #calcola riempimento superiore come metà del riempimento in altezza+riempimento sup. aggiuntivo
    bottom = pad_h - (pad_h // 2) + extra_padding[1]                            #calcola riempimento inferiore come metà del riempimento in altezza+riempimento inf. aggiuntivo
    left = pad_w // 2                                                           #calcola riempimento a sx come metà del riempimento in  w
    right = pad_w - left                                                        #calcola rimpimento a dx come come il riempimento in w - riempimento a sx già calcolato
    padded = cv.copyMakeBorder(                                                 #aggiungo bordi all'immagine ridimensionata usando il padding
        resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded

def normalize_shoe_image(img):
    '''
        Raddrizza la scarpa se inclinata, senza ritaglio pesante.
        Mantiene orientamento verticale e ridimensiona mantenendo aspetto.    
    '''
    img_copy = img.copy()
    gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return resize_keep_aspect(img_copy, (700, 1000))

    largest_contour = max(contours, key=cv.contourArea)
    rect = cv.minAreaRect(largest_contour)
    center, (width, height), angle = rect

    if width < height:
        angle = -angle
    else:
        angle = -(angle + 90)

    if angle < -90:
        angle += 180
    elif angle > 90:
        angle -= 180

    if abs(angle) < 10:
        rotated = img_copy
    else:
        (h, w) = img_copy.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv.warpAffine(img_copy, M, (w, h))

    final_img = resize_keep_aspect(rotated, (700, 1000))
    return final_img

cv.setUseOptimized(True)
shoes_path = input("Inserire nome scarpa da cercare: ")

path = f"images/{shoes_path}.jpg"
img = cv.imread(path)

if img is not None:
    img_normalized = normalize_shoe_image(img)

    # Nuovo: Calcola taglia usando funzioni utili dal precedente codice
    # Pre-processa per segmentazione
    preprocessed = Utils.preprocess(img_normalized)
    clustered = Utils.kMeans_cluster(preprocessed)
    edged = Utils.edgeDetection(clustered)
    boundRect, contours, contours_poly, _ = Utils.getBoundingBox(edged)
    
    if len(boundRect) < 1:
        print("Errore: Nessun contorno suola trovato. Verifica immagine.")
    else:
        # Calcola lunghezza suola in cm (usa A4 come scala)
        pcropedImg = img_normalized  # Usa normalizzata come "paper"
        sole_size_cm = Utils.calcFeetSize(pcropedImg, boundRect) / 10  # mm a cm
        print(f"Lunghezza suola: {sole_size_cm:.2f} cm")
        
        # Mappa a taglia
        size = Utils.getSizeFromLength(sole_size_cm)
        print(f"Taglia stimata: EU: {size['EU']}, US: {size['US']}, UK: {size['UK']}")

    # Procedi con estrazione contorni e match forma/DB (dal tuo codice originale)
    gray = cv.cvtColor(img_normalized, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    img_with_contours = img_normalized.copy()
    cv.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)

    temp_contour_path = "temp/scarpa_contorni_temp.jpg"
    os.makedirs("temp", exist_ok=True)
    cv.imwrite(temp_contour_path, img_with_contours)

    last_shoe = Config.get_last_inserted_shoe()

    best_match, all_results = Match.find_best_match(temp_contour_path, "contorno")

    SIMILARITY_THRESHOLD = 0.95

    if best_match and best_match["combined"] >= SIMILARITY_THRESHOLD:

        print(f"Best match details: {best_match}")  # Add this line to inspect the match object
        print(f"Combined similarity: {best_match['combined']}")

        print(f"Forma: {best_match['nome']}")

        input_img = cv.imread(temp_contour_path)
        match_img = cv.imread(best_match["path_contorno"])

        if match_img is not None:
            fixed_size = (700, 1000)
            extra_padding = (50, 50)
            input_resized = resize_keep_aspect(input_img, fixed_size, extra_padding)
            match_resized = resize_keep_aspect(match_img, fixed_size, extra_padding)
            comparison = np.hstack([input_resized, match_resized])
            cv.imshow("Input vs Best Match (scaled)", comparison)
            cv.waitKey(0)
            cv.destroyAllWindows()

    else:
        print(f"\n La suola non assomiglia a nessuna forma (soglia: {SIMILARITY_THRESHOLD})")
        print("Creando nuovo record nel database...")

        permanent_contour_path = f"images/scarpa_contorni_{Config.get_next_id()}.jpg"
        cv.imwrite(permanent_contour_path, img_with_contours)

        shoe_name = input("Inserisci il nome della scarpa: ") or "Scarpa Sconosciuta"
        new_id = Config.save_to_database(shoe_name, path, permanent_contour_path)

    if os.path.exists(temp_contour_path):
        os.remove(temp_contour_path)

else:
    print("Impossibile caricare l'immagine")
    sys.exit()