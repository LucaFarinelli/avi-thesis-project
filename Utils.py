from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np
import bisect
import config_settings as cfg


def generate_high_detail_reference(img, mask_rembg):
    """
    Crea un'immagine di riferimento ad alto dettaglio per il matching:
    - isola l'oggetto (sovrapposizione img originle con maschera rembg)
    - estrae i bordi interni e dettagli superficiali (Canny)
    - sovrappone i bordi in bianco sull'immagine originale a colori
    Restituisce l'immagine di riferimento pronta per il salvataggio o elaborazione ORB.
    """
    shoe_only = cv.bitwise_and(img, img, mask=mask_rembg)
    gray_shoe = cv.cvtColor(shoe_only, cv.COLOR_BGR2GRAY)
    internal_edges = cv.Canny(gray_shoe, 50, 150)
    
    reference_img = shoe_only.copy()
    reference_img[internal_edges > 0] = [255, 255, 255] # Sovrapponi bordi bianchi
    return reference_img


def resize_keep_aspect(img, target_size, extra_padding=(0, 0)):
    """Ridimensiona l'immagine mantenendo il rapporto d'aspetto e aggiungendo padding."""
    h, w = img.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv.resize(img, (new_w, new_h))

    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top = (pad_h // 2) + extra_padding[0]
    bottom = pad_h - (pad_h // 2) + extra_padding[1]
    left = pad_w // 2
    right = pad_w - left

    padded = cv.copyMakeBorder(
        resized, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded


def preprocess(img):
    """Applica normalizzazione e blur gaussiano all'immagine."""
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.GaussianBlur(img, (5, 5), 0)
    img = img / 255.0

    img_uint8 = (img * 255).astype(np.uint8)
    cv.imwrite('debug1/preprocessed.jpg', img_uint8)

    return img

def get_pixel(img, center, x, y):
    """Restituisce 1 se il pixel in (x, y) è >= al valore centrale, altrimenti 0."""
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    """Calcola il valore LBP del pixel (x, y) confrontando i suoi 8 vicini con il valore centrale."""
    center = img[x][y]
    val_ar = []

    # top-left, top, top-right, right, bottom-right, bottom, bottom-left, left
    val_ar.append(get_pixel(img, center, x-1, y-1))
    val_ar.append(get_pixel(img, center, x-1, y))
    val_ar.append(get_pixel(img, center, x-1, y+1))
    val_ar.append(get_pixel(img, center, x, y+1))
    val_ar.append(get_pixel(img, center, x+1, y+1))
    val_ar.append(get_pixel(img, center, x+1, y))
    val_ar.append(get_pixel(img, center, x+1, y-1))
    val_ar.append(get_pixel(img, center, x, y-1))

    # converto numeri binari in decimali
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]

    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val

def lbp_vectorized(img):
    """Calcola la mappa LBP dell'immagine in modo vettorializzato (molto più veloce)."""
    h, w = img.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    
    # Spostamenti per i 8 vicini
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    powers = [1, 2, 4, 8, 16, 32, 64, 128]
    
    for i, ((dy, dx), p) in enumerate(zip(offsets, powers)):
        # Crea una versione traslata dell'immagine
        # Usiamo il padding per gestire i bordi
        shifted = np.roll(img, shift=(-dy, -dx), axis=(0, 1))
        
        # Confronto con il centro
        mask = (shifted >= img).astype(np.uint8)
        lbp += mask * p
        
    return lbp

def calc_lbp_histogram(img_lbp, mask=None):
    """Calcola l'istogramma normalizzato della mappa LBP, opzionalmente usando una maschera."""
    if mask is not None:
        # Estrai solo i pixel coperti dalla maschera
        vals = img_lbp[mask > 0]
    else:
        vals = img_lbp.ravel()
        
    if len(vals) == 0:
        return np.zeros(256, dtype=np.float32)

    # Calcola l'istogramma (256 bin per i valori 0-255)
    hist, _ = np.histogram(vals, bins=256, range=(0, 256))
    
    # Normalizza l'istogramma affinché la somma sia 1 (indipendente dalla risoluzione)
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return hist

def extract_svm_features(img, mask_rembg=None):
    """
    Estrae un vettore di feature fuso (Texture + Colore) per l'addestramento SVM.
    Unisce l'istogramma LBP (256 bin) con un istogramma del colore HSV (60 bin).
    Maschera lo sfondo nero in modo da considerare solo i pixel della scarpa.
    
    Returns:
        Vettore 1D (numpy array) contenente i valori concatenati normalizzati.
    """
    if mask_rembg is None:
        mask_rembg = get_shoe_mask(img)
        
    # 1. Feature TEXTURE (LBP)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    lbp_map = lbp_vectorized(gray)
    hist_lbp = calc_lbp_histogram(lbp_map, mask=mask_rembg)
    
    # 2. Feature COLORE (HSV Histogram)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Calcolo istogramma 2D su H (Tono, 30 bin) e S (Saturazione, 32 bin)
    # Ignoriamo il canale V (Luminosità) per renderlo resistente alle ombre
    hist_hsv = cv.calcHist([hsv], [0, 1], mask_rembg, [30, 32], [0, 180, 0, 256])
    cv.normalize(hist_hsv, hist_hsv, 0, 1, cv.NORM_MINMAX)
    hist_hsv_flat = hist_hsv.flatten()
    
    # 3. Fusione (Concatenazione)
    # Diamo un piccolo peso extra al colore visto che LBP ha molti più bin (256 vs 960 appiattiti)
    # Per bilanciarli meglio, limitiamo i bin HSV o li scaliamo. 
    # Optiamo per Hist 1D per H e S separati per mantenere il vettore piccolo (30 + 32 = 62 bin)
    hist_h = cv.calcHist([hsv], [0], mask_rembg, [30], [0, 180])
    hist_s = cv.calcHist([hsv], [1], mask_rembg, [32], [0, 256])
    
    cv.normalize(hist_h, hist_h, 0, 1, cv.NORM_MINMAX)
    cv.normalize(hist_s, hist_s, 0, 1, cv.NORM_MINMAX)
    
    # Crea vettore finale: LBP(256) + H(30) + S(32) + Geometria(6) = 324 dimensioni
    
    # 3. Feature GEOMETRICHE / STRUTTURALI
    # Fondamentali per rilevare pieghe, rotture, deformazioni e irregolarità del contorno
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # 3a. Edge Density: rapporto tra pixel di bordo e pixel totali dell'oggetto
    # Un oggetto rotto/strappato ha più bordi interni
    shoe_gray = cv.bitwise_and(gray, gray, mask=mask_rembg)
    edges = cv.Canny(shoe_gray, 50, 150)
    total_pixels = np.count_nonzero(mask_rembg)
    edge_pixels = np.count_nonzero(edges)
    edge_density = edge_pixels / (total_pixels + 1e-7)
    
    # 3b. Compattezza del contorno: quanto è "regolare" la forma
    # Oggetti integri hanno contorni lisci, piegati/rotti hanno contorni frastagliati
    contours, _ = cv.findContours(mask_rembg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    compactness = 0.0
    solidity = 0.0
    extent = 0.0
    aspect_ratio_feat = 0.0
    symmetry = 0.0
    
    if contours:
        c = max(contours, key=cv.contourArea)
        area = cv.contourArea(c)
        perimeter = cv.arcLength(c, True)
        compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-7) # 1.0 = cerchio perfetto
        
        # 3c. Solidità: area / area convex hull (oggetto schiacciato ha bassa solidità)
        hull = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        solidity = area / (hull_area + 1e-7)
        
        # 3d. Aspect Ratio del bounding box
        x, y, w, h = cv.boundingRect(c)
        aspect_ratio_feat = max(w, h) / (min(w, h) + 1e-7)
        
        # 3e. Extent: rapporto area oggetto / area bounding box
        extent = area / (w * h + 1e-7)
        
        # 3f. Simmetria verticale: confronto L/R della maschera
        mid_x = mask_rembg.shape[1] // 2
        left_half = np.count_nonzero(mask_rembg[:, :mid_x])
        right_half = np.count_nonzero(mask_rembg[:, mid_x:])
        symmetry = 1.0 - abs(left_half - right_half) / (total_pixels + 1e-7)
    
    geo_features = np.array([edge_density, compactness, solidity, aspect_ratio_feat, extent, symmetry])
    
    combined_features = np.concatenate([hist_lbp, hist_h.flatten(), hist_s.flatten(), geo_features])
    
    return combined_features

def get_shoe_mask(img):
    """Crea una maschera binaria che isola la scarpa dallo sfondo in modo robusto."""
    h, w = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # 1. Edge-based mask (ottima per oggetti con bordi chiari)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blurred, 30, 100)
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv.dilate(edged, kernel, iterations=3)
    closed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel, iterations=3)
    
    # 2. Otsu-threshold mask (ottima per sfondi uniformi)
    _, otsu = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    # Se Otsu prende troppo sfondo, inverti
    if cv.countNonZero(otsu) > (h * w * 0.7):
        otsu = cv.bitwise_not(otsu)
        
    # Combiniamo le due maschere (OR logico)
    combined = cv.bitwise_or(closed, otsu)
    
    # 3. Pulizia e selezione del contorno più "scarpa"
    contours, _ = cv.findContours(combined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.ones((h, w), dtype=np.uint8) * 255 # Fallback: tutto
        
    # Filtriamo i contorni per area e posizione (deve essere centrale e grande)
    best_cnt = None
    max_score = -1
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < (h * w * 0.1): continue # Almeno il 10% dell'immagine
        
        # Calcolo baricentro
        M = cv.moments(cnt)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Score: più grande è e più è centrale, meglio è
        dist_center = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
        score = area / (dist_center + 1)
        
        if score > max_score:
            max_score = score
            best_cnt = cnt
            
    if best_cnt is not None:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv.drawContours(mask, [best_cnt], -1, 255, -1)
        # Operazioni morfologiche finali per smussare i bordi
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        return mask
    else:
        # Se non troviamo nulla di buono, usiamo Otsu pulito
        return otsu
    
def kMeans_cluster(img, n_clusters=3):
    """Esegue clustering K-Means sull'immagine per semplificare la segmentazione."""
    image_2D = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_2D)
    clustOut = kmeans.cluster_centers_[kmeans.labels_]
    clustered_3D = clustOut.reshape(img.shape[0], img.shape[1], img.shape[2])
    clusteredImg = np.uint8(clustered_3D * 255)
    cv.imwrite('debug1/clustered.jpg', clusteredImg)
    return clusteredImg


def edgeDetection(clusteredImage):
    """Rileva i bordi su un'immagine clusterizzata tramite Canny e morfologia."""
    gray = cv.cvtColor(clusteredImage, cv.COLOR_BGR2GRAY)
    edged = cv.Canny(gray, 50, 150)
    edged = cv.dilate(edged, None, iterations=2)
    edged = cv.erode(edged, None, iterations=1)
    cv.imwrite('debug1/edged.jpg', edged)
    return edged


def getBoundingBox(img):
    """Trova i contorni e i bounding box, ignorando quelli che toccano i bordi dell'immagine o sono sproporzionati."""
    h, w = img.shape[:2]
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    margin = 5 # Pixel di tolleranza dal bordo
    
    for cnt in contours:
        x, y, cw, ch = cv.boundingRect(cnt)
        area = cv.contourArea(cnt)
        
        # 1. Filtro Area: deve essere una frazione sensata della scarpa (es. almeno 5%)
        # Ma non deve essere l'intero foglio (es. massimo 90%)
        if area > (h * w * 0.90): continue 
        if area < (h * w * 0.05): continue 
        
        # 2. Se tocca quasi tutti i bordi contemporaneamente, e' sicuramente il perimetro del foglio
        touching_edges = 0
        if x <= margin: touching_edges += 1
        if y <= margin: touching_edges += 1
        if (x + cw) >= (w - margin): touching_edges += 1
        if (y + ch) >= (h - margin): touching_edges += 1
        
        if touching_edges >= 3: continue # Se tocca 3 o 4 bordi, è il foglio
        
        valid_contours.append(cnt)
        
    valid_contours = sorted(valid_contours, key=lambda x: cv.contourArea(x), reverse=True)
    
    contours_poly = [None] * len(valid_contours)
    boundRect = [None] * len(valid_contours)
    for i, c in enumerate(valid_contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])

    return boundRect, valid_contours, contours_poly, img


def calc_industrial_size(mask_rembg, pixel_per_mm=2.5):
    """
    Simula il calcolo dimensioni di un setup industriale AVI con telecamera fissa.
    Calcola la lunghezza e larghezza stimata della suola in mm ricavando il Bounding Box 
    dalla maschera prodotta da rembg.
    
    Args:
        mask_rembg: Maschera binaria (es. canale alpha di rembg).
        pixel_per_mm: Fattore di calibrazione precalcolato (Pixel/mm).
                      Simula Z (distanza telecamera) costante.
    
    Returns:
        length_mm, width_mm, bounding_box_coords (x, y, w, h)
    """
    contours, _ = cv.findContours(mask_rembg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0, 0.0, None
        
    # Troviamo il contorno della scarpa (il più grande)
    c = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(c)
    
    # In una scarpa, la dimensione maggiore è sempre la lunghezza
    length_px = max(w, h)
    width_px = min(w, h)
    
    length_mm = length_px / pixel_per_mm
    width_mm = width_px / pixel_per_mm
    
    return length_mm, width_mm, (x, y, w, h)


# Tabella di corrispondenza lunghezza suola (cm) -> taglia scarpa
SHOE_SIZES = [
    (23.5, {'EU': 37, 'US': 5,    'UK': 4.5}),
    (24.0, {'EU': 38, 'US': 6,    'UK': 5.5}),
    (24.5, {'EU': 39, 'US': 6.5,  'UK': 6}),
    (25.0, {'EU': 40, 'US': 7,    'UK': 6.5}),
    (25.5, {'EU': 41, 'US': 8,    'UK': 7.5}),
    (26.0, {'EU': 42, 'US': 9,    'UK': 8.5}),
    (26.5, {'EU': 43, 'US': 10,   'UK': 9.5}),
    (27.0, {'EU': 44, 'US': 10.5, 'UK': 10}),
    (27.5, {'EU': 45, 'US': 11,   'UK': 10.5}),
    (28.0, {'EU': 46, 'US': 12,   'UK': 11.5}),
]


def getSizeFromLength(length_cm):
    """Mappa la lunghezza della suola (in cm) alla taglia scarpa più vicina."""
    lengths = [row[0] for row in SHOE_SIZES]
    idx = bisect.bisect_left(lengths, length_cm)
    if idx == 0:
        return SHOE_SIZES[0][1]
    elif idx == len(lengths):
        return SHOE_SIZES[-1][1]
    else:
        if abs(length_cm - lengths[idx - 1]) < abs(length_cm - lengths[idx]):
            return SHOE_SIZES[idx - 1][1]
        else:
            return SHOE_SIZES[idx][1]