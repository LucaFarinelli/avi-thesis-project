import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity
import os


def compare_images(img1_path, img2_path, method="similarity"):
    """Confronta due immagini con preprocessing semplificato per contorni"""

    # Carica le immagini
    img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return 0.0

    # DEBUG: Salva originali per confronto
    debug_dir = "debug"
    os.makedirs(debug_dir, exist_ok=True)
    cv.imwrite(f"{debug_dir}/img1_original.png", img1)
    cv.imwrite(f"{debug_dir}/img2_original.png", img2)
    print(f"DEBUG: Dimensioni originali - img1: {img1.shape}, img2: {img2.shape}")

    # Ridimensionamento condizionale
    target_size = (500, 900)
    # Ridimensiona solo se le dimensioni differiscono significativamente
    if abs(img1.shape[0] - target_size[1]) > 50 or abs(img1.shape[1] - target_size[0]) > 50:
        img1_resized = cv.resize(img1, target_size)
    else:
        img1_resized = img1
    if abs(img2.shape[0] - target_size[1]) > 50 or abs(img2.shape[1] - target_size[0]) > 50:
        img2_resized = cv.resize(img2, target_size)
    else:
        img2_resized = img2

    # DEBUG: Salva ridimensionate
    cv.imwrite(f"{debug_dir}/img1_resized.png", img1_resized)
    cv.imwrite(f"{debug_dir}/img2_resized.png", img2_resized)

    # Opzionale: leggero blur per ridurre rumore minimo (se necessario)
    img1_final = cv.GaussianBlur(img1_resized, (3, 3), 0)  # Blur ridotto
    img2_final = cv.GaussianBlur(img2_resized, (3, 3), 0)

    # DEBUG: Salva finali
    cv.imwrite(f"{debug_dir}/img1_final.png", img1_final)
    cv.imwrite(f"{debug_dir}/img2_final.png", img2_final)

    if method == "similarity":
        score, _ = structural_similarity(img1_final, img2_final, full=True)
        print(f"DEBUG SSIM: {score:.4f}")
        return score
    elif method == "histogram":
        hist1 = cv.calcHist([img1_final], [0], None, [256], [0, 256])
        hist2 = cv.calcHist(
            [img1_final], [0], None, [256], [0, 256]
        ) 
        score = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
        print(f"DEBUG Histogram: {score:.4f}")
        return score


# Commentato: funzione che estrae dettagli avanzati (keypoints) della scarpa
# def compare_features(img1_path, img2_path, method="orb"):
#     """Confronta usando feature detection (invariante a scala e rotazione)"""
#
#     img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
#     img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)
#
#     if img1 is None or img2 is None:
#         print(f"❌ Impossibile caricare immagini: {img1_path} o {img2_path}")
#         return 0.0
#
#     try:
#         if method == "orb":
#             # ORB (Oriented FAST and Rotated BRIEF) - gratuito
#             detector = cv.ORB_create(nfeatures=500)
#         elif method == "sift":
#             # SIFT - più preciso ma può richiedere opencv-contrib-python
#             detector = cv.SIFT_create()
#         else:
#             print(f"❌ Metodo non supportato: {method}")
#             return 0.0
#
#         # Trova keypoints e descriptors
#         kp1, des1 = detector.detectAndCompute(img1, None)
#         kp2, des2 = detector.detectAndCompute(img2, None)
#
#         if des1 is None or des2 is None:
#             print("⚠️ Nessun descriptor trovato in una delle immagini")
#             return 0.0
#
#         print(f"🔍 Keypoints trovati: img1={len(kp1)}, img2={len(kp2)}")
#
#         # Matcher
#         if method == "orb":
#             bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
#         else:
#             bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
#
#         matches = bf.match(des1, des2)
#
#         # Filtra i match buoni
#         if len(matches) < 10:  # Troppo pochi match
#             print(f"⚠️ Troppo pochi match trovati: {len(matches)}")
#             return 0.0
#
#         # Ordina per distanza
#         matches = sorted(matches, key=lambda x: x.distance)
#
#         # Calcola score basato sui match buoni
#         distance_threshold = 50 if method == "orb" else 0.75
#         good_matches = [m for m in matches if m.distance < distance_threshold]
#
#         print(f"📊 Match totali: {len(matches)}, Match buoni: {len(good_matches)}")
#
#         if len(good_matches) == 0:
#             return 0.0
#
#         # Score basato sulla percentuale di buoni match
#         score = len(good_matches) / max(len(kp1), len(kp2))
#
#         return min(score, 1.0)  # Normalizza tra 0 e 1
#
#     except Exception as e:
#         print(f"❌ Errore durante feature matching: {e}")
#         return 0.0


# import numpy as np  # Aggiungi in alto


# Commentato: funzione ausiliaria per soglia dinamica nei dettagli
# def auto_threshold_features(matches, method="orb"):
#     """Calcola soglia dinamica basata sulla distribuzione delle distanze"""
#     if not matches:
#         return 0
#
#     distances = [m.distance for m in matches]
#
#     if method == "orb":
#         # Per ORB, usa percentile
#         threshold = np.percentile(distances, 75)  # 75° percentile
#         return max(min(threshold, 50), 10)  # Max 50
#     else:
#         # Per SIFT, usa media + deviazione standard
#         mean_dist = np.mean(distances)
#         std_dist = np.std(distances)
#         return max(mean_dist + 0.5 * std_dist, 0.1)


# Commentato: versione dinamica che estrae dettagli avanzati
# def compare_features_dynamic(img1_path, img2_path, method="orb"):
#     """Versione con soglia dinamica"""
#
#     img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
#     img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)
#
#     if img1 is None or img2 is None:
#         return 0.0
#
#     try:
#         if method == "orb":
#             detector = cv.ORB_create(nfeatures=500)
#             bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
#         elif method == "sift":
#             detector = cv.SIFT_create()
#             bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
#
#         kp1, des1 = detector.detectAndCompute(img1, None)
#         kp2, des2 = detector.detectAndCompute(img2, None)
#
#         if des1 is None or des2 is None:
#             return 0.0
#
#         matches = bf.match(des1, des2)
#
#         if len(matches) < 10:
#             return 0.0
#
#         matches = sorted(matches, key=lambda x: x.distance)
#
#         # USA SOGLIA DINAMICA
#         dynamic_threshold = auto_threshold_features(matches, method)
#         good_matches = [m for m in matches if m.distance < dynamic_threshold]
#
#         print(f"🎯 Soglia dinamica: {dynamic_threshold:.2f}")
#         print(f"📊 Match totali: {len(matches)}, Match buoni: {len(good_matches)}")
#
#         if len(good_matches) == 0:
#             return 0.0
#
#         score = len(good_matches) / max(len(kp1), len(kp2))
#         return min(score, 1.0)
#
#     except Exception as e:
#         print(f"❌ Errore: {e}")
#         return 0.0
