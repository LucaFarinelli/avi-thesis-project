import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity
import os
import config_settings as cfg


def compare_images(img1_path, img2_path, method="similarity"):
    """
    Confronta due immagini di scarpe (cutout su fondo nero).
    Metodi disponibili:
    - 'similarity': SSIM (analisi pixel grigi strutturali)
    - 'histogram': Istogramma HSV (analisi colorimetrica)
    - 'orb': Feature Matching (angoli, bordi interni, lacci)
    - 'shape': Hu Moments (analisi sagoma e silhouette)
    """

    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)

    if img1 is None or img2 is None:
        return 0.0

    target_size = (700, 1000) # Coerente con TARGET_WIDTH/HEIGHT di Main.py
    img1_resized = cv.resize(img1, target_size)
    img2_resized = cv.resize(img2, target_size)

    if method == "similarity":
        g1 = cv.cvtColor(img1_resized, cv.COLOR_BGR2GRAY)
        g2 = cv.cvtColor(img2_resized, cv.COLOR_BGR2GRAY)
        g1 = cv.GaussianBlur(g1, (3, 3), 0)
        g2 = cv.GaussianBlur(g2, (3, 3), 0)
        score, _ = structural_similarity(g1, g2, full=True)
        return max(0.0, score)
        
    elif method == "histogram":
        # Calcola maschere per escludere lo sfondo nero (pixel dove tutti i canali sono quasi 0)
        mask1 = cv.threshold(cv.cvtColor(img1_resized, cv.COLOR_BGR2GRAY), 5, 255, cv.THRESH_BINARY)[1]
        mask2 = cv.threshold(cv.cvtColor(img2_resized, cv.COLOR_BGR2GRAY), 5, 255, cv.THRESH_BINARY)[1]
        
        hsv1 = cv.cvtColor(img1_resized, cv.COLOR_BGR2HSV)
        hsv2 = cv.cvtColor(img2_resized, cv.COLOR_BGR2HSV)
        hist1 = cv.calcHist([hsv1], [0, 1], mask1, [50, 60], [0, 180, 0, 256])
        hist2 = cv.calcHist([hsv2], [0, 1], mask2, [50, 60], [0, 180, 0, 256])
        cv.normalize(hist1, hist1, 0, 1, cv.NORM_MINMAX)
        cv.normalize(hist2, hist2, 0, 1, cv.NORM_MINMAX)
        score = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
        return max(0.0, score)

    elif method == "orb":
        # Feature Matching (Trova punti chiave geometrici e dettagli come bordi, pieghe, spigoli)
        orb = cv.ORB_create(nfeatures=cfg.ORB_NFEATURES)
        g1 = cv.cvtColor(img1_resized, cv.COLOR_BGR2GRAY)
        g2 = cv.cvtColor(img2_resized, cv.COLOR_BGR2GRAY)
        
        # Mascheriamo lo sfondo nero per non trovare keypoints nel nulla
        mask1 = cv.threshold(g1, 1, 255, cv.THRESH_BINARY)[1]
        mask2 = cv.threshold(g2, 1, 255, cv.THRESH_BINARY)[1]
        
        kp1, des1 = orb.detectAndCompute(g1, mask1)
        kp2, des2 = orb.detectAndCompute(g2, mask2)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return 0.0
            
        # Brute-Force Matcher basato su nearest-neighbor multipli (KNN)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test: un match è valido solo se è il migliore in modo netto (non ambiguo)
        good_matches = []
        for m, n in matches:
            if m.distance < cfg.ORB_RATIO_THRESHOLD * n.distance:
                good_matches.append(m)
                
        if len(good_matches) == 0:
            return 0.0
            
        # Ponderiamo: numero di match robusti identificati sul totale possibile e ne moltiplichiamo il peso metrico
        match_ratio = len(good_matches) / max(len(kp1), len(kp2), 1)
        score = min(1.0, match_ratio * 3.0) # Scalato in base alla "ricchezza" di punti buoni
        return score

    elif method == "shape":
        # Estrarre la silhouette e confrontare i momenti di Hu (Invarianti a scala e rotazione)
        g1 = cv.cvtColor(img1_resized, cv.COLOR_BGR2GRAY)
        g2 = cv.cvtColor(img2_resized, cv.COLOR_BGR2GRAY)
        
        _, thresh1 = cv.threshold(g1, 5, 255, cv.THRESH_BINARY)
        _, thresh2 = cv.threshold(g2, 5, 255, cv.THRESH_BINARY)
        
        cnts1, _ = cv.findContours(thresh1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts2, _ = cv.findContours(thresh2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if not cnts1 or not cnts2:
            return 0.0
            
        c1 = max(cnts1, key=cv.contourArea)
        c2 = max(cnts2, key=cv.contourArea)
        
        # cv.matchShapes restituisce una misura di "diversità", quindi 0 significa perfette.
        # Inverto per farla diventare una similarità percentuale.
        similarity = cv.matchShapes(c1, c2, cv.CONTOURS_MATCH_I1, 0)
        
        # Normalizzazione heuristica (valori empirici: similarity 0=100%, similarity > 0.5=0%)
        score = max(0.0, 1.0 - (similarity * 2))
        return score

    return 0.0


def get_ssim_diff(img1_path, img2_path):
    """
    Calcola la mappa di differenza SSIM tra due immagini.
    Ritorna un'immagine (0-255) dove il bianco indica identità e il nero differenza.
    """
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)
    if img1 is None or img2 is None:
        return None

    target_size = (700, 1000)
    g1 = cv.cvtColor(cv.resize(img1, target_size), cv.COLOR_BGR2GRAY)
    g2 = cv.cvtColor(cv.resize(img2, target_size), cv.COLOR_BGR2GRAY)
    
    # Gaussian blur per ridurre il rumore come descritto nel testo della tesi
    g1 = cv.GaussianBlur(g1, (3, 3), 0)
    g2 = cv.GaussianBlur(g2, (3, 3), 0)

    # full=True restituisce la mappa di similarità locale (valori tra -1 e 1)
    _, diff = structural_similarity(g1, g2, full=True)
    
    # Trasformiamo la mappa (-1, 1) in un'immagine (0, 255)
    diff = (diff + 1.0) / 2.0  # Range [0, 1]
    diff = (diff * 255).astype("uint8")
    
    return diff
