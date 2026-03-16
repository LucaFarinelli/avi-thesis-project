import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity
import os


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
        hsv1 = cv.cvtColor(img1_resized, cv.COLOR_BGR2HSV)
        hsv2 = cv.cvtColor(img2_resized, cv.COLOR_BGR2HSV)
        hist1 = cv.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist2 = cv.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv.normalize(hist1, hist1, 0, 1, cv.NORM_MINMAX)
        cv.normalize(hist2, hist2, 0, 1, cv.NORM_MINMAX)
        score = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
        return max(0.0, score)

    elif method == "orb":
        # Feature Matching (Trova punti chiave geometrici come i lacci o il logo)
        orb = cv.ORB_create(nfeatures=500)
        g1 = cv.cvtColor(img1_resized, cv.COLOR_BGR2GRAY)
        g2 = cv.cvtColor(img2_resized, cv.COLOR_BGR2GRAY)
        
        # Mascheriamo lo sfondo nero per non trovare keypoints nel nulla
        mask1 = cv.threshold(g1, 1, 255, cv.THRESH_BINARY)[1]
        mask2 = cv.threshold(g2, 1, 255, cv.THRESH_BINARY)[1]
        
        kp1, des1 = orb.detectAndCompute(g1, mask1)
        kp2, des2 = orb.detectAndCompute(g2, mask2)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return 0.0
            
        # Brute-Force Matcher con distanza di Hamming (ottimale per ORB)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Calcoliamo uno score basato sul numero e qualità dei match
        if len(matches) == 0:
            return 0.0
            
        distances = [m.distance for m in matches]
        good_matches = [m for m in matches if m.distance < 50] # Distanza stretta
        
        # Ponderiamo: numero di buoni match rispetto al potenziale massimo (500)
        match_ratio = len(good_matches) / min(len(kp1), len(kp2))
        
        return min(1.0, match_ratio * 3.0) # Moltiplicatore per scalare il ratio tra 0 e 1

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
