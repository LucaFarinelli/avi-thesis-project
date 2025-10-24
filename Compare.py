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
    #print(f"DEBUG: Dimensioni originali - img1: {img1.shape}, img2: {img2.shape}")

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
        #print(f"DEBUG SSIM: {score:.4f}")
        return score
    elif method == "histogram":
        hist1 = cv.calcHist([img1_final], [0], None, [256], [0, 256])
        hist2 = cv.calcHist(
            [img1_final], [0], None, [256], [0, 256]
        ) 
        score = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
        #print(f"DEBUG Histogram: {score:.4f}")
        return score

