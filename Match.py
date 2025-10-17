import os
import mysql.connector
import cv2 as cv
from skimage.metrics import structural_similarity
from Compare import compare_images


def find_best_match(
    input_image_path, comparison_type="contorno", exclude_recent_seconds=10
):
    """Trova la scarpa più simile nel database, escludendo quelle inserite di recente"""

    conn = mysql.connector.connect(
        host="127.0.0.1", user="luca", password="password", database="SCARPE"
    )
    cursor = conn.cursor()

    # Esclude scarpe inserite negli ultimi X minuti
    query = """
    SELECT idScarpa, nome_scarpa, path_originale, path_contorno
    FROM dataset
    WHERE created_at < DATE_SUB(NOW(), INTERVAL %s SECOND)
    """
    cursor.execute(query, (exclude_recent_seconds,))
    shoes = cursor.fetchall()
    conn.close()

    print(
        f"Trovate {len(shoes)} scarpe nel database (escludendo ultimi {exclude_recent_seconds} secondi)"
    )

    best_match = None
    best_score = -1.0
    results = []

    for shoe in shoes:
        shoe_id, nome, path_orig, path_cont = shoe

        # Scegli quale immagine confrontare
        compare_path = path_cont if comparison_type == "contorno" else path_orig

        if os.path.exists(compare_path):
            print(f"Confrontando con: {compare_path}")

            # Metodi di confronto semplificati (solo contorno)
            sim_score = compare_images(input_image_path, compare_path, "similarity")
            hist_score = compare_images(
                input_image_path, compare_path, "histogram"
            )  # Riaggiunto per robustezza

            # DEBUG: Confronta direttamente senza preprocessing
            img1_debug = cv.imread(input_image_path, cv.IMREAD_GRAYSCALE)
            img2_debug = cv.imread(compare_path, cv.IMREAD_GRAYSCALE)
            if img1_debug is not None and img2_debug is not None:
                img1_res = cv.resize(img1_debug, (500, 900))
                img2_res = cv.resize(img2_debug, (500, 900))
                direct_ssim, _ = structural_similarity(img1_res, img2_res, full=True)
                print(f"Direct SSIM (solo resize): {direct_ssim:.4f}")

            # Score combinato semplice: media tra similarità e istogramma per contorno
            combined_score = (
                sim_score + hist_score
            ) / 2  # Solo contorno esterno, senza dettagli

            print(f" Similarity (contorno): {sim_score:.4f}")
            print(f" Histogram (contorno): {hist_score:.4f}")
            print(f" Score contorno combinato: {combined_score:.4f}")

            results.append(
                {
                    "id": shoe_id,
                    "nome": nome,
                    "similarity": sim_score,
                    "histogram": hist_score,  # Riaggiunto
                    "combined": combined_score,
                    "path_originale": path_orig,
                    "path_contorno": path_cont,
                }
            )

            if combined_score > best_score:
                best_score = combined_score
                best_match = results[-1]

        else:
            print(f"File non trovato: {compare_path}")

    # Ordina per score combinato (ora semplice media per contorno)
    results.sort(key=lambda x: x["combined"], reverse=True)

    print(f"Miglior score contorno trovato: {best_score:.4f}")

    return best_match, results
