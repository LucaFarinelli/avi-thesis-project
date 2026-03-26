import os
import mysql.connector
import cv2 as cv
from skimage.metrics import structural_similarity
from Compare import compare_images
import config_settings as cfg


def find_best_match(input_image_path, comparison_type="contorno", exclude_recent_seconds=10):
    """Trova la scarpa più simile nel database.

    Esclude le scarpe inserite negli ultimi `exclude_recent_seconds` secondi
    per evitare che la scarpa appena salvata venga confrontata con se stessa.

    Args:
        input_image_path: Path all'immagine di input (contorno o originale).
        comparison_type: "contorno" per confrontare i contorni, altrimenti usa l'immagine originale.
        exclude_recent_seconds: Intervallo in secondi entro cui escludere scarpe inserite di recente.

    Returns:
        Tupla (best_match, all_results) dove best_match è un dizionario con i dati
        della scarpa più simile, o None se nessuna scarpa è nel database.
    """
    from Config import DB_CONFIG
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    query = """
    SELECT idScarpa, nome_scarpa, path_originale, path_contorno
    FROM dataset
    WHERE created_at < DATE_SUB(NOW(), INTERVAL %s SECOND)
    """
    cursor.execute(query, (exclude_recent_seconds,))
    shoes = cursor.fetchall()
    conn.close()

    best_match = None
    best_score = -1.0
    results = []

    for shoe in shoes:
        shoe_id, nome, path_orig, path_cont = shoe

        compare_path = path_cont if comparison_type == "contorno" else path_orig

        if not os.path.exists(compare_path):
            print(f"File non trovato: {compare_path}")
            continue

        sim_score = compare_images(input_image_path, compare_path, "similarity")
        hist_score = compare_images(input_image_path, compare_path, "histogram")
        
        # Le due nuove metriche strutturali per evitare match tra Top vs Suola
        orb_score = compare_images(input_image_path, compare_path, "orb")
        shape_score = compare_images(input_image_path, compare_path, "shape")
        
        # Penalizzazione Strutturale Inversa
        str_penalty = 1.0
        if shape_score < cfg.SHAPE_PENALTY_THRESHOLD or orb_score < cfg.ORB_PENALTY_THRESHOLD:
            str_penalty = cfg.PENALTY_FACTOR
            
        # Media pesata strutturata dai config (Diamo più peso a Colore e SSIM, ma ORB/Shape fungono da validatori)
        combined_score = (
            (sim_score * cfg.WEIGHT_SIMILARITY) +
            (hist_score * cfg.WEIGHT_HISTOGRAM) +
            (orb_score * cfg.WEIGHT_ORB) +
            (shape_score * cfg.WEIGHT_SHAPE)
        ) * str_penalty

        results.append({
            "id": shoe_id,
            "nome": nome,
            "similarity": sim_score,
            "histogram": hist_score,
            "orb": orb_score,
            "shape": shape_score,
            "combined": combined_score,
            "path_originale": path_orig,
            "path_contorno": path_cont,
        })

        if combined_score > best_score:
            best_score = combined_score
            best_match = results[-1]

    results.sort(key=lambda x: x["combined"], reverse=True)

    return best_match, results
