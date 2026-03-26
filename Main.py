import cv2 as cv
import sys
import shutil
import os
import time
import numpy as np

# Impostiamo il backend di matplotlib in modalità non-interattiva
# per evitare crash quando viene chiamato da thread in background (es. Tkinter)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import joblib

os.environ['QT_LOGGING_RULES'] = 'qt.qpa.plugin=false'

import Config
import Match
import Utils
import Homography
import Compare
from rembg import remove
import config_settings as cfg

cv.setUseOptimized(True)

# Dimensioni standard per l'elaborazione delle immagini dalla configurazione globale
TARGET_WIDTH = cfg.TARGET_WIDTH
TARGET_HEIGHT = cfg.TARGET_HEIGHT

# Soglia minima di similarità combinata per considerare un match valido
SIMILARITY_THRESHOLD = cfg.SIMILARITY_THRESHOLD

# Percorso del modello SVM per i difetti
MODEL_PATH = cfg.MODEL_PATH
SCALER_PATH = cfg.SCALER_PATH


def capture_from_webcam():
    """Cattura un'immagine dalla webcam con anteprima e guide visive.

    Returns:
        Tupla (image, path) con l'immagine acquisita e il suo percorso di salvataggio,
        oppure (None, None) se la cattura viene annullata.
    """
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Errore: impossibile aprire la webcam.")
        print("Controlla che la webcam sia collegata e non sia usata da altri programmi.")
        return None, None

    print("\n" + "=" * 50)
    print("MODALITA' WEBCAM ATTIVATA")
    print("=" * 50)
    print("\nISTRUZIONI:")
    print("1. Inquadra la scarpa nel rettangolo bianco")
    print("2. Premi SPACE o ENTER per scattare")
    print("3. Premi 'q' sulla finestra per uscire")
    print("\n" + "=" * 50)

    target_aspect = TARGET_HEIGHT / TARGET_WIDTH

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Errore nella cattura del frame.")
            break

        h, w = frame.shape[:2]
        current_aspect = h / w

        if current_aspect > target_aspect:
            new_h = int(w * target_aspect)
            y_start = (h - new_h) // 2
            cropped = frame[y_start:y_start + new_h, :]
        else:
            new_w = int(h / target_aspect)
            x_start = (w - new_w) // 2
            cropped = frame[:, x_start:x_start + new_w]

        preview = cv.resize(cropped, (TARGET_WIDTH // 2, TARGET_HEIGHT // 2))

        preview_display = np.full(
            (TARGET_HEIGHT // 2 + 100, TARGET_WIDTH // 2, 3),
            (50, 50, 50), dtype=np.uint8
        )
        preview_display[50:50 + (TARGET_HEIGHT // 2), :] = preview

        cv.rectangle(
            preview,
            (50, 50), (preview.shape[1] - 50, preview.shape[0] - 50),
            (255, 255, 255), 2
        )

        center_x = preview.shape[1] // 2
        center_y = preview.shape[0] // 2
        cv.line(preview, (center_x, 0), (center_x, preview.shape[0]), (0, 255, 0), 1)
        cv.line(preview, (0, center_y), (preview.shape[1], center_y), (0, 255, 0), 1)

        cv.putText(preview_display, "ANTEPRIMA WEBCAM", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv.putText(preview_display, f"Risoluzione: {w}x{h}",
                   (10, preview_display.shape[0] - 70),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv.putText(preview_display, "Premi SPACE/ENTER per scattare",
                   (10, preview_display.shape[0] - 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.putText(preview_display, "Premi 'q' per uscire",
                   (10, preview_display.shape[0] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        cv.imshow("Anteprima Webcam - [q] esci | [SPACE] scatta", preview_display)

        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Modalita' webcam annullata.")
            cap.release()
            cv.destroyAllWindows()
            return None, None

        elif key == 32 or key == 13:  # SPACE o ENTER
            print("Foto scattata.")
            final_image = cropped.copy()

            cap.release()
            cv.destroyAllWindows()
            cv.waitKey(100)

            timestamp = int(time.time())
            filename = f"webcam_{timestamp}.jpg"
            path = f"input/{filename}"
            cv.imwrite(path, final_image)

            preview_final = cv.resize(final_image, (400, 570))
            cv.imshow("Foto scattata - Premi un tasto per continuare", preview_final)
            cv.waitKey(1000)
            cv.destroyAllWindows()

            print(f"[OK] Immagine salvata in: {path}")
            return final_image, path

    cap.release()
    cv.destroyAllWindows()
    return None, None


# Removed show_a4_guide

def acquire_image():
    """Gestisce il menu di selezione della sorgente immagine (file o webcam).

    Returns:
        Tupla (image, path)
    """
    print("\n" + "=" * 50)
    print("SISTEMA DI RICONOSCIMENTO SCARPE (AVI)")
    print("=" * 50)

    print("\nMODALITA' INPUT DISPONIBILI:")
    print("   1 - Da file immagine (.jpg, .png)")
    print("   2 - Da webcam (scatta foto)")

    while True:
        scelta = input("\nScegli modalita' (1/2): ").strip()

        if scelta == "1":
            print("\nMODALITA' FILE SELEZIONATA")
            shoes_path = input("Inserisci nome file (senza estensione): ").strip()

            extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            img = None
            path = None

            for ext in extensions:
                test_path = f"input/{shoes_path}{ext}"
                if os.path.exists(test_path):
                    img = cv.imread(test_path)
                    if img is not None:
                        path = test_path
                        print(f"[OK] File trovato: {path}")
                        break

            if img is None:
                print(f"[ERRORE] Nessun file trovato con nome '{shoes_path}' nella cartella 'input/'")
                continue

            return img, path

        elif scelta == "2":
            print("\nMODALITA' WEBCAM SELEZIONATA")

            img, path = capture_from_webcam()

            if img is None:
                print("[ERRORE] Cattura annullata, riprova.")
                continue

            return img, path

        else:
            print("[ERRORE] Scelta non valida. Inserisci 1 o 2.")
            continue


def process_image(img, path, is_ui=False, callback=None):
    """Esegue l'intera pipeline di elaborazione su un'immagine acquisita.

    Pipeline:
        1. Normalizzazione dimensioni (Simulazione Field of View telecamera)
        2. Analisi texture LBP della suola
        3. Analisi dei difetti tramite SVM
        4. Rimozione Sfondo tramite rembg
        5. Calcolo dimensioni industriali virtuali dal Bounding Box
        6. Estrazione dettagli per matching
        7. Ricerca nel database e visualizzazione

    Args:
        img: Immagine OpenCV in input.
        path: Path del file immagine sull'hard disk.
        is_ui: Booleano, se True disabilita cv.imshow che causano crash nei thread.
        callback: Opzionale, funzione da chiamare per aggiornare un visualizzatore esterno (UI).
    """
    print(f"\n{'=' * 50}")
    print("ELABORAZIONE IMMAGINE IN CORSO...")
    print("=" * 50)

    # Setup cartella debug1
    DEBUG_DIR = "debug1"
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    # Pulizia selettiva: rimuoviamo i file delle esecuzioni precedenti
    # MA preserviamo i grafici della tesi (prefix 'thesis_') generati dal trainer
    for f in os.listdir(DEBUG_DIR):
        if not f.startswith("thesis_") and f != ".gitkeep":
            file_path = os.path.join(DEBUG_DIR, f)
            try:
                if os.path.isfile(file_path): os.remove(file_path)
            except: pass

    # 1. Normalizza dimensioni (Mantiene proporzioni per simulare distanza fissa Z telecamera)
    img = Homography.ensure_standard_size(img, (TARGET_WIDTH, TARGET_HEIGHT))
    cv.imwrite('debug1/01_originale.jpg', img) # Salvataggio per tesi (Cap. 2)
    cv.imwrite('debug1/03_normalized.jpg', img) # Compatibilità retroattiva
    if callback: callback('debug1/01_originale.jpg')
    print(f"[OK] Immagine normalizzata a {TARGET_WIDTH}x{TARGET_HEIGHT}.")

    height, width = img.shape[:2]

    # 3b. Analisi texture LBP: converto in grigio (un solo canale) e calcolo il descrittore
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_lbp = Utils.lbp_vectorized(img_gray)

    cv.imwrite('debug1/03b_lbp.jpg', img_lbp)

    # plt.imshow(img[:, :, ::-1])  # BGR -> RGB per matplotlib
    # plt.title("Immagine normalizzata")
    # plt.show()

    # plt.imshow(img_lbp, cmap="gray")
    cv.imwrite('debug1/03b_lbp.jpg', img_lbp)

    # 3c. Analisi dei difetti tramite SVM
    if os.path.exists(MODEL_PATH):
        try:
            print("\nANALISI DIFETTI (SVM)...")
            model = joblib.load(MODEL_PATH)
            
            # Rimozione Sfondo tramite rembg per analisi precisa
            print("[INFO] Rimozione sfondo AI (rembg) in corso...")
            img_no_bg = remove(img)
            cv.imwrite('debug1/02_rimozione_sfondo.png', img_no_bg) # Salvataggio per tesi (Cap. 2)
            cv.imwrite('debug1/03_no_bg.png', img_no_bg) # Compatibilità
            if callback: callback('debug1/02_rimozione_sfondo.png')
            
            # Estrazione maschera dall'alpha channel di rembg
            mask = img_no_bg[:, :, 3]
            cv.imwrite('debug1/03_maschera_alpha.jpg', mask) # Salvataggio per tesi (Cap. 2)
            cv.imwrite('debug1/03c_mask.jpg', mask) # Compatibilità
            
            # --- PASSAGGI LBP PER TESI (CAP. 2.4) ---
            # Mappa LBP mascherata (solo scarpa)
            masked_lbp = cv.bitwise_and(img_lbp, img_lbp, mask=mask)
            cv.imwrite('debug1/03e_masked_lbp.jpg', masked_lbp)
            
            # Istogramma LBP puro (usando la funzione di Utils)
            hist_lbp_only = Utils.calc_lbp_histogram(img_lbp, mask=mask)
            plt.figure(figsize=(10, 4))
            plt.plot(hist_lbp_only, color='blue')
            plt.fill_between(range(256), hist_lbp_only, color='blue', alpha=0.3)
            plt.title('Istogramma dei descrittori LBP (Firma della Texture)')
            plt.xlabel('Codice LBP (0-255)')
            plt.ylabel('Frequenza normalizzata')
            plt.xlim([0, 255])
            plt.grid(True, alpha=0.2)
            plt.savefig('debug1/03f_lbp_histogram.png', dpi=150, bbox_inches='tight')
            plt.close()
            # ----------------------------------------
            
            # --- PASSAGGI AGGIUNTIVI PER DOCUMENTAZIONE TESI (CAP. 2) ---
            print("[INFO] Generazione passaggi intermedi per la tesi...")
            # K-Means (usando la funzione di Utils)
            img_rgb_no_bg = img_no_bg[:, :, :3]
            clustered = Utils.kMeans_cluster(img_rgb_no_bg / 255.0, n_clusters=3)
            cv.imwrite('debug1/04_kmeans.jpg', clustered)
            
            # Canny Raw
            gray_clustered = cv.cvtColor(clustered, cv.COLOR_BGR2GRAY)
            edged_raw = cv.Canny(gray_clustered, 50, 150)
            cv.imwrite('debug1/05_canny_raw.jpg', edged_raw)
            
            # Morphology (contorni finali)
            dilated = cv.dilate(edged_raw, None, iterations=2)
            final_edges = cv.erode(dilated, None, iterations=1)
            cv.imwrite('debug1/06_contorni_finali.jpg', final_edges)
            # ------------------------------------------------------------
            
            # Estrazione combinata Texture(LBP) + Colore(HSV) + Geometria
            combined_features = Utils.extract_svm_features(img, mask_rembg=mask)
            
            # Normalizzazione feature (deve usare lo stesso scaler del training)
            if os.path.exists(SCALER_PATH):
                scaler = joblib.load(SCALER_PATH)
                combined_features_scaled = scaler.transform([combined_features])[0]
            else:
                combined_features_scaled = combined_features
            
            # --- SALVATAGGIO GRAFICO HISTOGRAMMA PER TESI ---
            plt.figure(figsize=(12, 5))
            plt.bar(range(len(combined_features)), combined_features, color='indigo', alpha=0.8)
            plt.title('Firma Unificata (LBP + HSV) della Scarpa Corrente')
            plt.xlabel('Dimensione Vettore (LBP: 0-255 | HSV: 256-317)')
            plt.ylabel('Valore feature normalizzato')
            plt.grid(axis='y', alpha=0.3)
            plt.savefig('debug1/03d_svm_unified_features.png', dpi=150, bbox_inches='tight')
            plt.close()
            # --------------------------------------------------
            
            # Predizione basata sulle probabilità per massima coerenza
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([combined_features_scaled])[0]
                class_probs = {int(c): p for c, p in zip(model.classes_, probs)}
                p_conforme = class_probs.get(0, 0.0)
                p_difettosa = class_probs.get(1, 0.0)
                
                if p_conforme >= p_difettosa:
                    print(">>> RISULTATO: Scarpa CONFORME (Nessun difetto rilevato)")
                else:
                    print(">>> RISULTATO: Scarpa DIFETTOSA (Possibile anomalia rilevata)")
                print(f"Probabilità - Conforme: {p_conforme:.2%}, Difettosa: {p_difettosa:.2%}")
            else:
                prediction = model.predict([combined_features_scaled])[0]
                if prediction == 0:
                    print(">>> RISULTATO: Scarpa CONFORME (Nessun difetto rilevato)")
                else:
                    print(">>> RISULTATO: Scarpa DIFETTOSA (Possibile anomalia rilevata)")
        except Exception as e:
            print(f"[AVVISO] Errore durante la predizione SVM: {e}")
    else:
        print("\n[AVVISO] Modello SVM non trovato. Salta analisi difetti.")

    # 4. Calcola Dimensioni Industriali Simulare
    print("\nCALCOLO DIMENSIONI E TAGLIA (Simulazione Industriale)...")
    length_mm, width_mm, bounding_box = Utils.calc_industrial_size(mask, pixel_per_mm=cfg.PIXEL_PER_MM)
    
    if bounding_box:
        sole_size_cm = length_mm / 10
        print(f"Dimensioni stimate (Calibrazione Virtuale): L {length_mm:.1f}mm - W {width_mm:.1f}mm")
        size = Utils.getSizeFromLength(sole_size_cm)
        print(f"Taglia stimata -> EU: {size['EU']}  US: {size['US']}  UK: {size['UK']}")
        
        # --- SANITY CHECK: Aspect Ratio (Lunghezza / Larghezza) ---
        length = max(length_mm, width_mm)
        width = min(length_mm, width_mm)
        aspect_ratio = length / width if width > 0 else 0
        
        if aspect_ratio < 1.8 or aspect_ratio > 4.2:
            print(f"[AVVISO] Rapporto d'aspetto anomalo ({aspect_ratio:.1f}). L'oggetto potrebbe NON essere una scarpa.")
        # ---------------------------------------------------------
        
        # Disegno quote per demo / tesi
        x, y, w, h = bounding_box
        blueprint_img = img.copy()
        cv.rectangle(blueprint_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv.putText(blueprint_img, f"L: {length_mm:.1f}mm", (x, max(y - 10, 20)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv.putText(blueprint_img, f"W: {width_mm:.1f}mm", (min(x + w + 10, TARGET_WIDTH - 150), y + h // 2), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv.imwrite('debug1/04_dimensions.jpg', blueprint_img)
        if callback: callback('debug1/04_dimensions.jpg')
        
        # Variante tecnica: Bounding Box sulla maschera binaria
        mask_blueprint = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        cv.rectangle(mask_blueprint, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv.imwrite('debug1/04b_mask_dimensions.jpg', mask_blueprint)
    else:
        print("[ATTENZIONE] Impossibile calcolare dimensioni: maschera vuota.")

    # 5. Estrai maschera rembg e DETTAGLI INTERNI per matching
    print("\nESTRAZIONE RIFERIMENTO DETTAGLIATO PER MATCHING...")
    
    # 5a. Scarpa pulita (colori reali)
    shoe_only = cv.bitwise_and(img, img, mask=mask)
    
    # 5b. Dettagli Interni (Lacci, Loghi, Cuciture)
    # Applichiamo Canny sulla scarpa pulita per ignorare lo sfondo rimosso
    gray_shoe = cv.cvtColor(shoe_only, cv.COLOR_BGR2GRAY)
    internal_edges = cv.Canny(gray_shoe, 50, 150)
    cv.imwrite('debug1/05a_internal_edges.jpg', internal_edges) # Salvataggio per tesi
    
    # Creiamo un'immagine di riferimento che combina Colore + Bordi Interni
    # I bordi bianchi aiutano l'algoritmo di similarità a trovare "ancore" strutturali
    reference_img = shoe_only.copy()
    reference_img[internal_edges > 0] = [255, 255, 255] # Overlay bordi bianchi
    
    cv.imwrite('debug1/05b_high_detail_reference.jpg', reference_img)

    temp_contour_path = "images/scarpa_contorni_temp.jpg"
    cv.imwrite(temp_contour_path, reference_img)
    print(f"[OK] Riferimento High-Detail salvato: {temp_contour_path}")

    # 6. Cerca match nel database
    print("\nRICERCA MATCH NEL DATABASE...")
    best_match, all_results = Match.find_best_match(temp_contour_path, "contorno")

    if best_match and best_match["combined"] >= SIMILARITY_THRESHOLD:
        print(f"\n{'=' * 50}")
        print("MATCH TROVATO")
        print("=" * 50)
        print(f"Modello:     {best_match['nome']}")
        print(f"Match Globale: {best_match['combined']:.2%}")
        print(f"  - Colore(HST): {best_match['histogram']:.2%} | Texture(SSIM): {best_match['similarity']:.2%}")
        print(f"  - Struttura(ORB): {best_match['orb']:.2%}  | Forma(SHP):  {best_match['shape']:.2%}")

        output_path = "output/scarpa_1.jpg"
        cv.imwrite(output_path, img)
        print(f"[OK] Immagine salvata: {output_path}")

        input_img = cv.imread(temp_contour_path)
        match_img = cv.imread(best_match["path_contorno"])

        if match_img is not None:
            # --- GENERAZIONE GRAFICI TESI (CAP. 2.6) ---
            print("[INFO] Generazione analisi strutturale per la tesi...")
            
            # 1. Mappa SSIM Diff
            ssim_diff = Compare.get_ssim_diff(temp_contour_path, best_match["path_contorno"])
            if ssim_diff is not None:
                cv.imwrite('debug1/thesis_04_ssim_map.jpg', ssim_diff)
            
            # 2. Confronto Istogrammi HSV
            img1 = cv.imread(temp_contour_path)
            img2 = cv.imread(best_match["path_contorno"])
            if img1 is not None and img2 is not None:
                hsv1 = cv.cvtColor(cv.resize(img1, (700, 1000)), cv.COLOR_BGR2HSV)
                hsv2 = cv.cvtColor(cv.resize(img2, (700, 1000)), cv.COLOR_BGR2HSV)
                # Istogrammi 1D su H per visualizzazione semplice
                hist1 = cv.calcHist([hsv1], [0], None, [180], [0, 180])
                hist2 = cv.calcHist([hsv2], [0], None, [180], [0, 180])
                plt.figure(figsize=(10, 4))
                plt.plot(hist1, label='Input', color='blue', alpha=0.8)
                plt.plot(hist2, label='Database', color='red', linestyle='--', alpha=0.6)
                plt.fill_between(range(180), hist1.flatten(), color='blue', alpha=0.1)
                plt.fill_between(range(180), hist2.flatten(), color='red', alpha=0.1)
                plt.title(f'Confronto Cromatico: Input vs {best_match["nome"]}')
                plt.legend()
                plt.savefig('debug1/thesis_05_hsv_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()
            # -------------------------------------------

            fixed_size = (TARGET_WIDTH, TARGET_HEIGHT)
            extra_padding = (50, 50)
            input_resized = Utils.resize_keep_aspect(input_img, fixed_size, extra_padding)
            match_resized = Utils.resize_keep_aspect(match_img, fixed_size, extra_padding)
            comparison = np.hstack([input_resized, match_resized])

            cv.imwrite('debug1/06_comparison.jpg', comparison)
            if callback: callback('debug1/06_comparison.jpg')
            
            if not is_ui:
                # Non bloccare il programma se siamo in modalità UI
                print("[INFO] Visualizzazione confronto...")
                cv.imshow("CONFRONTO: Input vs Database", comparison)
                cv.waitKey(2000) # Aspetta 2 secondi anziché all'infinito
                # cv.destroyAllWindows() # Evitiamo di chiudere subito per permettere all'utente di vedere
            else:
                print(f"[INFO] Match trovato: {best_match['nome']}. Confronto salvato in debug1/06_comparison.jpg")

    else:
        print(f"\n{'=' * 50}")
        print("NUOVA SCARPA RILEVATA")
        print("=" * 50)
        print(f"Nessun match trovato (soglia: {SIMILARITY_THRESHOLD:.0%})")

        next_id = Config.get_next_id()
        output_path = f"output/scarpa_{next_id}.jpg"
        cv.imwrite(output_path, img)
        print(f"[OK] Immagine salvata: {output_path}")

        permanent_contour_path = f"images/scarpa_contorni_{next_id}.jpg"
        cv.imwrite(permanent_contour_path, reference_img)

        shoe_name = input("\nInserisci nome modello scarpa: ").strip() or "Scarpa Sconosciuta"
        
        print("\nQuesta scarpa e' conforme o difettosa?")
        print("   0 - Conforme (Buona)")
        print("   1 - Difettosa (Scarto)")
        label_input = input("Scelta (0/1, default 0): ").strip()
        label = 1 if label_input == "1" else 0
        
        new_id = Config.save_to_database(shoe_name, path, permanent_contour_path, label=label)
        print(f"[OK] Nuova scarpa salvata con ID: {new_id} (Etichetta: {label})")

    # Rimuovi file temporaneo
    if os.path.exists(temp_contour_path):
        os.remove(temp_contour_path)

    print(f"\n{'=' * 50}")
    print("PROCESSO COMPLETATO CON SUCCESSO")
    print("=" * 50)


def main():
    img, path = acquire_image()

    if img is None:
        print("[ERRORE] Impossibile procedere: nessuna immagine valida.")
        sys.exit(1)

    process_image(img, path, is_ui=False)


if __name__ == "__main__":
    main()
