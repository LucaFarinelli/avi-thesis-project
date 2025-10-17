import cv2 as cv
import numpy as np
import math


class ImagePreprocessor:
    """Classe per il preprocessing avanzato delle immagini di scarpe"""

    def __init__(self, target_size=(400, 300)):
        self.target_size = target_size
        self.target_width = target_size[0]
        self.target_height = target_size[1]

    def normalize_shoe_image(self, img):
        """
        Normalizzazione avanzata dell'immagine scarpa:
        1. Orientamento verticale forzato (scarpa in piedi)
        2. Miglior rilevamento contorni
        3. Centratura e padding intelligente
        4. Dimensioni standard
        """
        if img is None:
            return None

        img_copy = img.copy()

        # 1. PREPROCESSING per miglior rilevamento contorni
        processed = self._preprocess_for_contours(img_copy)

        # 2. TROVA contorno principale
        main_contour = self._find_main_contour(processed)

        if main_contour is None:
            print("⚠️ Contorno non trovato, usando immagine originale")
            return cv.resize(img_copy, self.target_size)

        # 3. CALCOLA orientamento e forza verticale
        oriented_img = self._force_vertical_orientation(img_copy, main_contour)

        # 4. RITAGLIA con padding intelligente
        cropped = self._smart_crop_with_padding(oriented_img, main_contour, padding_ratio=0.15)

        # 5. RIDIMENSIONA mantenendo aspect ratio
        final = self._resize_with_aspect_ratio(cropped)

        return final

    def _preprocess_for_contours(self, img):
        """Preprocessing ottimizzato per rilevamento contorni"""

        # Converti in scala di grigi
        if len(img.shape) == 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Applica equalizzazione istogramma per miglior contrasto
        equalized = cv.equalizeHist(gray)

        # Filtro bilaterale per ridurre rumore mantenendo bordi
        bilateral = cv.bilateralFilter(equalized, 9, 75, 75)

        # Applica morfologia per chiudere gaps nei contorni
        kernel = np.ones((3, 3), np.uint8)
        morphology = cv.morphologyEx(bilateral, cv.MORPH_CLOSE, kernel)

        return morphology

    def _find_main_contour(self, processed_img):
        """Trova il contorno principale (scarpa) con parametri ottimizzati"""

        # Parametri Canny ottimizzati per scarpe
        # Test multipli soglie per trovare la migliore
        edge_params = [
            (30, 80),   # Soglie basse per dettagli fini
            (50, 150),  # Soglie medie (originali)
            (80, 200),  # Soglie alte per contorni netti
        ]

        best_contour = None
        max_area = 0

        for low, high in edge_params:
            # Applica Canny
            edges = cv.Canny(processed_img, low, high)

            # Applica chiusura morfologica per connettere i bordi
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            edges_closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

            # Trova contorni
            contours, _ = cv.findContours(edges_closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            # Trova il contorno più grande che sembra una scarpa
            for contour in contours:
                area = cv.contourArea(contour)

                # Filtra contorni troppo piccoli o troppo grandi
                img_area = processed_img.shape[0] * processed_img.shape[1]
                if area < img_area * 0.1 or area > img_area * 0.9:
                    continue

                # Verifica che il contorno sia ragionevolmente convesso (forma scarpa)
                hull = cv.convexHull(contour)
                hull_area = cv.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    if solidity > 0.3:  # La scarpa deve essere abbastanza "solida"
                        if area > max_area:
                            max_area = area
                            best_contour = contour

        return best_contour

    def _force_vertical_orientation(self, img, contour):
        """Forza orientamento verticale della scarpa"""

        # Calcola il rettangolo di area minima
        rect = cv.minAreaRect(contour)
        width, height = rect[1]
        angle = rect[2]

        # Determina se la scarpa è orizzontale o verticale
        is_horizontal = width > height

        # Calcola l'angolo di rotazione necessario per renderla verticale
        if is_horizontal:
            # Se è orizzontale, ruota di 90 gradi
            if angle < -45:
                rotation_angle = angle + 90
            else:
                rotation_angle = angle - 90
        else:
            # Se è già più o meno verticale, piccoli aggiustamenti
            if angle < -45:
                rotation_angle = angle + 90
            else:
                rotation_angle = angle

        # Applica la rotazione
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)

        # Calcola la matrice di rotazione
        M = cv.getRotationMatrix2D(center, rotation_angle, 1.0)

        # Calcola le nuove dimensioni per non tagliare l'immagine
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Aggiusta la matrice di traslazione
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Ruota l'immagine
        rotated = cv.warpAffine(img, M, (new_w, new_h), flags=cv.INTER_LINEAR, borderValue=(255, 255, 255))

        return rotated

    def _smart_crop_with_padding(self, img, original_contour, padding_ratio=0.15):
        """Ritaglio intelligente con padding per evitare tagli"""

        # Trova il nuovo contorno nell'immagine ruotata
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        processed = self._preprocess_for_contours(gray)
        new_contour = self._find_main_contour(processed)

        if new_contour is None:
            # Fallback: usa l'immagine completa
            return img

        # Calcola bounding box
        x, y, w, h = cv.boundingRect(new_contour)

        # Calcola padding basato sulle dimensioni dell'oggetto
        padding_x = int(w * padding_ratio)
        padding_y = int(h * padding_ratio)

        # Applica padding assicurandosi di non uscire dai bordi
        img_h, img_w = img.shape[:2]
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(img_w, x + w + padding_x)
        y2 = min(img_h, y + h + padding_y)

        cropped = img[y1:y2, x1:x2]

        return cropped

    def _resize_with_aspect_ratio(self, img):
        """Ridimensiona mantenendo l'aspect ratio e aggiungendo padding se necessario"""

        h, w = img.shape[:2]
        target_w, target_h = self.target_size

        # Calcola il rapporto di scala
        scale = min(target_w / w, target_h / h)

        # Calcola nuove dimensioni
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Ridimensiona
        resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LANCZOS4)

        # Crea immagine finale con padding bianco
        final = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

        # Calcola offset per centrare l'immagine
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2

        # Inserisci l'immagine ridimensionata al centro
        if len(resized.shape) == 3:
            final[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
        else:
            # Se è in scala di grigi, converte a 3 canali
            resized_3ch = cv.cvtColor(resized, cv.COLOR_GRAY2BGR)
            final[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_3ch

        return final

    def enhance_contours_image(self, img):
        """Crea un'immagine ottimizzata per il rilevamento dei contorni"""

        if img is None:
            return None

        # Usa l'immagine normalizzata come base
        normalized = self.normalize_shoe_image(img)

        if normalized is None:
            return None

        # Converti in scala di grigi
        gray = cv.cvtColor(normalized, cv.COLOR_BGR2GRAY)

        # Preprocessing avanzato
        processed = self._preprocess_for_contours(gray)

        # Applica Canny con parametri ottimizzati
        edges = cv.Canny(processed, 30, 80)

        # Dilata leggermente i contorni per renderli più visibili
        kernel = np.ones((2, 2), np.uint8)
        edges_dilated = cv.dilate(edges, kernel, iterations=1)

        # Converti di nuovo a BGR per consistenza
        contours_img = cv.cvtColor(edges_dilated, cv.COLOR_GRAY2BGR)

        return contours_img
