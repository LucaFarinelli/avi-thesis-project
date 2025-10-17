import cv2 as cv
import os
import numpy as np
from ImagePreprocessor import ImagePreprocessor
from Compare import AdvancedComparer


def test_preprocessing():
    """Test del preprocessing su immagini esistenti"""

    print("🧪 Testing preprocessing...")

    preprocessor = ImagePreprocessor(target_size=(400, 300))

    # Test su immagini esistenti
    test_images = ["images/scarpa_contorni_10.jpg", "images/scarpa_contorni_11.jpg"]

    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n📊 Testing: {img_path}")

            img = cv.imread(img_path)
            if img is None:
                print(f"❌ Impossibile caricare {img_path}")
                continue

            # Test normalizzazione
            normalized = preprocessor.normalize_shoe_image(img)
            if normalized is not None:
                print(f"✅ Normalizzazione completata - Dimensioni: {normalized.shape}")

                # Salva risultato per ispezione
                output_path = f"test_normalized_{os.path.basename(img_path)}"
                cv.imwrite(output_path, normalized)
                print(f"💾 Salvato come: {output_path}")
            else:
                print(f"❌ Errore normalizzazione")

            # Test estrazione contorni
            contours = preprocessor.enhance_contours_image(img)
            if contours is not None:
                print(f"✅ Estrazione contorni completata")

                output_contours_path = f"test_contours_{os.path.basename(img_path)}"
                cv.imwrite(output_contours_path, contours)
                print(f"💾 Contorni salvati come: {output_contours_path}")
            else:
                print(f"❌ Errore estrazione contorni")


def test_comparison():
    """Test del sistema di confronto avanzato"""

    print("\n🧪 Testing advanced comparison...")

    comparer = AdvancedComparer()

    # Test con due immagini conosciute
    img1_path = "test_normalized_scarpa_contorni_10.jpg"
    img2_path = "test_normalized_scarpa_contorni_11.jpg"

    if os.path.exists(img1_path) and os.path.exists(img2_path):
        print(f"🔄 Confrontando {img1_path} vs {img2_path}")

        # Test confronto strutturale
        structural_results = comparer.compare_images_advanced(img1_path, img2_path)
        print(f"📊 Structural results: {structural_results}")

        # Test feature matching
        orb_score = comparer.compare_features_improved(img1_path, img2_path, "orb")
        print(f"🔑 ORB score: {orb_score:.4f}")

        try:
            sift_score = comparer.compare_features_improved(
                img1_path, img2_path, "sift"
            )
            print(f"🎯 SIFT score: {sift_score:.4f}")
        except Exception as e:
            print(f"⚠️ SIFT error: {e}")
            sift_score = 0.0

        # Score combinato
        combined = (
            0.20 * structural_results.get("similarity", 0)
            + 0.15 * structural_results.get("histogram", 0)
            + 0.15 * structural_results.get("template", 0)
            + 0.30 * orb_score
            + 0.20 * sift_score
        )

        print(f"⭐ Combined score: {combined:.4f}")
    else:
        print("❌ File di test non trovati. Esegui prima test_preprocessing()")


def test_orientation_detection():
    """Test specifico per il rilevamento dell'orientamento"""

    print("\n🧪 Testing orientation detection...")

    preprocessor = ImagePreprocessor(target_size=(400, 300))

    test_images = [
        "images/scarpa_contorni_10.jpg",  # Orizzontale
        "images/scarpa_contorni_11.jpg",  # Verticale
    ]

    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n🔍 Analizzando orientamento: {img_path}")

            img = cv.imread(img_path)
            if img is None:
                continue

            # Analizza orientamento originale
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            processed = preprocessor._preprocess_for_contours(gray)
            contour = preprocessor._find_main_contour(processed)

            if contour is not None:
                rect = cv.minAreaRect(contour)
                width, height = rect[1]
                angle = rect[2]

                print(f"  📏 Dimensioni bounding box: {width:.1f} x {height:.1f}")
                print(f"  🔄 Angolo: {angle:.1f}°")
                print(
                    f"  📐 Orientamento: {'Orizzontale' if width > height else 'Verticale'}"
                )

                # Test normalizzazione
                normalized = preprocessor.normalize_shoe_image(img)
                if normalized is not None:
                    print(f"  ✅ Normalizzato a: {normalized.shape}")

                    # Salva per controllo visivo
                    output_path = f"test_orientation_{os.path.basename(img_path)}"
                    cv.imwrite(output_path, normalized)
                    print(f"  💾 Salvato: {output_path}")
            else:
                print(f"  ❌ Contorno non trovato")


def run_all_tests():
    """Esegue tutti i test"""

    print("🚀 Avvio test completi del sistema migliorato\n")

    test_preprocessing()
    test_comparison()
    test_orientation_detection()

    print("\n✅ Test completati! Controlla i file generati per verificare i risultati.")
    print("📁 File generati:")

    generated_files = [
        f for f in os.listdir(".") if f.startswith("test_") and f.endswith(".jpg")
    ]

    for file in generated_files:
        print(f"  - {file}")


if __name__ == "__main__":
    run_all_tests()
