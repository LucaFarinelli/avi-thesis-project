import mysql.connector
import os

# Credenziali di connessione al database
# N.B. Aggiornare questi valori prima di usare in produzione
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "luca",
    "password": "password",
    "database": "SCARPE",
}


def _get_connection():
    """Restituisce una nuova connessione al database."""
    return mysql.connector.connect(**DB_CONFIG)


def save_to_database(name, path_value, contours_data=None, label=0):
    """Inserisce una nuova scarpa nel database e restituisce il suo ID."""
    conn = _get_connection()
    cursor = conn.cursor()

    query = "INSERT INTO dataset (nome_scarpa, path_originale, path_contorno, etichetta) VALUES (%s, %s, %s, %s)"
    cursor.execute(query, (name, path_value, contours_data, label))

    conn.commit()
    shoe_id = cursor.lastrowid
    conn.close()

    print(f"Scarpa salvata con ID: {shoe_id} (Etichetta: {label})")
    return shoe_id

def get_all_labeled_data():
    """Recupera tutti i record con il relativo path e etichetta per il training."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT path_originale, etichetta FROM dataset")
    results = cursor.fetchall()
    
    conn.close()
    return results


def get_next_id():
    """Ottiene il prossimo ID disponibile per la nomenclatura dei file."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT MAX(idScarpa) FROM dataset")
    result = cursor.fetchone()
    next_id = (result[0] or 0) + 1

    conn.close()
    return next_id


def get_last_inserted_shoe():
    """Restituisce le informazioni sull'ultima scarpa inserita."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT idScarpa, nome_scarpa, path_originale, path_contorno, created_at
        FROM dataset
        ORDER BY idScarpa DESC
        LIMIT 1
    """)

    result = cursor.fetchone()
    conn.close()

    if result:
        return result
    else:
        print("Nessuna scarpa trovata nel database.")
        return None


def clean_invalid_paths():
    """Rimuove dal database i record con path file non trovati sul disco."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT idScarpa, nome_scarpa, path_originale, path_contorno FROM dataset"
    )
    rows = cursor.fetchall()

    invalid_ids = []
    for row in rows:
        shoe_id, nome, path_orig, path_cont = row
        if not os.path.exists(path_orig) or not os.path.exists(path_cont):
            invalid_ids.append(shoe_id)
            print(f"Record non valido ID {shoe_id}: {nome}")
            print(f"  Path originale: {path_orig} (esiste: {os.path.exists(path_orig)})")
            print(f"  Path contorno:  {path_cont} (esiste: {os.path.exists(path_cont)})")

    if invalid_ids:
        print(f"\nTrovati {len(invalid_ids)} record con path non validi.")
        risposta = input("Vuoi eliminarli? (y/n): ")

        if risposta.lower() == "y":
            for shoe_id in invalid_ids:
                cursor.execute("DELETE FROM dataset WHERE idScarpa = %s", (shoe_id,))
            conn.commit()
            print(f"Eliminati {len(invalid_ids)} record non validi.")
        else:
            print("Eliminazione annullata.")
    else:
        print("Nessun record non valido trovato.")

    conn.close()


def remove_all_from_db():
    """Elimina tutti i record dalla tabella dataset. Usare con cautela."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM dataset")
    conn.commit()
    conn.close()
    print("Tutti i record sono stati eliminati dal database.")
