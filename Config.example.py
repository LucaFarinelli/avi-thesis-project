import mysql.connector
import os

# Copiare questo file in Config.py e impostare le credenziali locali.
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "YOUR_DB_USER",
    "password": "YOUR_DB_PASSWORD",
    "database": "SCARPE",
}


def _get_connection():
    """Restituisce una nuova connessione al database."""
    return mysql.connector.connect(**DB_CONFIG)


def save_to_database(object_name, path_value, contours_data=None, label=0, object_type=None):
    """Inserisce un nuovo oggetto nel database e restituisce il suo ID."""
    conn = _get_connection()
    cursor = conn.cursor()

    object_type = object_type or "unknown"
    query = (
        "INSERT INTO dataset (object_type, object_name, path_originale, path_contorno, etichetta) "
        "VALUES (%s, %s, %s, %s, %s)"
    )
    cursor.execute(query, (object_type, object_name, path_value, contours_data, label))

    conn.commit()
    object_id = cursor.lastrowid
    conn.close()

    print(f"Oggetto salvato con ID: {object_id} (Etichetta: {label})")
    return object_id


def get_all_labeled_data(object_type=None):
    """Recupera i record con path e etichetta per il training."""
    conn = _get_connection()
    cursor = conn.cursor()

    if object_type:
        cursor.execute(
            "SELECT path_originale, etichetta FROM dataset WHERE object_type = %s",
            (object_type,),
        )
    else:
        cursor.execute("SELECT path_originale, etichetta FROM dataset")
    results = cursor.fetchall()

    conn.close()
    return results


def get_next_id():
    """Ottiene il prossimo ID disponibile per la nomenclatura dei file."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT MAX(id) FROM dataset")
    result = cursor.fetchone()
    next_id = (result[0] or 0) + 1

    conn.close()
    return next_id


def get_available_object_types():
    """Restituisce i tipi oggetto presenti nel database."""
    conn = _get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT DISTINCT object_type FROM dataset ORDER BY object_type")
        results = [row[0] for row in cursor.fetchall() if row[0]]
    except mysql.connector.errors.ProgrammingError:
        results = ["shoe", "bottle"]
    finally:
        conn.close()

    return results


def get_last_inserted_object():
    """Restituisce le informazioni sull'ultimo oggetto inserito."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, object_type, object_name, path_originale, path_contorno, created_at
        FROM dataset
        ORDER BY id DESC
        LIMIT 1
    """
    )

    result = cursor.fetchone()
    conn.close()

    if result:
        return result
    else:
        print("Nessun oggetto trovato nel database.")
        return None


def get_last_inserted_shoe():
    """Alias compatibilita': usa get_last_inserted_object."""
    return get_last_inserted_object()


def clean_invalid_paths():
    """Rimuove dal database i record con path file non trovati sul disco."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, object_type, object_name, path_originale, path_contorno FROM dataset"
    )
    rows = cursor.fetchall()

    invalid_ids = []
    for row in rows:
        object_id, object_type, object_name, path_orig, path_cont = row
        if not os.path.exists(path_orig) or not os.path.exists(path_cont):
            invalid_ids.append(object_id)
            print(f"Record non valido ID {object_id}: {object_type} - {object_name}")
            print(f"  Path originale: {path_orig} (esiste: {os.path.exists(path_orig)})")
            print(f"  Path contorno:  {path_cont} (esiste: {os.path.exists(path_cont)})")

    if invalid_ids:
        print(f"\nTrovati {len(invalid_ids)} record con path non validi.")
        risposta = input("Vuoi eliminarli? (y/n): ")

        if risposta.lower() == "y":
            for object_id in invalid_ids:
                cursor.execute("DELETE FROM dataset WHERE id = %s", (object_id,))
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
