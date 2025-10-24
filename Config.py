import mysql.connector
import os


def save_to_database(name, path_value, contours_data=None):
    # ----- STABILIRE CONNESSIONE CON BD
    conn = mysql.connector.connect(
        host="127.0.0.1", user="luca", password="password", database="SCARPE"
    )

    cursor = conn.cursor()

    query = "INSERT INTO dataset (nome_scarpa,path_originale,path_contorno) VALUES (%s, %s, %s)"
    val = (name, path_value, contours_data)
    cursor.execute(query, val)

    conn.commit()
    shoe_id = cursor.lastrowid
    conn.close()

    print(f"Scarpa salvata con ID: {shoe_id}")
    return shoe_id


def get_next_id():
    """Ottiene il prossimo ID disponibile per nomenclatura file"""
    conn = mysql.connector.connect(
        host="127.0.0.1", user="luca", password="password", database="SCARPE"
    )
    cursor = conn.cursor()

    cursor.execute("SELECT MAX(idScarpa) FROM dataset")
    result = cursor.fetchone()
    next_id = (result[0] or 0) + 1

    conn.close()
    return next_id


def get_last_inserted_shoe():
    """Mostra info sull'ultima scarpa inserita"""
    conn = mysql.connector.connect(
        host="127.0.0.1", user="luca", password="password", database="SCARPE"
    )
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
        shoe_id, nome, path_orig, path_cont, created_at = result
        # print(f"\n ULTIMA SCARPA INSERITA:")
        # print(f"   ID: {shoe_id}")
        # print(f"   Nome: {nome}")
        # print(f"   Path originale: {path_orig}")
        # print(f"   Path contorno: {path_cont}")
        # print(f"   Inserita: {created_at}")
        # print("------------------------------------------")
        return result
    else:
        print("Nessuna scarpa trovata nel database")
        return None


def clean_invalid_paths():
    """Rimuove record con path non validi"""
    conn = mysql.connector.connect(
        host="127.0.0.1", user="luca", password="password", database="SCARPE"
    )
    cursor = conn.cursor()

    # Trova record con path che non esistono o sono invalidi
    cursor.execute(
        "SELECT idScarpa, nome_scarpa, path_originale, path_contorno FROM dataset"
    )
    rows = cursor.fetchall()

    invalid_ids = []
    for row in rows:
        shoe_id, nome, path_orig, path_cont = row

        # Se il path non esiste come file, è probabilmente un dato errato
        if not os.path.exists(path_orig) or not os.path.exists(path_cont):
            invalid_ids.append(shoe_id)
            print(f"Record invalido ID {shoe_id}: {nome}")
            print(f"   Path orig: {path_orig} (esiste: {os.path.exists(path_orig)})")
            print(f"   Path cont: {path_cont} (esiste: {os.path.exists(path_cont)})")

    # Chiedi conferma prima di eliminare
    if invalid_ids:
        print(f"\n Trovati {len(invalid_ids)} record con path invalidi")
        risposta = input("Vuoi eliminarli? (y/n): ")

        if risposta.lower() == "y":
            for shoe_id in invalid_ids:
                cursor.execute("DELETE FROM dataset WHERE idScarpa = %s", (shoe_id,))
            conn.commit()
            print(f"✅ Eliminati {len(invalid_ids)} record invalidi")
        else:
            print("Eliminazione annullata")
    else:
        print("Nessun record invalido trovato")

    conn.close()


def remove_all_from_bd():
    """
    Elimina tutti i record del db
    """
    conn = mysql.connector.connect(
        host="127.0.0.1", user="luca", password="password", database="SCARPE"
    )
    cursor = conn.cursor()

    cursor.execute("DELETE FROM dataset")
