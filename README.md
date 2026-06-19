# Sistema AVI Multi-oggetto

Progetto di Automated Visual Inspection (AVI) basato su Python e OpenCV.
Il sistema include:
- classificazione conformita/difetto tramite SVM,
- matching strutturale su database immagini,
- stima dimensionale da maschera segmentata.

Supporta piu tipi di oggetto, ad esempio `shoe`, `bottle`, `can`.

## Requisiti

- Python 3.10+
- MySQL Server attivo

## Installazione

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configurazione database

1. Creare un database MySQL chiamato `SCARPE`.
2. Copiare `Config.example.py` in `Config.py`.
3. Impostare in `Config.py` le credenziali locali MySQL.
4. Creare la tabella `dataset` con lo schema seguente:

```sql
CREATE TABLE dataset (
  id             INT AUTO_INCREMENT PRIMARY KEY,
  object_type    VARCHAR(50) NOT NULL,
  object_name    VARCHAR(255),
  path_originale VARCHAR(1024),
  path_contorno  VARCHAR(1024),
  etichetta      INT DEFAULT 0,
  created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Avvio rapido (con modelli .pkl gia presenti)

```bash
python ui.py
```

## Flusso completo (ricostruzione da zero)

1. Inserire immagini in `input/`.
2. Importare dataset in DB:

```bash
python bulk_importer.py
```

3. Rigenerare riferimenti strutturali:

```bash
python refresh_db_contours.py
```

4. Addestrare modello per tipo oggetto:

```bash
python svm_trainer.py
```

5. Avviare GUI:

```bash
python ui.py
```

## File principali

- `ui.py`: interfaccia grafica.
- `Main.py`: pipeline principale.
- `svm_trainer.py`: training SVM.
- `bulk_importer.py`: import su database.
- `refresh_db_contours.py`: rigenerazione riferimenti.
- `Match.py`, `Compare.py`, `Utils.py`: matching e feature extraction.
- `Config.py`: connessione DB e query.
- `config_settings.py`: parametri globali.

## Note per il test

- Se i modelli `.pkl` sono presenti, il sistema puo essere testato subito.
- Se i modelli non sono presenti, eseguire prima `svm_trainer.py`.
- Le cartelle `input/`, `output/`, `images/`, `dbimage/`, `debug1/` devono esistere.
