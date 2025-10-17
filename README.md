# DetectShoe

# Scarpe — Riconoscimento suola per robot

Breve: progetto che prende in input un'immagine della suola di una scarpa (da telecamera), estrae la forma/contorno e
la confronta con un database per identificare la scarpa e la taglia.
Pensato per integrazione con un macchinario robotico che preleva e riconosce scarpe.

## Funzionalità principali
- Normalizzazione immagine: rotazione/centratura della suola per allineamento verticale.
- Ridimensionamento senza stretch e padding per confronto visivo coerente.
- Estrazione dei contorni e confronto con database di immagini.
- Confronto basato su SSIM (similarità) e confronto istogrammi; generazione di punteggio combinato.
- Salvataggio di nuove entry nel DB se non viene trovato match significativo.
- Debug: salvataggio immagini intermedie in `debug/` e temporanee in `temp/`.

## Requisiti
- Python 3.8+
- Pacchetti Python:
  - opencv-python (cv2)
  - mysql-connector-python
  - scikit-image
  - numpy

Installazione rapida:
python3 -m pip install opencv-python mysql-connector-python scikit-image numpy

## Struttura progetto (principali file)
- Main.py — entrypoint: normalizza immagine, estrae contorni, cerca match e mostra confronto.
- Compare.py — funzioni di confronto (SSIM, istogramma, debug).
- Match.py — logica di interrogazione DB e scoring tra immagini.
- Config.py — funzioni di utilità DB (salvataggio, next id, pulizia).
- images/ — immagini originali e contorni salvati.
- temp/, debug/ — cartelle usate per output temporanei / debug.


## Shema DB
CREATE TABLE dataset (
  idScarpa INT AUTO_INCREMENT PRIMARY KEY,
  nome_scarpa VARCHAR(255),
  path_originale VARCHAR(1024),
  path_contorno VARCHAR(1024),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

N.B Aggiornare file Config.py prima dell'uso in produzione

## Esempio di Utilizzo
1. Inserire immagine di input in images/ con nome <nome>.jpg (o puntare la telecamera)
2. Eseguire da riga di comando dentro la cartella contenente il file
      python3 Main.py
3. Inserire il nome della scarpa (immagine dentro images/) richiesta dal programma
4. Se non viene trovato match sopra la soglia (0.85) verrà chiesto il nome della scarpa e il record verrà aggiunto al db

## Parametri rilevanti e soglie
- Soglia similarità combinata: SIMILARITY_THRESHOLD = 0.85 (regolabile in Main.py)
- Normalizzazione: angolo di rotazione ignorato se < +- n°
- Dimensioni standard usate per confronto visivo

## Debug e Diagnostica
- Vengono salvati file in debug/(img originali, resize, final) dentro Compare.py per ispezione
- Messaggi di debug stampati su stdout (punteggi)
- Se confronto fallisce, verifica l'esistenza dei path memorizzati nel DB con Config.clean_invalid_path()
- Commentata anche funzione per pulire completamente record della tabella Dataset del DB 