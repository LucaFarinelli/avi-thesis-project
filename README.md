# DetectShoe — Riconoscimento Suola per Robot (AVI)

Sistema di **Automated Visual Inspection (AVI)** per il riconoscimento di suole di scarpe tramite AI e Computer Vision.
Il sistema utilizza la rimozione automatica dello sfondo via AI (`rembg`) per estrarre dettagli ad alta precisione, permettendo l'identificazione del modello, il calcolo della taglia e il rilevamento di difetti tramite SVM.

---

## 🚀 Guida Rapida: Operazioni Preliminari

Prima di avviare l'interfaccia principale (`python ui.py`), è fondamentale configurare l'ambiente e popolare il database per garantire la massima precisione.

### 1. Configurazione Database
Assicurati che MySQL sia attivo e che lo schema sia creato (vedi sezione [Schema Database](#schema-database)).
Modifica il file `Config.py` con le tue credenziali:
```python
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "tuo_utente",
    "password": "tua_password",
    "database": "SCARPE",
}
```

### 2. Importazione Iniziale dei Modelli
Metti le foto delle scarpe "perfette" (senza difetti) nella cartella `input/`.
Esegui l'importatore automatico:
```bash
python bulk_importer.py
```
*Questo caricherà i modelli nel database e creerà le anteprime pulite in `dbimage/`.*

### 3. Generazione Dettagli AI (High-Detail)
Per massimizzare la precisione del matching (loghi, lacci, texture), rigenera i riferimenti usando l'IA:
```bash
python refresh_db_contours.py
```
*Questo script isola la scarpa e ne estrae i dettagli strutturali interni su fondo nero.*

### 4. Addestramento Rilevatore Difetti (SVM)
Addestra il modello di intelligenza artificiale per riconoscere la differenza tra scarpe conformi e difettose:
```bash
python svm_trainer.py
```

### 5. Avvio del Sistema
Ora sei pronto per avviare l'interfaccia utente:
```bash
python ui.py
```

---

## ✨ Funzionalità Avanzate

- **AI Background Removal**: Integrazione intelligente con `rembg` per isolare perfettamente la scarpa eliminando ombre, background complessi e rumore ambientale.
- **High-Detail Structural Matching (ORB + Hu Moments)**: Il sistema non confronta solo la sagoma esterna o i colori simili, ma cerca geometricamente i riferimenti interni (loghi Nike, lacci, cuciture) e la forma vettoriale della silhouette (Top-view vs Sole-view), estromettendo i falsi positivi.
- **Advanced SVM Defect Detection (Texture + HSV)**: Analisi della conformità tramite Support Vector Machine operante su un vettore matematico unificato di 318 dimensioni che fonde i descrittori LBP (texture strutturale) agli istogrammi HSV (variazioni e macchie cromatiche).
- **Simulazione Dimensionale Industriale (Fixed Z-Axis)**: Implementazione di un calcolatore dimensionale simulato calcolato sul Bounding Box AI e un fattore di calibrazione virtuale configurabile (`pixel_per_mm`), replicando una telecamera di fabbrica a distanza focale fissa per determinare le grandezze precise al millimetro senza riferimenti cartacei analogici.

---

## 🛠️ Requisiti e Installazione

Il progetto è ottimizzato per essere eseguito in un **Virtual Environment**.

```bash
# Attiva il tuo virtual environment (se già creato)
source venv/bin/activate

# Installa le dipendenze necessarie
pip install opencv-python mysql-connector-python scikit-image scikit-learn numpy rembg[cpu] joblib pillow
```

---

## 📂 Struttura del Progetto (File da includere su GitHub)

### File Principali
- `ui.py`: Interfaccia grafica principale (Tkinter).
- `Main.py`: Motore di elaborazione e pipeline di analisi.
- `bulk_importer.py`: Utility per l'inserimento massivo di nuovi modelli.
- `refresh_db_contours.py`: Rigenerazione dei riferimenti database con AI High-Detail.
- `svm_trainer.py`: Script per l'addestramento del modello SVM.

### Moduli di Supporto
- `Homography.py`: Gestione prospettiva e riferimento A4.
- `Utils.py`: Calcolo taglie e algoritmi di Computer Vision.
- `Compare.py`: Logica di confronto SSIM e istogrammi (ora sensibile ai dettagli AI).
- `Config.py`: Gestione connessione database e query.
- `Match.py`: Orchestrazione dei confronti nel database.

### Risorse e Modelli
- `shoe_svm_model.pkl`: Il modello SVM addestrato (caricare dopo il primo training).
- `requirements.txt`: Lista delle dipendenze Python.

### Cartelle (Assicurati che esistano)
- `input/`: Cartella sorgente per nuove immagini.
- `output/`: Risultati finali delle analisi.
- `images/`: Database dei riferimenti High-Detail salvati.
- `dbimage/`: Archivio immagini con sfondo rimosso per revisione.
- `debug1/`: Immagini intermedie per diagnostica.

---

## 📊 Schema Database

```sql
CREATE TABLE dataset (
  idScarpa       INT AUTO_INCREMENT PRIMARY KEY,
  nome_scarpa    VARCHAR(255),
  path_originale VARCHAR(1024),
  path_contorno  VARCHAR(1024), -- Contiene il riferimento High-Detail AI
  etichetta      INT DEFAULT 0,  -- 0 per Conforme, 1 per Difettosa
  created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 📝 Note per lo Sviluppatore
Il sistema è progettato per operare con un foglio **A4** come standard di calibrazione. Per risultati ottimali, assicurarsi che la scarpa sia ben visibile e che il foglio A4 sia posizionato su una superficie piana con un buon contrasto cromatico.
