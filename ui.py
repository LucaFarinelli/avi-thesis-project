import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import threading
import os
import sys

# Importa i moduli principali
import Main
import Config

class ShoeRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Riconoscimento Oggetti")
        self.root.geometry("520x540")
        self.root.resizable(False, False)
        
        # Impostazione stile moderno con tema 'clam'
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except:
            pass  # Se 'clam' non è disponibile, usa il default
        
        # Colori personalizzati
        self.bg_color = "#f5f5f7"
        self.accent_color = "#3b82f6"
        self.success_color = "#10b981"
        self.warning_color = "#f59e0b"
        self.danger_color = "#ef4444"
        self.text_color = "#1f2937"
        
        self.root.configure(bg=self.bg_color)
        
        self.is_processing = False
        
        # Configurazione stili per i widget
        self._configure_styles()
        
        # Frame principale con padding e bordo arrotondato (simulato con frame normale)
        main_frame = tk.Frame(root, bg=self.bg_color, padx=30, pady=30)
        main_frame.pack(fill="both", expand=True)
        
        # Intestazione con icona testuale e sottotitolo
        header_frame = tk.Frame(main_frame, bg=self.bg_color)
        header_frame.pack(fill="x", pady=(0, 20))
        
        title_label = tk.Label(
            header_frame,
            text="🔍 Sistema Riconoscimento Oggetti",
            font=("Segoe UI", 18, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="Seleziona il tipo di oggetto e avvia l'analisi",
            font=("Segoe UI", 10),
            bg=self.bg_color,
            fg="#6b7280"
        )
        subtitle_label.pack()
        
        # Separatore personalizzato (linea sottile)
        separator = tk.Frame(main_frame, height=2, bg="#e5e7eb")
        separator.pack(fill="x", pady=10)
        
        # Frame selezione oggetto
        selection_frame = tk.Frame(main_frame, bg=self.bg_color)
        selection_frame.pack(pady=10, fill="x")
        
        tk.Label(
            selection_frame,
            text="Tipo di oggetto:",
            font=("Segoe UI", 11),
            bg=self.bg_color,
            fg=self.text_color
        ).pack(side="left", padx=(0, 10))
        
        # Combobox con stile personalizzato
        self.object_options = ["Shoes", "Smartphone", "Object"]
        self.object_var = tk.StringVar()
        self.cb_object = ttk.Combobox(
            selection_frame,
            textvariable=self.object_var,
            values=self.object_options,
            state="readonly",
            width=22,
            font=("Segoe UI", 10)
        )
        self.cb_object.set("Shoes")
        self.cb_object.pack(side="left")
        
        # Frame per i bottoni principali (griglia 2x2?)
        buttons_frame = tk.Frame(main_frame, bg=self.bg_color)
        buttons_frame.pack(pady=25, fill="both", expand=True)
        
        # Primo bottone: Carica da file
        btn_file = ttk.Button(
            buttons_frame,
            text="📁 Carica da File",
            command=self.process_from_file,
            style="File.TButton"
        )
        btn_file.pack(pady=8, ipadx=10, ipady=8, fill="x")
        
        # Secondo bottone: Cattura da webcam
        btn_webcam = ttk.Button(
            buttons_frame,
            text="📷 Cattura da Webcam",
            command=self.process_from_webcam,
            style="Webcam.TButton"
        )
        btn_webcam.pack(pady=8, ipadx=10, ipady=8, fill="x")
        
        # Terzo bottone: Pulisci database
        btn_clean = ttk.Button(
            buttons_frame,
            text="🗑️ Pulisci Database",
            command=self.clean_database,
            style="Clean.TButton"
        )
        btn_clean.pack(pady=8, ipadx=10, ipady=8, fill="x")
        
        # Quarto bottone: Chiudi Programma (Rosso)
        btn_close = ttk.Button(
            buttons_frame,
            text="❌ Chiudi Programma",
            command=self.quit_app,
            style="Quit.TButton"
        )
        btn_close.pack(pady=8, ipadx=10, ipady=8, fill="x")
        
        # Frame per lo stato
        status_frame = tk.Frame(main_frame, bg=self.bg_color)
        status_frame.pack(fill="x", pady=(20, 0))
        
        # Etichetta di stato (tk.Label per poter cambiare colore facilmente)
        self.status_label = tk.Label(
            status_frame,
            text="✅ Pronto",
            font=("Segoe UI", 10),
            bg=self.bg_color,
            fg=self.success_color,
            anchor="w"
        )
        self.status_label.pack(side="left")
        
    def _configure_styles(self):
        """Configura gli stili personalizzati per i bottoni ttk"""
        # Stile base per tutti i bottoni
        self.style.configure(
            "Base.TButton",
            font=("Segoe UI", 11, "bold"),
            padding=(20, 8),
            borderwidth=0,
            focusthickness=0,
            focuscolor="none",
            relief="flat"
        )
        
        # Stile per bottone "Carica da File"
        self.style.configure(
            "File.TButton",
            background="#10b981",      # verde
            foreground="white"
        )
        self.style.map(
            "File.TButton",
            background=[("active", "#059669"), ("pressed", "#047857")],
            foreground=[("active", "white")]
        )
        
        # Stile per bottone "Cattura da Webcam"
        self.style.configure(
            "Webcam.TButton",
            background="#3b82f6",      # blu
            foreground="white"
        )
        self.style.map(
            "Webcam.TButton",
            background=[("active", "#2563eb"), ("pressed", "#1d4ed8")],
            foreground=[("active", "white")]
        )
        
        # Stile per bottone "Pulisci Database"
        self.style.configure(
            "Clean.TButton",
            background="#f59e0b",      # arancione
            foreground="white"
        )
        self.style.map(
            "Clean.TButton",
            background=[("active", "#d97706"), ("pressed", "#b45309")],
            foreground=[("active", "white")]
        )
        
        # Stile per bottone "Chiudi" (Rosso)
        self.style.configure(
            "Quit.TButton",
            background=self.danger_color,
            foreground="white"
        )
        self.style.map(
            "Quit.TButton",
            background=[("active", "#dc2626"), ("pressed", "#b91c1c")],
            foreground=[("active", "white")]
        )
        
        # Configurazione Combobox
        self.style.configure(
            "TCombobox",
            fieldbackground="white",
            background="white",
            foreground=self.text_color,
            arrowcolor=self.accent_color,
            padding=5
        )
        
    def get_selected_object(self):
        """Ritorna il tipo di oggetto selezionato"""
        return self.object_var.get()
    
    def update_status(self, message, color="black"):
        """Aggiorna lo stato in modo sicuro dal thread."""
        def _update():
            self.status_label.config(text=message, fg=color)
        self.root.after(0, _update)
    
    def safe_messagebox(self, title, message, type="info"):
        """Mostra un messaggio in modo sicuro dal thread."""
        def _show():
            if type == "error":
                messagebox.showerror(title, message)
            elif type == "warning":
                messagebox.showwarning(title, message)
            else:
                messagebox.showinfo(title, message)
        self.root.after(0, _show)
    
    def process_from_file(self):
        """Elabora un'immagine da file"""
        if self.is_processing:
            messagebox.showwarning("Avviso", "Elaborazione in corso...")
            return
        
        selected_object = self.get_selected_object()
        self.update_status(f"📂 Oggetto selezionato: {selected_object}", self.accent_color)
        
        file_path = filedialog.askopenfilename(
            initialdir="input/",
            filetypes=[("Immagini", "*.jpg *.jpeg *.png"), ("Tutti i file", "*.*")]
        )
        
        if not file_path:
            return
        
        thread = threading.Thread(target=self._process_image_thread, 
                                 args=(file_path, selected_object))
        thread.start()
    
    def process_from_webcam(self):
        """Elabora un'immagine da webcam"""
        if self.is_processing:
            messagebox.showwarning("Avviso", "Elaborazione in corso...")
            return
        
        selected_object = self.get_selected_object()
        self.update_status(f"📷 Oggetto selezionato: {selected_object}", self.accent_color)
        
        thread = threading.Thread(target=self._process_webcam_thread, 
                                 args=(selected_object,))
        thread.start()
    
    def _process_image_thread(self, file_path, object_type):
        """Thread per elaborare immagine da file"""
        try:
            self.is_processing = True
            self.update_status(f"📂 Caricamento immagine ({object_type})...", self.accent_color)
            
            import cv2
            img = cv2.imread(file_path)
            
            if img is None:
                self.update_status("❌ Errore: impossibile caricare l'immagine", self.danger_color)
                messagebox.showerror("Errore", "Impossibile caricare l'immagine")
                return
            
            self.update_status("⚙️ Elaborazione in corso...", self.accent_color)
            
            print(f"[INFO] Tipo oggetto: {object_type}")
            Main.process_image(img, file_path, is_ui=True)
            
            self.update_status("✅ Elaborazione completata!", self.success_color)
            self.safe_messagebox("Successo", f"Elaborazione di {object_type} completata con successo!")
            
        except Exception as e:
            self.update_status(f"❌ Errore: {str(e)}", self.danger_color)
            messagebox.showerror("Errore", f"Errore durante l'elaborazione:\n{str(e)}")
        
        finally:
            self.is_processing = False
    
    def _process_webcam_thread(self, object_type):
        """Thread per elaborare immagine da webcam"""
        try:
            self.is_processing = True
            self.update_status(f"📷 Cattura da webcam ({object_type})... (premi SPACE per scattare)", self.accent_color)
            
            img, path = Main.capture_from_webcam()
            
            if img is None:
                self.update_status("⚠️ Cattura annullata", self.warning_color)
                return
            
            self.update_status("⚙️ Elaborazione in corso...", self.accent_color)
            
            print(f"[INFO] Tipo oggetto: {object_type}")
            Main.process_image(img, path, is_ui=True)
            
            self.update_status("✅ Elaborazione completata!", self.success_color)
            self.safe_messagebox("Successo", f"Elaborazione di {object_type} completata con successo!")
            
        except Exception as e:
            self.update_status(f"❌ Errore: {str(e)}", self.danger_color)
            self.safe_messagebox("Errore", f"Errore durante l'elaborazione:\n{str(e)}", "error")
        
        finally:
            self.is_processing = False
    
    def clean_database(self):
        """Pulisci il database rimuovendo path non validi"""
        if messagebox.askyesno("Conferma", "Ripulire il database dai percorsi non validi?"):
            Config.clean_invalid_paths()
            messagebox.showinfo("Completato", "Database ripulito con successo")
            
    def quit_app(self):
        """Chiude l'applicazione e termina il processo"""
        if messagebox.askyesno("Esci", "Vuoi davvero chiudere il programma?"):
            self.root.destroy()
            sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = ShoeRecognitionApp(root)
    root.mainloop()