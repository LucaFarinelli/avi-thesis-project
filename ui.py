import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk, ImageDraw
import threading
import os
import sys

import Main
import Config

class ShoeRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Riconoscimento Oggetti")
        self.root.attributes("-fullscreen", True)  # fullscreen funziona ora
        # self.root.geometry("520x540")   # opzionale per debug
        # self.root.resizable(False, False)

        # Stile e colori
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except:
            pass

        self.bg_color = "#213d72"
        self.accent_color = "#3b82f6"
        self.success_color = "#10b981"
        self.warning_color = "#f59e0b"
        self.danger_color = "#ef4444"
        self.text_color = "#ffffff"

        self.root.configure(bg=self.bg_color)
        self.is_processing = False

        self._configure_styles()

        # --- Frame principale (ora senza padding esterno per permettere full-width) ---
        main_frame = tk.Frame(root, bg=self.bg_color)
        main_frame.pack(fill="both", expand=True)

        # Configura griglia per main_frame
        main_frame.grid_rowconfigure(0, weight=0)   # header
        main_frame.grid_rowconfigure(1, weight=0)   # separatore
        main_frame.grid_rowconfigure(2, weight=0)   # combobox
        main_frame.grid_rowconfigure(3, weight=0)   # bottoni
        main_frame.grid_rowconfigure(4, weight=1)   # opzioni (espandibile)
        main_frame.grid_rowconfigure(5, weight=0)   # status
        
        main_frame.grid_columnconfigure(0, weight=1) # Colonna Controlli
        main_frame.grid_columnconfigure(1, weight=2) # Colonna Immagine (più larga)

        # --- Header (ora a larghezza piena) ---
        header_frame = tk.Frame(main_frame, bg=self.bg_color)
        header_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(30, 20))

        title_frame = tk.Frame(header_frame, bg=self.bg_color)
        title_frame.pack()

        try:
            img_sx = Image.open("logo_unife.png").resize((70, 70))
            self.logo_sx = ImageTk.PhotoImage(img_sx)
            logo_sx_label = tk.Label(title_frame, image=self.logo_sx, bg=self.bg_color)
            logo_sx_label.pack(side="left", padx=(0, 10))

            img_dx = Image.open("logo_unife.png").resize((70, 70))
            self.logo_dx = ImageTk.PhotoImage(img_dx)
            logo_dx_label = tk.Label(title_frame, image=self.logo_dx, bg=self.bg_color)
        except Exception as e:
            print(f"Errore caricamento immagini: {e}")

        title_label = tk.Label(
            title_frame,
            text="Automated Visual Inspection",
            font=("Segoe UI", 18, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        )
        title_label.pack(side="left")
        logo_dx_label.pack(side="left", padx=(10, 0))

        subtitle_label = tk.Label(
            header_frame,
            text="Seleziona il tipo di oggetto e avvia l'analisi",
            font=("Segoe UI", 10),
            bg=self.bg_color,
            fg="#9ca3af"
        )
        subtitle_label.pack()

        # --- Separatore (ora a larghezza piena) ---
        separator = tk.Frame(main_frame, height=2, bg="#e5e7eb")
        separator.grid(row=1, column=0, columnspan=2, sticky='ew', pady=10)

        # --- Combobox (allineato a sinistra con margine) ---
        selection_frame = tk.Frame(main_frame, bg=self.bg_color)
        selection_frame.grid(row=2, column=0, sticky='w', padx=40, pady=(10,0))

        db_object_types = Config.get_available_object_types()
        if not db_object_types:
            db_object_types = ["shoe", "bottle"]
        self.object_options = ["Selezionare il tipo di oggetto"] + db_object_types
        self.object_var = tk.StringVar()
        
        self.cb_object = ttk.Combobox(
            selection_frame,
            textvariable=self.object_var,
            values=self.object_options,
            state="readonly",
            style="Custom.TCombobox",
            width=49,
            font=("Segoe UI", 15, "bold"),
        )
        self.cb_object.set("Selezionare il tipo di oggetto")
        self.cb_object.pack(pady=12, anchor="w", ipady=15)   # margine sinistro già dato dal frame

        # --- Griglia Opzioni (Checkbox Circolari) ---
        options_frame = tk.Frame(main_frame, bg=self.bg_color)
        options_frame.grid(row=4, column=0, sticky='nw', padx=40, pady=(0, 25))
        
        self.options_vars = {}
        # (etichetta, chiave_flag, abilitata_default)
        options_list = [
            ("Rimozione Sfondo (AI)",   "do_bg_remove",  True),
            ("Analisi Difetti (SVM)",   "do_svm",        True),
            ("Analisi Texture (LBP)",   "do_lbp",        True),
            ("Analisi Colore (HSV)",    "do_hsv",        True),
            ("Matching Database",       "do_matching",   True),
            ("Calcolo Dimensioni",      "do_dimensions", True),
            ("Segmentazione K-Means",   "do_kmeans",     False),
            ("Rilevamento Bordi (Canny)","do_canny",     False),
        ]
        
        for i, (label, key, default) in enumerate(options_list):
            var = tk.BooleanVar(value=default)
            self.options_vars[key] = var
            cb = ttk.Checkbutton(
                options_frame,
                text=label,
                variable=var,
                style="Circular.TCheckbutton"
            )
            cb.grid(row=i // 2, column=i % 2, sticky='w', padx=(0, 40), pady=6)

        # --- Bottoni (impilati verticalmente, allineati a sinistra) ---
        buttons_frame = tk.Frame(main_frame, bg=self.bg_color)
        buttons_frame.grid(row=3, column=0, sticky='nw', padx=40, pady=(10, 25))

        btn_file = ttk.Button(
            buttons_frame,
            text="📁 Carica da File",
            command=self.process_from_file,
            style="File.TButton",
            width=50
        )
        btn_file.pack(pady=12, anchor="w", ipadx=5, ipady=20)

        btn_webcam = ttk.Button(
            buttons_frame,
            text="📷 Cattura da Webcam",
            command=self.process_from_webcam,
            style="Webcam.TButton",
            width=50
        )
        btn_webcam.pack(pady=12, anchor="w", ipadx=5, ipady=20)

        btn_clean = ttk.Button(
            buttons_frame,
            text="🗑️ Pulisci Database",
            command=self.clean_database,
            style="Clean.TButton",
            width=50
        )
        btn_clean.pack(pady=12, anchor="w", ipadx=5, ipady=20)

        btn_close = ttk.Button(
            buttons_frame,
            text="❌ Chiudi Programma",
            command=self.quit_app,
            style="Quit.TButton",
            width=50
        )
        btn_close.pack(pady=12, anchor="w", ipadx=5, ipady=20)

        # --- Pannello Visualizzazione Immagine (Destra, in alto) ---
        self.image_view_frame = tk.Frame(main_frame, bg=self.bg_color, bd=0, highlightthickness=0)
        self.image_view_frame.grid(row=2, column=1, rowspan=2, sticky='nsew', padx=(20, 40), pady=(5, 6))
        
        self.image_label = tk.Label(
            self.image_view_frame, 
            text="ANTEPRIMA ELABORAZIONE", 
            font=("Segoe UI", 12, "bold"),
            bg=self.bg_color, 
            fg="#9ca3af"
        )
        self.image_label.pack(expand=True, fill="both", padx=10, pady=10)

        # --- Box Risultati (Destra, sotto l'imagebox) ---
        results_frame = tk.Frame(main_frame, bg="#162d58", bd=0, highlightthickness=1,
                                 highlightbackground="#334155")
        results_frame.grid(row=4, column=1, rowspan=2, sticky='nsew', padx=(20, 40), pady=(0, 20))

        results_title = tk.Label(
            results_frame,
            text="📊 Risultati Analisi",
            font=("Segoe UI", 10, "bold"),
            bg="#162d58",
            fg="#93c5fd",
            anchor="w"
        )
        results_title.pack(fill="x", padx=10, pady=(6, 2))

        sep_res = tk.Frame(results_frame, height=1, bg="#334155")
        sep_res.pack(fill="x", padx=10, pady=(0, 4))

        self.results_label = tk.Label(
            results_frame,
            text="In attesa di elaborazione...",
            font=("Consolas", 9),
            bg="#162d58",
            fg="#94a3b8",
            anchor="nw",
            justify="left",
            wraplength=550
        )
        self.results_label.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        # --- Status bar ---
        status_frame = tk.Frame(main_frame, bg=self.bg_color)
        status_frame.grid(row=5, column=0, columnspan=2, sticky='ew', padx=40, pady=(10, 0))

        self.status_label = tk.Label(
            status_frame,
            text="✅ Pronto",
            font=("Segoe UI", 16, "bold"),
            bg=self.bg_color,
            fg=self.success_color,
            anchor="w"
        )
        self.status_label.pack(side="left")

    def _configure_styles(self):
        # Stile base per bottoni (dimensioni ragionevoli)
        self.style.configure(
            "Base.TButton",
            font=("Segoe UI", 25, "bold"),
            padding=(40, 20),          # (orizzontale, verticale)
            borderwidth=0,
            focusthickness=0,
            focuscolor="none",
            relief="flat"
        )

        # Ereditano da Base.TButton
        self.style.configure("File.TButton", parent="Base.TButton",
                             background="#10b981", foreground="white", font=("Segoe UI", 15, "bold"))
        self.style.map("File.TButton",
                       background=[("active", "#059669"), ("pressed", "#047857")])

        self.style.configure("Webcam.TButton", parent="Base.TButton",
                             background="#3b82f6", foreground="white", font=("Segoe UI", 15, "bold"))
        self.style.map("Webcam.TButton",
                       background=[("active", "#2563eb"), ("pressed", "#1d4ed8")])

        self.style.configure("Clean.TButton", parent="Base.TButton",
                             background="#f59e0b", foreground="white", font=("Segoe UI", 15, "bold"))
        self.style.map("Clean.TButton",
                       background=[("active", "#d97706"), ("pressed", "#b45309")])

        self.style.configure("Quit.TButton", parent="Base.TButton",
                             background=self.danger_color, foreground="white", font=("Segoe UI", 15, "bold"))
        self.style.map("Quit.TButton",
                       background=[("active", "#dc2626"), ("pressed", "#b91c1c")])

        # Stile per combobox (stessa altezza e font)
        self.style.configure(
            "Custom.TCombobox",
            font=("Segoe UI", 25, "bold"),
            padding=8,
            fieldbackground="#334155", # Grigio fumo scuro
            background="#334155",
            foreground="white",
            arrowcolor=self.accent_color
        )
        
        # --- Configurazione Font Dropdown (Menu a tendina) ---
        # ttk.Style non influenza il Listbox interno del Combobox. 
        # Bisogna usare option_add per modificare il font degli elementi nel menu.
        self.root.option_add('*TCombobox*Listbox.font', ("Segoe UI", 15))
        self.root.option_add('*TCombobox*Listbox.background', "#334155")
        self.root.option_add('*TCombobox*Listbox.foreground', "white")
        self.root.option_add('*TCombobox*Listbox.selectBackground', self.accent_color)
        self.root.option_add('*TCombobox*Listbox.selectForeground', "white")

        # --- Generazione Immagini Checkbox Circolari ---
        self._create_round_checkbox_images()

        # Layout per Circular.TCheckbutton
        # Usiamo l'elemento 'Circular.indicator' creato dinamicamente
        self.style.element_create("Circular.indicator", "image", 
                                  self.img_unchecked, 
                                  ("selected", self.img_checked))
        
        self.style.layout("Circular.TCheckbutton", [
            ("Checkbutton.padding", {"sticky": "nswe", "children": [
                ("Circular.indicator", {"side": "left", "sticky": ""}),
                ("Checkbutton.label", {"side": "left", "sticky": ""})
            ]})
        ])
        
        self.style.configure("Circular.TCheckbutton", 
                             font=("Segoe UI", 15, "bold"), 
                             background=self.bg_color,
                             foreground=self.text_color,
                             padding=5)
        self.style.map("Circular.TCheckbutton", 
                       foreground=[("active", self.accent_color)])

    def _create_round_checkbox_images(self, size=24):
        """Crea dinamicamente le immagini per i checkbox circolari"""
        # Immagine Unchecked (cerchio vuoto)
        img_un = Image.new('RGBA', (size, size), (0,0,0,0))
        draw_un = ImageDraw.Draw(img_un)
        draw_un.ellipse([2, 2, size-3, size-3], outline="#9ca3af", width=2)
        self.img_unchecked = ImageTk.PhotoImage(img_un)

        # Immagine Checked (cerchio con pallino colorato)
        img_ch = Image.new('RGBA', (size, size), (0,0,0,0))
        draw_ch = ImageDraw.Draw(img_ch)
        # Bordo
        draw_ch.ellipse([2, 2, size-3, size-3], outline=self.accent_color, width=2)
        # Nucleo (pallino)
        draw_ch.ellipse([6, 6, size-7, size-7], fill=self.accent_color)
        self.img_checked = ImageTk.PhotoImage(img_ch)

    def update_image_display(self, img_path):
        """Aggiorna l'immagine nel pannello di destra in modo thread-safe, con delay di 2s tra le immagini."""
        import time
        time.sleep(2)  # Pausa visibile tra i passaggi
        def _update():
            try:
                if not os.path.exists(img_path):
                    return
                
                img = Image.open(img_path)
                
                target_w = self.image_view_frame.winfo_width() - 20
                target_h = self.image_view_frame.winfo_height() - 20
                
                if target_w < 100: target_w = 600
                if target_h < 100: target_h = 500
                
                img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo
                
            except Exception as e:
                print(f"[ERRORE] Aggiornamento immagine: {e}")
        
        self.root.after(0, _update)

    def update_results_display(self, text):
        """Aggiorna il box dei risultati in modo thread-safe."""
        def _update():
            self.results_label.config(text=text, fg="#e2e8f0")
        self.root.after(0, _update)

    def get_selected_object(self):
        """Ritorna il tipo di oggetto selezionato"""
        return self.object_var.get()

    def get_options(self):
        """Ritorna un dizionario con i flag attuali delle checkbox."""
        return {key: var.get() for key, var in self.options_vars.items()}
    
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
        if selected_object == "Selezionare il tipo di oggetto":
            messagebox.showwarning("Avviso", "Seleziona un tipo di oggetto valido.")
            return

        object_type = selected_object
        self.update_status(f"📂 Oggetto selezionato: {selected_object}", self.accent_color)
        self.update_results_display("")
        
        file_path = filedialog.askopenfilename(
            initialdir="input/",
            filetypes=[("Immagini", "*.jpg *.jpeg *.png"), ("Tutti i file", "*.*")]
        )
        
        if not file_path:
            return
        
        thread = threading.Thread(
            target=self._process_image_thread,
            args=(file_path, selected_object, object_type, self.get_options()),
        )
        thread.start()
    
    def process_from_webcam(self):
        """Elabora un'immagine da webcam"""
        if self.is_processing:
            messagebox.showwarning("Avviso", "Elaborazione in corso...")
            return
        
        selected_object = self.get_selected_object()
        if selected_object == "Selezionare il tipo di oggetto":
            messagebox.showwarning("Avviso", "Seleziona un tipo di oggetto valido.")
            return

        object_type = selected_object
        self.update_status(f"📷 Oggetto selezionato: {selected_object}", self.accent_color)
        self.update_results_display("")
        
        thread = threading.Thread(
            target=self._process_webcam_thread,
            args=(selected_object, object_type, self.get_options()),
        )
        thread.start()
    
    def _process_image_thread(self, file_path, object_label, object_type, options):
        """Thread per elaborare immagine da file"""
        try:
            self.is_processing = True
            self.update_status(f"📂 Caricamento immagine ({object_label})...", self.accent_color)
            
            import cv2
            img = cv2.imread(file_path)
            
            if img is None:
                self.update_status("❌ Errore: impossibile caricare l'immagine", self.danger_color)
                messagebox.showerror("Errore", "Impossibile caricare l'immagine")
                return
            
            self.update_status("⚙️ Elaborazione in corso...", self.accent_color)
            
            print(f"[INFO] Tipo oggetto: {object_type}")
            Main.process_image(img, file_path, is_ui=True, 
                               callback=self.update_image_display,
                               result_callback=self.update_results_display,
                               options=options,
                               object_type=object_type)
            
            self.update_status("✅ Elaborazione completata!", self.success_color)
            self.safe_messagebox("Successo", f"Elaborazione di {object_label} completata con successo!")
            
        except Exception as e:
            self.update_status(f"❌ Errore: {str(e)}", self.danger_color)
            messagebox.showerror("Errore", f"Errore durante l'elaborazione:\n{str(e)}")
        
        finally:
            self.is_processing = False
    
    def _process_webcam_thread(self, object_label, object_type, options):
        """Thread per elaborare immagine da webcam"""
        try:
            self.is_processing = True
            self.update_status(
                f"📷 Cattura da webcam ({object_label})... (premi SPACE per scattare)",
                self.accent_color,
            )
            
            img, path = Main.capture_from_webcam()
            
            if img is None:
                self.update_status("⚠️ Cattura annullata", self.warning_color)
                return
            
            self.update_status("⚙️ Elaborazione in corso...", self.accent_color)
            
            print(f"[INFO] Tipo oggetto: {object_type}")
            Main.process_image(img, path, is_ui=True, 
                               callback=self.update_image_display,
                               result_callback=self.update_results_display,
                               options=options,
                               object_type=object_type)
            
            self.update_status("✅ Elaborazione completata!", self.success_color)
            self.safe_messagebox("Successo", f"Elaborazione di {object_label} completata con successo!")
            
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