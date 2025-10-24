from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np
import bisect


def preprocess(img):
    '''
        Funzione che applica una serie di trasformazioni per processare l'immagine e utilizzare il machine learning
        tramite l'algoritmo K-Means clustering
    '''
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)        # Converte immagine BGR --> RGB per K-Means
    img = cv.GaussianBlur(img, (5, 5), 0)           # Rimuove rumore e piccoli dettagli (5x5 kernel, sigmaX=0)
    img = img / 255.0                               # Normalizza integer --> floating-point
    cv.imwrite('output/preprocessedOimg.jpg', img)  #posiziono immagine preprocessata in file system

    return img

def kMeans_cluster(img, n_clusters=3):
    '''
        Funzione che esegue clustering K-Means per ridurre numero di colori a un numero specificato di cluster
        Tratta ogni pixel come un punto dati nello spazio RGB, li raggruppa in cluster e ricostruisce l'immagine 
        utilizzando i centroidi dei cluster come nuovi valori dei pixel.
        Colori simili vengono uniti.

        Kmeans()    -->  Modello apprende i centri dei cluster (centriodi) minimizzando la distanza dei punti dai centri loro assegnati
                    -->  input:numero specifico di cluster e uno stato casuale fisso per riproducibilità
    '''
    image_2D = img.reshape(img.shape[0] * img.shape[1], img.shape[2])           #3D image --> 2D array (riga = valore RGB pixel )
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_2D) 
    #Creo array dove ogni pixel è sostituito dal centroide RGB del proprio cluster 
    clustOut = kmeans.cluster_centers_[kmeans.labels_]                          #per ogni pixel(labels_) assegna corrispondente centriodo dal modello adattato
    clustered_3D = clustOut.reshape(img.shape[0], img.shape[1], img.shape[2])   #2D image --> 3D image
    clusteredImg = np.uint8(clustered_3D * 255)                                 #floating-point --> unit8 integer
    cv.imwrite('output/clusteredImg.jpg', clusteredImg)
    return clusteredImg

def edgeDetection(clusteredImage):
    '''
        Funzione che processa una immagine clusterizzata per trovare e rifinire i bordi
    '''
    gray = cv.cvtColor(clusteredImage, cv.COLOR_BGR2GRAY)       #BGR --> scala del grigio (riduce a singolo canale)
    edged = cv.Canny(gray, 50, 150)                             #identifica i bordi in base all'intensità del gradiente
    edged = cv.dilate(edged, None, iterations=2)                #espande regioni dei bordi bianchi rendendoli + spessi e ripenendo piccoli spazi vuoti
    edged = cv.erode(edged, None, iterations=1)                 #riduce leggermente aree bianche, rimuove rumore/sporgenze sottili
    cv.imwrite('output/edgedImg.jpg', edged)
    return edged

def getBoundingBox(img):
    '''
        Processa una immagine binaria per trovare il contorno, lo ordina in aree, approssima le figure come poligoni e 
        calcola i rettangoli di delimitazione per ognuno

        Output: rettangolo di delimitazione, contorni originali, poligono approssimativi, immagine di input
    '''
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)    #contorni= elenco di punti che  delianenao componenti connesse, la gerarchia viene ignorata
    #i contorni più larghi, spesso corrispondono all'oggetto principale
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)  #ordina i contorni in ordine discendente di area usando lambda function come chiavi
    contours_poly = [None] * len(contours)                                      #elenco per memorizzare poligoni approssimativi per ogni contorno con stessa lunghezza dell'elenco 
    boundRect = [None] * len(contours)                                          #elenco per memorizzare rettangolo di delimitazione per ogni contorno
    for i, c in enumerate(contours):                                            #loop per ogni contonro con indice i e contorno c
        contours_poly[i] = cv.approxPolyDP(c, 3, True)                          #approssima il contorno come un poligono (con Douglas-Peacker algoritm e tolleranza = 3)
        #Funzione che restitusci il triangolo verticale più piccolo che racchiuda il contorno
        boundRect[i] = cv.boundingRect(contours_poly[i])                        #calcola rettangolo di delimitazione per poligono approssimato 

    return boundRect, contours, contours_poly, img  

def calcFeetSize(pcropedImg, boundRect):
    '''
        Calcola la lunghezza stimata della suola in mm basandosi sull'immagine della suola (considerata su un foglio A4) e il suo rettangolo
        di delimitazione
    '''
    if not boundRect:                                   #controllo che lista dei delimitatori del rettangono sia piena
        return 0.0
    fh = boundRect[0][3]                                # Altezza suola (primo/principale contorno)
    fw = boundRect[0][2]                                # Larghezza
    ph = pcropedImg.shape[0]                            # Altezza immagine (A4)
    pw = pcropedImg.shape[1]                            # Larghezza
    opw = 210                                           # A4 mm width
    oph = 297                                           # A4 mm height
    scale_h = oph / ph                                  # Calcola fattore di scala dell'altezza mm reali per pixel in altezza
    scale_w = opw / pw                                  # Calcola fattore di scala della lunghezza, mm reali per pixel in lunghezza
    sole_length_mm = max(fh * scale_h, fw * scale_w)    # Calcola lunghezza suola in mm ridimensionando h e w prendendo valore max
    return sole_length_mm

#Dizionario contente tabella delle taglie
SHOE_SIZES = [
    (23.5, {'EU': 37, 'US': 5, 'UK': 4.5}),
    (24.0, {'EU': 38, 'US': 6, 'UK': 5.5}),
    (24.5, {'EU': 39, 'US': 6.5, 'UK': 6}),
    (25.0, {'EU': 40, 'US': 7, 'UK': 6.5}),
    (25.5, {'EU': 41, 'US': 8, 'UK': 7.5}),
    (26.0, {'EU': 42, 'US': 9, 'UK': 8.5}),
    (26.5, {'EU': 43, 'US': 10, 'UK': 9.5}),
    (27.0, {'EU': 44, 'US': 10.5, 'UK': 10}),
    (27.5, {'EU': 45, 'US': 11, 'UK': 10.5}),
    (28.0, {'EU': 46, 'US': 12, 'UK': 11.5}),
]

def getSizeFromLength(length_cm):
    '''
        Prendono la lunghezza della suola in  cm e la mappano con  la taglia più vicina definita dal dizionario sopra.
        Utilizzano la ricercha binaria per trovare il punto di inserimento, quindi sleziona la dimensione più vicina confrontando con le distanza
    '''
    lengths = [row[0] for row in SHOE_SIZES]                                #estrare la lunghezza da tutte le righe del dizionario (che DEVE essere ordinato)
    idx = bisect.bisect_left(lengths, length_cm)                            #uso ricerca binaria per trovare l'indice corrispondende alla lunghezza in cm della suola
    if idx == 0:                                                            #se idx = 0 cm suola è minore o uguale alla lunghezza minima
        return SHOE_SIZES[0][1]
    elif idx == len(lengths):                                               #se idx è uguale alla lunghezza dell'elenco, cm suola è maggiore a tutto e restitusco la taglia più grande 
        return SHOE_SIZES[-1][1]
    else:
        if abs(length_cm - lengths[idx-1]) < abs(length_cm - lengths[idx]): #altrimenti idx è compreso tra 1 e lengths - 1 confronta le distanze con le due taglie più vicine
            return SHOE_SIZES[idx-1][1]                                     #cm suola è più vicina alla taglia inferiore
        else:
            return SHOE_SIZES[idx][1]                                       #cm suola è più vicina alla taglia superiore