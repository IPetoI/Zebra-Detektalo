from random import randint

import cv2 as cv
import numpy as np

# Eredeti kepek betoltese
kep = cv.imread('crosswalk.jpg')
Szurke_kep = cv.imread('crosswalk.jpg', cv.IMREAD_GRAYSCALE)

kep1 = cv.imread('dupla-zebra.jpg')
Szurke_kep1 = cv.imread('dupla-zebra.jpg', cv.IMREAD_GRAYSCALE)

kep2 = cv.imread('kerekparos-zebra.jpg')
Szurke_kep2 = cv.imread('kerekparos-zebra.jpg', cv.IMREAD_GRAYSCALE)

kep3 = cv.imread('ff-zebra.jpg')
Szurke_kep3 = cv.imread('ff-zebra.jpg', cv.IMREAD_GRAYSCALE)


# So bors zaj
def soBors(img):
    sor, oszlop = img.shape

    # Feher lesz random a kisorsolt szamnak megfelel mennyisegnek
    db = randint(300, 10000)
    for i in range(db):
        # Random koordinata
        y_koord = randint(0, sor - 1)

        # Random koordinata
        x_koord = randint(0, oszlop - 1)

        # Feherre tetel
        img[y_koord][x_koord] = 255

    # Ugyan az feketevel
    db = randint(300, 10000)
    for i in range(db):
        y_koord = randint(0, sor - 1)

        x_koord = randint(0, oszlop - 1)

        img[y_koord][x_koord] = 0

    return img


# Lepteto ablak letrehozasa
cv.namedWindow('Lepteto')
cv.resizeWindow('Lepteto', 320, 50)


# Kezelofelulet megvalositasa
def kezeloFelulet(valtozo):
    kivalasztott = cv.getTrackbarPos('Kep', 'Lepteto')

    folyamat(kivalasztott)


cv.createTrackbar('Kep', 'Lepteto', 0, 3, kezeloFelulet)


# Kep kivalasztas / Kontraszt / Fekete-Feher / Morfologiai szures / Simitas
def folyamat(kivalasztott):
    global Vegeredmeny, kep_szurke

    if kivalasztott == 0:
        Vegeredmeny = kep.copy()
        kep_szurke = Szurke_kep.copy()

    elif kivalasztott == 1:
        Vegeredmeny = kep1.copy()
        kep_szurke = Szurke_kep1.copy()

    elif kivalasztott == 2:
        Vegeredmeny = kep2.copy()
        kep_szurke = Szurke_kep2.copy()

    elif kivalasztott == 3:
        Vegeredmeny = kep3.copy()
        kep_szurke = Szurke_kep3.copy()

    # soBors(kep_szurke)
    # cv.imshow("soBors", kep_szurke)

    # Addit alap
    # zaj = np.zeros(kep_szurke.shape[:2], np.int16)
    # cv.randn(zaj, 0.0, 20.0)

    # Gyenge addit | képpontok intenzitását változtatja meg / előjel nélküli
    # kep_szurke = cv.add(kep_szurke, zaj, dtype=cv.CV_8UC1)
    # cv.imshow("gyenge_add", kep_szurke)

    # Eros addit | képpontok intenzitását változtatja meg / előjeles
    # zaj2 = cv.add(kep_szurke, zaj, dtype=cv.CV_16SC1)
    # kep_szurke = cv.normalize(zaj2, None, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    # cv.imshow("eros_add", kep_szurke)

    # Kontraszt
    faktor = (259 * (150 + 255)) / (255 * (259 - 150))  # Kontraszt
    x = np.arange(0, 256, 1)
    zebra_kontraszt = np.uint8(np.clip(-80 + faktor * (np.float32(x) - 128.0) + 128, 0, 255))   # Fenyesseg
    kontraszt_kesz = cv.LUT(kep_szurke, zebra_kontraszt, None)

    # Fekete feher kep letrehozasa
    (kuka, fekete_feher) = cv.threshold(kontraszt_kesz, 130, 255, cv.THRESH_BINARY)

    # Morfologiai szures | Min-Max
    struktura = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    dst1 = cv.dilate(fekete_feher, struktura)   # max szűrő
    dst2 = cv.erode(dst1, struktura)    # min szűrő
    dst3 = cv.erode(dst2, struktura)
    dst4 = cv.dilate(dst3, struktura)

    # cv.imshow("dst1", dst1)
    # cv.imshow("dst2", dst2)
    # cv.imshow("dst3", dst3)
    # cv.imshow("dst4", dst4)
    # cv.waitKey(0)

    # Simitas
    homalyos = cv.GaussianBlur(dst4, (3, 3), 5)

    # cv.imshow("homalyos", homalyos)
    # cv.waitKey(0)

    # Eltunjon az arnyek ---> dst5 a konturkeresobe 
    # el = cv.Canny(kep_szurke, 1000, 7000, None, 5)
    # cv.imshow('Canny', el)
    # dst5 = cv.dilate(el, struktura)
    # cv.imshow('dst5', dst5)

    # Kontur megrajzolasa
    kontur, hierarchia = cv.findContours(homalyos, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for szam in kontur:
        ter = cv.contourArea(szam)  # Kontur terulete
        x, y, w, h = cv.boundingRect(szam)  # Kontur parameterei
        if (w > 160 and h > 5) or ter > 210:  # Feltetelek
            cv.drawContours(Vegeredmeny, [szam], -1, (0, 0, 255), cv.FILLED, cv.LINE_4)  # Kontur rajzolasa

    cv.imshow("Vegeredmeny", Vegeredmeny)
    cv.imshow("Kontraszt / Fekete-Feher / Morfologiai szures / Simitas", homalyos)


kezeloFelulet(0)

cv.waitKey(0)
cv.destroyAllWindows()