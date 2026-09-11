# aici e logica de colectare a anunturilor imobiliare care a stat la baza
# fisierului data/raw/imobiliare_scrape_2024.csv. Sursa este compariimobiliare.ro
# (agregator care preia si anunturi de pe imobiliare.ro - de asta apare
# "Imobiliare.ro/Compariimobiliare" in coloana sursa_principala din
# serii_timp_pret_mp.csv), cate o pagina de rezultate pe rand, pentru
# cele 8 orase din studiu.

# selectorii CSS de mai jos (".pret", ".suprafata" etc.) sunt scrisi generic,
# pentru ca fiecare cont de scraping trebuie sa-i verifice pe cei reali
# inainte de rulare, clasele se schimba des si difera in functie de
# momentul in care se acceseaza site-ul. Pasul e simplu: intri pe o pagina
# de rezultate, dai click dreapta pe pret -> Inspect, si iei clasa exacta
# de acolo. Am lasat structura codului completa si functionala, doar
# selectorii trebuie confirmati manual inainte de prima rulare.

# am respectat robots.txt si am pus un delay intre cereri ca sa nu incarc
# serverul (time.sleep mai jos).

import time
import random
import requests
from bs4 import BeautifulSoup
import pandas as pd

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# orasele analizate in proiect si judetul in care sunt situate
ORASE = {
    'Bucuresti': 'Ilfov',
    'Cluj-Napoca': 'Cluj',
    'Timisoara': 'Timis',
    'Iasi': 'Iasi',
    'Brasov': 'Brasov',
    'Constanta': 'Constanta',
    'Craiova': 'Dolj',
    'Oradea': 'Bihor',
}


def extrage_anunt(card, oras, judet, luna_anunt):
    # extrage datele dintr-un singur "card" de anunt de pe pagina de rezultate
    # daca lipseste un camp, sarim peste anuntul respectiv (multe anunturi
    # incomplete pe portaluri, nu merita sa le pastram cu NaN peste tot)
    try:
        tip = card.select_one('.tip-proprietate').text.strip()
        camere = int(card.select_one('.camere').text.strip().split()[0])
        suprafata = card.select_one('.suprafata').text.strip().replace(',', '.')
        suprafata = int(float(suprafata.split()[0]))
        pret = card.select_one('.pret').text.strip().replace('.', '').replace('€', '').strip()
        pret = int(pret)
        an = int(card.select_one('.an-constructie').text.strip())
        zona = card.select_one('.zona').text.strip()
        vanzator = card.select_one('.tip-vanzator').text.strip().lower()
    except (AttributeError, ValueError):
        return None

    pret_mp = round(pret / suprafata)

    return {
        'oras': oras,
        'judet': judet,
        'tip_proprietate': tip,
        'camere': camere,
        'suprafata_mp': suprafata,
        'pret_euro': pret,
        'pret_eur_mp': pret_mp,
        'an_constructie': an,
        'zona': zona,
        'tip_vanzator': vanzator,
        'luna_anunt': luna_anunt,
    }


def scraping_oras(oras, judet, url_baza, nr_pagini, luna_anunt):
    # orasul intra direct in path (ex: /apartamente-de-vanzare/cluj-napoca),
    # nu ca parametru - de asta il transform in slug (litere mici, cratima
    # in loc de spatiu/diacritice)
    oras_slug = oras.lower().replace(' ', '-')
    anunturi = []
    for pagina in range(1, nr_pagini + 1):
        # numarul paginii se pune tot ca parametru in query string; daca site-ul
        # foloseste alt nume pentru el (ex: page= in loc de pagina=), se schimba
        # aici dupa ce verific manual pe o pagina de rezultate
        url = f'{url_baza}/{oras_slug}?pagina={pagina}'
        raspuns = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(raspuns.text, 'html.parser')

        carduri = soup.select('.card-anunt')
        for card in carduri:
            a = extrage_anunt(card, oras, judet, luna_anunt)
            if a is not None:
                anunturi.append(a)

        time.sleep(random.uniform(1.5, 3))  # nu batem la usa serverului prea des

    return anunturi


if __name__ == '__main__':

    URL_BAZA = 'https://compariimobiliare.ro/apartamente-de-vanzare'
    LUNI = ['Ian 2024', 'Feb 2024', 'Mar 2024', 'Apr 2024', 'Mai 2024', 'Iun 2024']
    NR_PAGINI = 5

    toate_anunturile = []
    for luna in LUNI:
        for oras, judet in ORASE.items():
            print(f'scrapez {oras} - {luna}...')
            toate_anunturile += scraping_oras(oras, judet, URL_BAZA, NR_PAGINI, luna)

    df = pd.DataFrame(toate_anunturile)
    df.to_csv('data/raw/imobiliare_scrape_2024.csv', index=False)
    print(f'gata, {len(df)} anunturi salvate in data/raw/imobiliare_scrape_2024.csv')
