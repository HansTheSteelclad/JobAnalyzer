from django.http import HttpResponseRedirect
from django.shortcuts import render, HttpResponse
from .forms import *
import io
import urllib, base64
import matplotlib
import matplotlib.colors as mcolors
matplotlib.use('TkAgg')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.dates as mdates
import statsmodels.api as sm
from datetime import datetime
from datetime import time
from dateutil.relativedelta import relativedelta
import time
import datetime as dt
import calendar
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import textwrap
import numpy as np
import requests
from collections import defaultdict
import pandas as pd
import json
from django.shortcuts import render
# Globals
global nr_strony, nazwa_miasta, nazwa_zawodu, odleglosc, minimalna
nr_strony = 1
nazwa_miasta = ''
nazwa_zawodu = ''
odleglosc = 0
minimalna = 0

# Functions

def jobs_get(url, params):

    response = requests.get(url, params=params)
    response_json = response.json()

    jobs = np.empty((0, 5), dtype=object)

    for i in response_json['results']:
        job = np.empty(5, dtype=object)

        try:
            job[0] = i['title']
        except:
            job[0] = ' '

        try:
            job[1] = i['company']['display_name']
        except:
            job[1] = ' '

        try:
            job[2] = i['location']['display_name']
        except:
            job[2] = ' '

        try:
            job[3] = i['description']
        except:
            job[3] = ' '

        try:
            job[4] = i['redirect_url']
        except:
            job[4] = ' '

        jobs = np.vstack((jobs, job))

    jobs_list = jobs.tolist()

    return jobs_list

def create_plot(url, params):

    response = requests.get(url, params=params)

    response_json = response.json()  # Zamiana z JSON na słownik Python

    # Sprawdzenie, czy w odpowiedzi są wyniki
    if 'results' in response_json and response_json['results']:
        jobs = []  # Lista, w której będziemy przechowywać oferty pracy
        region_count = defaultdict(int)  # Słownik do zliczania ofert w województwach
        # salary_data = []  # Lista, w której będziemy przechowywać dane o wynagrodzeniu
        company_count = defaultdict(int)  # Słownik do zliczania ofert w firmach
        job_count = defaultdict(int)  # Słownik do zliczania ofert według stanowisk
        salary_data = defaultdict(int)  # Słownik: klucz=(company_name, salary_min, salary_max), wartość=liczba ofert

        # Pętla po wynikach
        for i in response_json['results']:
            job = []

            # Zabezpieczenie przy pobieraniu danych
            job_title = i.get('title', 'Brak tytułu')
            company_name = i.get('company', {}).get('display_name', 'Brak firmy')
            location = i.get('location', {}).get('display_name', 'Brak lokalizacji')
            description = i.get('description', 'Brak opisu')

            salary_min = i.get('salary_min', None)  # Minimalne wynagrodzenie
            salary_max = i.get('salary_max', None)  # Maksymalne wynagrodzenie

            if salary_min is not None and salary_max is not None:
                # Dodawanie unikalnych kombinacji firmy i widełek wynagrodzenia
                salary_data[(company_name, salary_min, salary_max)] += 1

            # Przypisanie województwa
            region = location.split(",")[-1].strip()  # Zakłada się, że województwo jest na końcu lokalizacji

            # Zliczanie ofert pracy w województwie
            region_count[region] += 1

            # Zliczanie ofert pracy w firmach
            company_count[company_name] += 1

            # Zliczanie ofert pracy według stanowisk
            job_count[job_title] += 1

            # Dodanie oferty do listy jobs
            job.append(job_title)
            job.append(company_name)
            job.append(location)
            job.append(description)

            # Dodanie do ogólnej listy 'jobs'
            jobs.append(job)

            # Wypisanie szczegółów oferty
            # print(f"Job Title: {job_title}")
            # print(f"Company: {company_name}")
            # print(f"Location: {location}")
            # print(f"Description: {description}\n")

        # Sortowanie województw według liczby ofert
        sorted_regions = sorted(region_count.items(), key=lambda x: x[1], reverse=True)

        # Sortowanie firm według liczby ofert
        sorted_companies = sorted(company_count.items(), key=lambda x: x[1], reverse=True)

        # Sortowanie stanowisk według liczby ofert
        sorted_jobs = sorted(job_count.items(), key=lambda x: x[1], reverse=True)

        # Wybieramy tylko top 10 zawodów
        top_10_jobs = sorted(sorted_jobs[:10], key=lambda x: x[1], reverse=False)



        cols = ['xkcd:purple', 'xkcd:green', 'xkcd:blue', 'xkcd:pink', 'xkcd:brown', 'xkcd:red', 'xkcd:orange',
                'xkcd:yellow', 'xkcd:grey', 'xkcd:teal', 'xkcd:light green', 'xkcd:light purple',
                'xkcd:turquoise', 'xkcd:lavender', 'xkcd:dark blue', 'xkcd:tan', 'xkcd:cyan', 'xkcd:aqua',
                'xkcd:maroon',
                'xkcd:light blue',
                'xkcd:salmon', 'xkcd:mauve', 'xkcd:hot pink', 'xkcd:lilac', 'xkcd:beige', 'xkcd:pale green',
                'xkcd:peach',
                'xkcd:mustard', 'xkcd:periwinkle', 'xkcd:rose', 'xkcd:forest green', 'xkcd:bright blue', 'xkcd:navy',
                'xkcd:baby blue', 'xkcd:light brown', 'xkcd:mint green', 'xkcd:gold', 'xkcd:grey blue',
                'xkcd:light orange',
                'xkcd:dark orange']

        # WYKRES 1
        try:
            if salary_data:
                # Rozpakowanie danych o wynagrodzeniu do osobnych list
                companies, min_salaries, max_salaries = zip(*salary_data)

                # Ustawienie szerokości słupków dla wykresu
                x = np.arange(len(companies))  # Pozycje dla firm
                width = 0.20  # Szerokość słupka

                # Tworzymy wykres
                plt.figure(figsize=(10, 7))

                # Rysowanie słupków (minimalne i maksymalne wynagrodzenia)
                plt.bar(x - width / 2, min_salaries, width, label='Minimalne wynagrodzenie', color='mediumaquamarine')
                plt.bar(x + width / 2, max_salaries, width, label='Maksymalne wynagrodzenie', color='salmon')

                max_length = 20  # Ustawiamy maksymalną długość jednej linii
                wrapped_labels = [textwrap.fill(company, width=max_length) for company in companies]

                # Dodanie tytułów i etykiet osi
                plt.title('Porównanie minimalnego i maksymalnego wynagrodzenia w ofertach pracy')
                plt.xlabel('Firma')
                plt.ylabel('Wynagrodzenie (PLN)')
                plt.xticks(x, wrapped_labels, rotation=45, ha='right')  # Etykiety na osi X (nazwy firm)
                plt.legend()

                plt.gca().spines['left'].set_color('black')
                plt.gca().spines['left'].set_linewidth(2)
                plt.gca().spines['bottom'].set_color('black')
                plt.gca().spines['bottom'].set_linewidth(2)
                plt.gca().spines['right'].set_color('black')
                plt.gca().spines['right'].set_linewidth(2)
                plt.gca().spines['top'].set_color('black')
                plt.gca().spines['top'].set_linewidth(2)

                # Dopasowanie układu z dodatkowymi marginesami na dole
                plt.tight_layout(pad=2.0)  # Zwiększono margines, aby uniknąć obcinania etykiet

                #plt.show()


                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                uri_1 = 'data:image/png;base64,' + urllib.parse.quote(string)
        except:
            uri_1 = ''

        # WYKRES 2

        try:

            # Tworzenie wykresu słupkowego
            regions, counts = zip(*sorted_regions)  # Rozpakowanie krotek do dwóch list (regiony i liczba ofert)

            # Tworzymy wykres za pomocą matplotlib.pyplot
            plt.figure(figsize=(10, 6))  # Ustalamy wielkość wykresu
            colors = cols[:len(regions)]
            plt.bar(regions, counts, color=colors)  # Tworzenie wykresu słupkowego

            # Dodanie tytułu i etykiet osi
            plt.title('Liczba ofert pracy w poszczególnych województwach')
            plt.xlabel('Województwa')
            plt.ylabel('Liczba ofert pracy')

            # Obrót nazw województw na osi X
            plt.xticks(rotation=45, ha='right')

            plt.gca().spines['left'].set_color('black')
            plt.gca().spines['left'].set_linewidth(2)
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['right'].set_color('black')
            plt.gca().spines['right'].set_linewidth(2)
            plt.gca().spines['top'].set_color('black')
            plt.gca().spines['top'].set_linewidth(2)

            # Dopasowanie układu z dodatkowymi marginesami na dole
            plt.tight_layout(pad=2.0)  # Zwiększono margines, aby uniknąć obcinania etykiet

            # Ustawienie liczb całkowitych na osi X
            plt.yticks(np.arange(0, max(counts) + 1, 2))

            # Pokazanie wykresu
            #plt.show()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri_2 = 'data:image/png;base64,' + urllib.parse.quote(string)

        except:
            uri_2 = ''

        # WYKRES 3

        try:

            # Tworzenie wykresu kołowego
            companies, counts = zip(*sorted_companies)  # Rozpakowanie krotek do dwóch list (firmy i liczba ofert)

            explode = tuple([0.025] * len(companies))

            plt.figure(figsize=(10,6))  # Ustalamy wielkość wykresu
            plt.pie(counts, labels=companies, autopct='%1.1f%%', startangle=90,
                    colors=cols,
                    textprops={'fontsize': 6},
                    explode=explode
                    )


            # Dodanie tytułu wykresu
            plt.title('Procentowy rozkład liczby ofert pracy w firmach')

            # Pokazanie wykresu
            plt.axis('equal')  # Upewniamy się, że wykres będzie okrągły

            #plt.show()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri_3 = 'data:image/png;base64,' + urllib.parse.quote(string)

        except:
            uri_3 = ''


        #WYKRES 4

        try:

            # Tworzenie wykresu słupkowego poziomego
            job_titles, counts = zip(*top_10_jobs)  # Rozpakowanie krotek do dwóch list (stanowiska i liczba ofert)

            # Tworzenie wykresu
            plt.figure(figsize=(10, 6))  # Ustawiamy szerokość i wysokość wykresu
            bars = plt.barh(job_titles, counts,
                     color=list(mcolors.TABLEAU_COLORS))  # Tworzymy wykres słupkowy poziomy

            # Dodanie tytułu i etykiet osi
            plt.title('Top 10 zawodów z największą liczbą ofert pracy')
            plt.xlabel('Liczba ofert')
            plt.ylabel('')

            # Dopasowanie układu wykresu
            plt.tight_layout(pad=1.0)
            plt.subplots_adjust(left=0.03, right=0.55,top=0.95, bottom=0.05)  # Zmniejsz prawy margines (sprawi, że wszystko będzie bardziej na lewo)


            # Dopasowanie czcionki etykiet (jeśli są zbyt długie)
            plt.yticks(fontsize=10, color='white')  # Można zmniejszyć czcionkę etykiet na osi Y

            # Umieszczenie nazw firm za słupkami (po prawej stronie)
            for bar, job in zip(bars, job_titles):
                width = bar.get_width()  # Pobierz szerokość słupka (liczba ofert)
                y_position = bar.get_y()  # Pobierz nazwę stanowiska (na osi Y)

                # Wstawienie etykiety z nazwą firmy po prawej stronie słupka
                plt.text(width + 0.1, y_position + bar.get_height() / 2,  # Pozycjonowanie etykiety
                         job,  # Nazwa stanowiska (firma)
                         va='center', ha='left', fontsize=10, color='purple',fontweight='bold')  # Ustawienie wyrównania i czcionki

            # Usunięcie górnej i prawej ramki
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)

            plt.gca().spines['left'].set_color('black')
            plt.gca().spines['left'].set_linewidth(4)
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['bottom'].set_linewidth(4)

            # Ustawienie liczb całkowitych na osi X
            plt.xticks(np.arange(0, max(counts) + 10, 1))

            # Ustawienie limitu osi X (rozciągnięcie osi X)
            plt.xlim(0, max(counts) * 1.1)  # Ustawia zakres osi X

            # Pokazanie wykresu
            #plt.show()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            string = base64.b64encode(buf.read())
            uri_4 = 'data:image/png;base64,' + urllib.parse.quote(string)

        except:
            uri_4 = ''

        #WYKRES 5

        try:

            data = response.json()

            # Zliczamy oferty pracy na pełny etat, część etatowe oraz brak danych
            full_time_count = 0
            part_time_count = 0
            unknown_count = 0  # Liczba ofert, dla których nie ma informacji o etacie

            for job in data['results']:
                contract_time = job.get('contract_time', '').lower()  # Pobieramy typ umowy i zamieniamy na małe litery

                if contract_time == 'full_time':
                    full_time_count += 1
                elif contract_time == 'part_time':
                    part_time_count += 1
                else:
                    unknown_count += 1  # Oferty, które nie mają danych o etacie lub mają inne wartości

            # Wyświetlenie zgromadzonych danych w konsoli
            '''print(f"Liczba ofert pełnoetatowych: {full_time_count}")
            print(f"Liczba ofert część etatowych: {part_time_count}")
            print(f"Liczba ofert bez informacji o etacie: {unknown_count}")'''

            # Przygotowanie listy do wykresu, usuwamy kategorie z liczbą ofert równą 0
            labels = []
            sizes = []

            if full_time_count > 0:
                labels.append('Full-Time')
                sizes.append(full_time_count)
            if part_time_count > 0:
                labels.append('Part-Time')
                sizes.append(part_time_count)
            if unknown_count > 0:
                labels.append('Brak danych')
                sizes.append(unknown_count)

            # Tworzymy wykres tylko jeśli są dostępne jakiekolwiek oferty
            if sizes:
                plt.figure(figsize=(8, 6))
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                        colors=['#66b3ff', '#99ff99', '#ffcc99'])  # Pogrubienie granic i zmiana koloru granicy)
                plt.title('Procent ofert pracy ze względu na etat')
                plt.axis('equal')  # Upewniamy się, że wykres jest okrągły

                # Wyświetlenie wykresu
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                uri_5 = 'data:image/png;base64,' + urllib.parse.quote(string)

        except:
            uri_5 = ''

        #WYKRES 6

        try:

            data = response.json()

            # Zliczamy oferty pracy na pełny etat, część etatowe oraz brak danych
            contract_count = 0
            permanent_count = 0
            unknown_count = 0  # Liczba ofert, dla których nie ma informacji o etacie

            for job in data['results']:
                contract_type = job.get('contract_type', '').lower()  # Pobieramy typ umowy i zamieniamy na małe litery

                if contract_type == 'contract':
                    contract_count += 1
                elif contract_type == 'permanent':
                    permanent_count += 1
                else:
                    unknown_count += 1  # Oferty, które nie mają danych o etacie lub mają inne wartości

            # Wyświetlenie zgromadzonych danych w konsoli
            '''print(f"Liczba ofert stałych: {permanent_count}")
            print(f"Liczba ofert na umowę: {contract_count}")
            print(f"Liczba ofert bez informacji o typie umowy: {unknown_count}")'''

            # Przygotowanie listy do wykresu, usuwamy kategorie z liczbą ofert równą 0
            labels = []
            sizes = []

            if permanent_count > 0:
                labels.append('Permanent')
                sizes.append(permanent_count)
            if contract_count > 0:
                labels.append('Contract')
                sizes.append(contract_count)
            if unknown_count > 0:
                labels.append('Brak danych')
                sizes.append(unknown_count)

            # Tworzymy wykres tylko jeśli są dostępne jakiekolwiek oferty
            if sizes:
                plt.figure(figsize=(8, 6))
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                        colors=['#F3B06A', '#FDF041', '#FD4B41'])  # Pogrubienie granic i zmiana koloru granicy)
                plt.title('Procent ofert pracy ze względu na typ umowy')
                plt.axis('equal')  # Upewniamy się, że wykres jest okrągły

                # Dostosowanie przestrzeni wokół wykresu (przesunięcie wykresu w dół)
                plt.subplots_adjust(top=1.25)  # Zwiększenie odstępu na górze

                # Wyświetlenie wykresu
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                string = base64.b64encode(buf.read())
                uri_6 = 'data:image/png;base64,' + urllib.parse.quote(string)

        except:
            uri_6 = ''

    else:
        print("Brak wyników dla zapytania.")

    uri_tab = [uri_1, uri_2, uri_3, uri_4, uri_5, uri_6]

    return uri_tab

def create_plot_earnings(url):
    # Wyślij zapytanie HTTP
    response = requests.get(url)

    data = response.json()
    # print(data['month'])
    data = data['month']

    dwuwymiarowa_tablica = [[klucz, wartosc] for klucz, wartosc in data.items()]  # Wyświetlenie wyniku

    for i in range(len(dwuwymiarowa_tablica)):
        dwuwymiarowa_tablica[i][0] = datetime.strptime(dwuwymiarowa_tablica[i][0], '%Y-%m')
        # print(i, dwuwymiarowa_tablica[i][0])

    posortowana_tablica = sorted(dwuwymiarowa_tablica, key=lambda x: x[0], reverse=False)

    '''for i in range(len(posortowana_tablica)):
        print(i + 1, posortowana_tablica[i][0], posortowana_tablica[i][1])'''

        # WYKRESSSSSSS------------------------------------------------------------------

    # Przygotowanie danych do wykresu
    miesiace = [item[0] for item in posortowana_tablica]
    salary = [item[1] for item in posortowana_tablica]

    # Przygotowanie danych do regresji
    # Przekształcamy daty na liczby (miesiące od początku)
    dates_as_numbers = np.array([(date - min(miesiace)).days // 30 for date in miesiace])

    # Dodanie stałej (intercept) do danych (ważne dla regresji)
    X = sm.add_constant(dates_as_numbers)
    y = salary

    # Dopasowanie modelu regresji liniowej
    model = sm.OLS(y, X)
    results = model.fit()

    # Przewidywanie wartości
    predictions = results.predict(X)

    # Przewidywanie wartości na 3 przyszłe miesiące
    # Załóżmy, że ostatnia data to ostatni miesiąc w Twoich danych
    last_month = miesiace[-1]
    future_months = [last_month + relativedelta(months=i) for i in range(1, 4)]  # Dodać 3 miesiące

    # Tworzymy nowe dane (liczby miesięcy) do predykcji
    future_months_as_numbers = np.array([(date - min(miesiace)).days // 30 for date in future_months])
    X_future = sm.add_constant(future_months_as_numbers)

    # Przewidywanie na przyszłość
    future_predictions = results.predict(X_future)

    # Dodajemy te przewidywania do listy miesięcy i wynagrodzeń
    miesiace_future = miesiace + future_months
    salary_future = salary + list(future_predictions)

    # Tworzenie wykresu
    plt.figure(figsize=(10, 6))
    # plt.plot(miesiace, salary, marker='o', linestyle='-', color='teal')

    # Rysowanie wykresu z kolorami w zależności od wzrostu/spadku wynagrodzenia
    for i in range(1, len(miesiace)):
        # Jeżeli wynagrodzenie wzrosło
        if salary[i] > salary[i - 1]:
            plt.plot(miesiace[i - 1:i + 1], salary[i - 1:i + 1], marker='o', color='green')  # Zielony dla wzrostu
        # Jeżeli wynagrodzenie spadło
        elif salary[i] < salary[i - 1]:
            plt.plot(miesiace[i - 1:i + 1], salary[i - 1:i + 1], marker='o', color='red')  # Czerwony dla spadku

    plt.plot(miesiace, predictions, linestyle='--', label='Regresja liniowa', color='grey')
    plt.plot(miesiace_future, salary_future, linestyle=':', label='Przewidywania (3 miesiące)', color='#F08080')

    # Pobranie współczynników regresji
    intercept = results.params[0]
    slope = results.params[1]

    # Dodanie równania na wykresie
    equation = f'y = {slope:.2f}x + {intercept:.2f}'  # Formatowanie do dwóch miejsc po przecinku
    plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='black')

    # Formatowanie wykresu
    plt.title('Wysokość wynagrodzeń w czasie z regresją liniową i przewidywaniami', fontsize=16)
    plt.xlabel('Miesiące', fontsize=12)
    plt.ylabel('Wysokość wynagrodzenia', fontsize=12)

    # Dostosowanie formatu daty na osi X, aby wyświetlały się po polsku
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    # Wyświetlenie wszystkich miesięcy na osi X, z rotacją etykiet
    all_months = miesiace + future_months
    plt.xticks(all_months, rotation=45)
    plt.grid(True)

    # Wyświetlanie wykresu
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    chart_1 = 'data:image/png;base64,' + urllib.parse.quote(string)

    return chart_1

def create_plot_offers(params):

    response_full = []

    ile_stron = 4
    j = 0

    for i in range(1, ile_stron + 1):
        j = j + i

        url = f'https://api.adzuna.com/v1/api/jobs/pl/search/{j}'

        response = requests.get(url, params=params)

        # print(response.status_code)

        response_json = response.json()

        response_full = response_full + response_json['results']

        j = j + 9

    # print(len(response_full))

    wystapienia = []

    for i in range(0, len(response_full)):
        month = int(response_full[i]['created'][5:7])
        wystapienia.append(month)

    ilosc_ofert = []
    for i in range(1, 13):
        ilosc_w_miesiacu = []

        now = dt.datetime.now()
        rok = now.year
        miesiac = now.month

        if i <= miesiac:
            data = dt.date(rok, i, 1)
        else:
            data = dt.date(rok - 1, i, 1)

        ilosc_w_miesiacu.append(data)

        ilosc = wystapienia.count(i)
        ilosc_w_miesiacu.append(ilosc)

        ilosc_ofert.append(ilosc_w_miesiacu)

    '''for i in range(len(ilosc_ofert)):
        print(ilosc_ofert[i][0], ilosc_ofert[i][1])'''

    # ile_stron - ile stron po 50 ofert (jesli nie wychodzi poza ilosc dostepnych danych)
    # ilosc_ofert - pierwsza kolumna: miesiąc, druga kolumna: ilość dodanych ofert

    # ------------WYKRESSSSSSS--------------------------------------
    # Tworzymy DataFrame dla łatwiejszej analizy
    df = pd.DataFrame(ilosc_ofert, columns=["Miesiąc", "Liczba ofert"])

    # Używamy pd.to_datetime, aby upewnić się, że mamy odpowiedni typ daty
    df["Miesiąc"] = pd.to_datetime(df["Miesiąc"])
    # Sortowanie po kolumnie "Miesiąc"
    df = df.sort_values("Miesiąc")

    # Formatowanie daty na osi X w formacie 'MMM YYYY' (np. Jan 2024)
    df["Miesiąc"] = df["Miesiąc"].dt.strftime('%b %Y')

    # Tworzymy wykres
    plt.figure(figsize=(10, 6))
    plt.plot(df["Miesiąc"], df["Liczba ofert"], marker='o', linestyle='-', color='#ff78dc')

    # Tytuł i etykiety
    plt.title('Liczba dodanych ofert pracy na dany zawód w czasie', fontsize=16)
    plt.xlabel('Miesiąc', fontsize=12)
    plt.ylabel('Liczba ofert', fontsize=12)

    # Ustawienie etykiet osi X jako skróty miesięcy i roku
    plt.xticks(rotation=45)

    # Włączanie siatki i dostosowanie układu
    plt.grid(True)
    plt.tight_layout()

    # Wyświetlenie wykresu
    #plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri_1 = 'data:image/png;base64,' + urllib.parse.quote(string)

    # --------------------------------------

    response_full = []

    ile_stron = 4
    j = 0

    for i in range(1, ile_stron + 1):
        j = j + i

        url = f'https://api.adzuna.com/v1/api/jobs/pl/search/{j}'

        response = requests.get(url, params=params)

        # print(response.status_code)

        response_json = response.json()

        response_full = response_full + response_json['results']

        j = j + 9

    # print(len(response_full))

    now = dt.datetime.now()
    miesiac = now.month
    rok = now.year

    if miesiac == 1:
        miesiac_min_1 = 12
        rok_min_1 = rok - 1
    else:
        miesiac_min_1 = miesiac - 1
        rok_min_1 = rok

    wystapienia_miesiac = []
    wystapienia_miesiac_min_1 = []

    for i in range(0, len(response_full)):

        if int(response_full[i]['created'][5:7]) == miesiac:
            dzien = int(response_full[i]['created'][8:10])
            wystapienia_miesiac.append(dzien)

        if int(response_full[i]['created'][5:7]) == miesiac_min_1:
            dzien = int(response_full[i]['created'][8:10])
            wystapienia_miesiac_min_1.append(dzien)

    ilosc_ofert_miesiac = []
    for i in range(1, now.day + 1):
        ilosc_w_miesiacu = []

        data = dt.date(rok, miesiac, i)
        ilosc_w_miesiacu.append(data)

        ilosc = wystapienia_miesiac.count(i)
        ilosc_w_miesiacu.append(ilosc)

        ilosc_ofert_miesiac.append(ilosc_w_miesiacu)

    ilosc_ofert_miesiac_min_1 = []
    for i in range(1, calendar.monthrange(now.year, miesiac_min_1)[1] + 1):
        ilosc_w_miesiacu = []

        data = dt.date(rok_min_1, miesiac_min_1, i)
        ilosc_w_miesiacu.append(data)

        ilosc = wystapienia_miesiac_min_1.count(i)
        ilosc_w_miesiacu.append(ilosc)

        ilosc_ofert_miesiac_min_1.append(ilosc_w_miesiacu)

    ilosc_ofert = ilosc_ofert_miesiac_min_1 + ilosc_ofert_miesiac

    '''for i in range(len(ilosc_ofert)):
        print(ilosc_ofert[i][0], ilosc_ofert[i][1])'''

    # ------------WYKRESSSSSSS--------------------------------------
    # Tworzymy DataFrame dla łatwiejszej analizy
    df = pd.DataFrame(ilosc_ofert, columns=["Dzień", "Liczba ofert"])

    # Używamy pd.to_datetime, aby upewnić się, że mamy odpowiedni typ daty
    df["Dzień"] = pd.to_datetime(df["Dzień"])

    # Formatowanie daty na osi X w formacie 'MMM YYYY' (np. Jan 2024)
    df["Dzień"] = df["Dzień"].dt.strftime('%d/%m/%Y')

    # Tworzymy wykres
    plt.figure(figsize=(10, 6))
    plt.plot(df["Dzień"], df["Liczba ofert"], marker='o', linestyle='-', color='#9de0ad')

    # Tytuł i etykiety
    plt.title('Liczba dodanych ofert pracy na dany zawód dziennie', fontsize=16)
    plt.xlabel('Dzień', fontsize=12)
    plt.ylabel('Liczba ofert', fontsize=12)

    # Ustawienie etykiet osi X co 5 dat:
    # Obliczamy co 5 datę i ustawiamy etykiety ręcznie
    step = 5
    tick_indices = list(range(0, len(df), step))  # Indeksy co 5 datę
    tick_labels = df["Dzień"].iloc[tick_indices]  # Etykiety odpowiadające tym indeksom
    plt.xticks(tick_indices, tick_labels, rotation=45)

    # Ustawienie wartości osi Y co 5:
    y_ticks = range(0, max(df["Liczba ofert"]) + 5, 5)
    plt.yticks(y_ticks)

    # Włączanie siatki i dostosowanie układu
    plt.grid(True)
    plt.tight_layout()

    # Wyświetlenie wykresu
    #plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri_2 = 'data:image/png;base64,' + urllib.parse.quote(string)


    uri_tab = [uri_1, uri_2]

    return uri_tab

def analiza_opisowa_dane(url, params):

    response = requests.get(url, params=params)

    # print(response.status_code)

    response_json = response.json()  # Zamiana z JSON na słownik Python

    dane_tab = {}

    # Sprawdzenie, czy w odpowiedzi są wyniki
    if 'results' in response_json and response_json['results']:
        jobs = []  # Lista, w której będziemy przechowywać oferty pracy
        region_count = defaultdict(int)  # Słownik do zliczania ofert w województwach
        # salary_data = []  # Lista, w której będziemy przechowywać dane o wynagrodzeniu
        company_count = defaultdict(int)  # Słownik do zliczania ofert w firmach
        job_count = defaultdict(int)  # Słownik do zliczania ofert według stanowisk
        salary_data = defaultdict(int)  # Słownik: klucz=(company_name, salary_min, salary_max), wartość=liczba ofert

        # Pętla po wynikach
        for i in response_json['results']:
            job = []

            # Zabezpieczenie przy pobieraniu danych
            job_title = i.get('title', 'Brak tytułu')
            company_name = i.get('company', {}).get('display_name', 'Brak firmy')
            location = i.get('location', {}).get('display_name', 'Brak lokalizacji')
            description = i.get('description', 'Brak opisu')

            salary_min = i.get('salary_min', None)  # Minimalne wynagrodzenie
            salary_max = i.get('salary_max', None)  # Maksymalne wynagrodzenie

            if salary_min is not None and salary_max is not None:
                # Dodawanie unikalnych kombinacji firmy i widełek wynagrodzenia
                salary_data[(company_name, salary_min, salary_max)] += 1

            # Przypisanie województwa
            region = location.split(",")[-1].strip()  # Zakłada się, że województwo jest na końcu lokalizacji

            # Zliczanie ofert pracy w województwie
            region_count[region] += 1

            # Zliczanie ofert pracy w firmach
            company_count[company_name] += 1

            # Zliczanie ofert pracy według stanowisk
            job_count[job_title] += 1

            # Dodanie oferty do listy jobs
            job.append(job_title)
            job.append(company_name)
            job.append(location)
            job.append(description)

            # Dodanie do ogólnej listy 'jobs'
            jobs.append(job)

        # Analiza liczby ofert wg regionów
        sorted_regions = sorted(region_count.items(), key=lambda x: x[1], reverse=True)
        regions, countss = zip(*sorted_regions)
        slowo = "województwo "
        top_region = regions[0]
        if top_region != "Polska":
            top_region = slowo + top_region
        top_region_counts = countss[0]
        sum_counts_regions = sum(countss)
        percent_counts_regions = (top_region_counts / sum_counts_regions) * 100
        if countss[0] == 1:
            percent_counts_regions = "Stanowi ona " + str(percent_counts_regions)
        else:
            percent_counts_regions = "Stanowią one " + str(percent_counts_regions)


        dane_tab['top_region'] = top_region
        dane_tab['percent_counts_regions'] = percent_counts_regions
        dane_tab['top_region_counts'] = top_region_counts

        # Analiza liczby ofert wg firm
        sorted_companies = sorted(company_count.items(), key=lambda x: x[1], reverse=True)
        companies, counts = zip(*sorted_companies)
        top_company = companies[0]
        top_company_counts = counts[0]
        sum_counts = sum(counts)
        percent_counts = round((top_company_counts / sum_counts) * 100, 2)
        if counts[0] == 1:
            percent_counts = "Stanowi ona " + str(percent_counts)
        else:
            percent_counts = "Stanowią one " + str(percent_counts)


        dane_tab['top_company'] = top_company
        dane_tab['top_company_counts'] = top_company_counts
        dane_tab['percent_counts'] = percent_counts
        dane_tab['sum_counts'] = sum_counts

        if salary_data:
            # Analiza zarobkow wg firm
            companies, min_salaries, max_salaries = zip(*salary_data)

            # Szukamy min i max płac
            max_min_salary = max(min_salaries)  # Najwyższa minimalna pensja
            min_min_salary = min(min_salaries)
            max_max_salary = max(max_salaries)
            min_max_salary = min(max_salaries)
            index_of_max_min_salary = min_salaries.index(max_min_salary)  # Indeks tej pensji
            index_of_max_max_salary = max_salaries.index(max_max_salary)
            index_of_min_min_salary = min_salaries.index(min_min_salary)
            index_of_min_max_salary = max_salaries.index(min_max_salary)
            # Ile razy coś jest większe
            ile_razy_min_salary = round(max_min_salary / min_min_salary, 1)
            ile_razy_max_salary = round(max_max_salary / min_max_salary, 1)
            slowoo = "również "
            # Szukamy firm najlepszych
            company_with_max_min_salary = companies[index_of_max_min_salary]
            company_with_max_max_salary = companies[index_of_max_max_salary]
            company_with_min_min_salary = companies[index_of_min_min_salary]
            company_with_min_max_salary = companies[index_of_min_max_salary]
            if company_with_max_max_salary == company_with_max_min_salary:
                company_with_max_max_salary = slowoo + company_with_max_max_salary

            dane_tab['company_with_max_min_salary'] = company_with_max_min_salary
            dane_tab['company_with_max_max_salary'] = company_with_max_max_salary
            dane_tab['company_with_min_min_salary'] = company_with_min_min_salary
            dane_tab['company_with_min_max_salary'] = company_with_min_max_salary
            dane_tab['max_min_salary'] = max_min_salary
            dane_tab['max_max_salary'] = max_max_salary
            dane_tab['ile_razy_min_salary'] = ile_razy_min_salary
            dane_tab['ile_razy_max_salary'] = ile_razy_max_salary


        else:
            print("Brak danych o wynagrodzeniach")

        # Sortowanie stanowisk według liczby ofert
        sorted_jobs = sorted(job_count.items(), key=lambda x: x[1], reverse=True)

        # Tworzenie wykresu słupkowego poziomego
        job_titles, countsss = zip(*sorted_jobs)  # Rozpakowanie krotek do dwóch list (stanowiska i liczba ofert)
        sum_counts_jobs = sum(countsss)
        max_counts_jobs = countsss[0]
        job_with_max_offers = job_titles[0]
        percent_counts_offers = round((max_counts_jobs / sum_counts_jobs) * 100, 2)
        if countsss[0] == 1:
            percent_counts_offers = "Stanowi ona " + str(percent_counts_offers)
        else:
            percent_counts_offers = "Stanowią one " + str(percent_counts_offers)

        dane_tab['percent_counts_offers'] = percent_counts_offers
        dane_tab['job_with_max_offers'] = job_with_max_offers
        dane_tab['max_counts_jobs'] = max_counts_jobs

    else:
        print("Brak wyników dla zapytania.")

    response_full = []

    ile_stron = 4
    j = 0

    for i in range(1, ile_stron + 1):
        j = j + i

        url = f'https://api.adzuna.com/v1/api/jobs/pl/search/{j}'

        response = requests.get(url, params=params)

        # print(response.status_code)

        response_json = response.json()

        response_full = response_full + response_json['results']

        j = j + 9

        # print(len(response_full))

    wystapienia = []

    for i in range(0, len(response_full)):
        month = int(response_full[i]['created'][5:7])
        wystapienia.append(month)

    ilosc_ofert = []
    for i in range(1, 13):
        ilosc_w_miesiacu = []

        now = dt.datetime.now()
        rok = now.year
        miesiac = now.month
        if i <= miesiac:
            data = dt.date(rok, i, 1)
        else:
            data = dt.date(rok - 1, i, 1)

        ilosc_w_miesiacu.append(data)

        ilosc = wystapienia.count(i)
        ilosc_w_miesiacu.append(ilosc)

        ilosc_ofert.append(ilosc_w_miesiacu)

    # Słownik mapujący numer miesiąca na pełną nazwę miesiąca po polsku
    miesiace = {
        1: "styczeń", 2: "luty", 3: "marzec", 4: "kwiecień", 5: "maj", 6: "czerwiec",
        7: "lipiec", 8: "sierpień", 9: "wrzesień", 10: "październik", 11: "listopad", 12: "grudzień"
    }

    # Funkcja zamieniająca numer miesiąca na polską nazwę miesiąca
    def get_polish_month(month_number):
        return miesiace.get(month_number, "Nieznany miesiąc")

    max_oferta = max(ilosc_ofert, key=lambda x: x[1])  # Znajdź miesiąc z maksymalną liczbą ofert
    max_miesiac = max_oferta[0]  # Miesiąc z największą liczbą ofert

    max_liczba_ofert = max_oferta[1]  # Liczba ofert w tym miesiącu

    max_miesiac = max_oferta[0].month  # Wyciągamy numer miesiąca z daty

    # Teraz używamy funkcji get_polish_month, aby zamienić numer miesiąca na nazwę
    polski_miesiac = get_polish_month(max_miesiac)
    rok_do_miesiaca = max_oferta[0].year

    # Zamiana numeru miesiąca na nazwę w języku polskim

    dane_tab['max_liczba_ofert'] = max_liczba_ofert
    dane_tab['polski_miesiac'] = polski_miesiac
    dane_tab['rok_do_miesiaca'] = rok_do_miesiaca

    return dane_tab

# Views
def home(request):
    return render(request, "Home.html")

def oferty_pracy_disp(request):
    # id i klucze
    API_ID = 'd85d9e1a'
    API_KEY = 'c36f6bf6947de94278ed9f035eddfc8e'

    # Ustawienia zapytania
    global nr_strony, nazwa_miasta, nazwa_zawodu, odleglosc, minimalna

    #url = f'https://api.adzuna.com/v1/api/jobs/pl/search/{nr_strony}'
    params = {
        'app_id': API_ID,
        'app_key': API_KEY,
        'results_per_page': 15,
        'what': nazwa_zawodu,  # przykładowe wyszukiwanie
        'where': nazwa_miasta,
        #'distance': odleglosc,
        'salary_min': minimalna,
    }

    if request.method == 'POST':

        nastepny_value = request.POST.get('nastepny')
        if nastepny_value:
            nr_strony = nr_strony + int(nastepny_value)
            #print(nr_strony)

        poprzedni_value = request.POST.get('poprzedni')
        if poprzedni_value and nr_strony > 1:
            nr_strony = nr_strony - int(poprzedni_value)
            #print(nr_strony)

        wyczysc_filtr = request.POST.get('wyczysc_filtr')
        if wyczysc_filtr:
            nazwa_miasta = ''
            nazwa_zawodu = ''
            #odleglosc = ''
            minimalna = 0

            params['where'] = str(nazwa_miasta)
            params['what'] = str(nazwa_zawodu)
            #params['distance'] = odleglosc
            params['salary_min'] = minimalna
            nr_strony = 1

        form = MyForm(request.POST)
        if form.is_valid():

            #miasto
            if form.cleaned_data['miasto']:
                nazwa_miasta = form.cleaned_data['miasto']
                if nazwa_miasta == 'empty_string': nazwa_miasta = ''
                params['where'] = str(nazwa_miasta)
                nr_strony = 1

            #zawod
            if form.cleaned_data['zawod']:
                nazwa_zawodu = form.cleaned_data['zawod']
                if nazwa_zawodu == 'empty_string': nazwa_zawodu = ''
                params['what'] = str(nazwa_zawodu)
                nr_strony = 1

            #odleglosc
            '''if form.cleaned_data['odleglosc_od']:
                odleglosc = form.cleaned_data['odleglosc_od']
                if odleglosc == 'empty_string': odleglosc = ''
                params['distance'] = str(odleglosc)
                nr_strony = 1'''

            #minimalna pensja
            if form.cleaned_data['minimalna_pensja']:
                minimalna = form.cleaned_data['minimalna_pensja']
                if minimalna == 'empty_string': minimalna = ''
                params['salary_min'] = minimalna
                nr_strony = 1



    url = f'https://api.adzuna.com/v1/api/jobs/pl/search/{nr_strony}'

    form = MyForm()

    time.sleep(1)

    #print(params)

    try:
        jobs = jobs_get(url, params)
    except:
        jobs = []

    return render(request, "Oferty_pracy.html", {'jobs': jobs, 'form': form})

def analiza_wykresy(request):

    # Adzuna - id i klucze
    API_ID = 'd85d9e1a'
    API_KEY = 'c36f6bf6947de94278ed9f035eddfc8e'

    # Ustawienia zapytania
    url = 'https://api.adzuna.com/v1/api/jobs/pl/search/1'
    params = {
        'app_id': API_ID,
        'app_key': API_KEY,
        'results_per_page': 300,
        'what': '',  # przykładowe wyszukiwanie
        'where': ''  # Możesz określić lokalizację, jeśli chcesz
    }

    if request.method == 'POST':
        form = MyForm_1(request.POST)
        if form.is_valid():

            #if form.cleaned_data['miasto_wykres']:
            nazwa_miasta_wykres = form.cleaned_data['miasto_wykres']
            #print(nazwa_miasta_wykres)
            params['where'] = str(nazwa_miasta_wykres)

            #if form.cleaned_data['zawod_wykres']:
            nazwa_zawodu_wykres = form.cleaned_data['zawod_wykres']
            params['what'] = str(nazwa_zawodu_wykres)

            wyczysc_filtr = request.POST.get('wyczysc_filtr')
            if wyczysc_filtr:
                nazwa_miasta_wykres = ''
                nazwa_zawodu_wykres = ''

                params['where'] = str(nazwa_miasta_wykres)
                params['what'] = str(nazwa_zawodu_wykres)

                # Wysłanie zapytania GET do API

    #print(params)

    form = MyForm_1()

    try:
        chart_tab = create_plot(url, params)

        chart_1 = chart_tab[0]
        chart_2 = chart_tab[1]
        chart_3 = chart_tab[2]
        chart_4 = chart_tab[3]
        chart_5 = chart_tab[4]
        chart_6 = chart_tab[5]

    except:
        return render(request, "error.html", {'form': form})


    return render(request, "Analiza_ofert.html", {'form': form, 'chart_1': chart_1, 'chart_2': chart_2, 'chart_3': chart_3, 'chart_4': chart_4, 'chart_5': chart_5, 'chart_6': chart_6})

def analiza_pensje(request):

    API_ID = 'd85d9e1a'
    API_KEY = 'c36f6bf6947de94278ed9f035eddfc8e'
    months_back = 12
    country = 'pl'

    if request.method == 'POST':
        form = MyForm_2(request.POST)
        if form.is_valid():

            #if form.cleaned_data['miasto_wykres']:
            skrot_kraju = form.cleaned_data['skrot_kraju']
            #print(skrot_kraju)
            country = skrot_kraju.lower()

            wyczysc_filtr = request.POST.get('wyczysc_filtr')
            if wyczysc_filtr:
                country = 'pl'


    # Ustawienia zapytania
    url = f'https://api.adzuna.com/v1/api/jobs/{country}/history?app_id=d85d9e1a&app_key=c36f6bf6947de94278ed9f035eddfc8e&months={months_back}'  # numer na koncu to strona

    form = MyForm_2()
    try:
        chart_1 = create_plot_earnings(url)
    except:
        return render(request, "error.html", {'form': form})

    return render(request, "Analiza_pensji.html", {'form': form, 'chart_1': chart_1})

def analiza_ilosc(request):

    # id i klucze
    API_ID = 'd85d9e1a'
    API_KEY = 'c36f6bf6947de94278ed9f035eddfc8e'

    # Ustawienia zapytania
    params = {
        'app_id': API_ID,
        'app_key': API_KEY,
        'results_per_page': 50,
        'what': 'software developer',  # przykładowe wyszukiwanie
        # 'where': 'Warsaw'
    }

    if request.method == 'POST':
        form = MyForm_1(request.POST)
        if form.is_valid():

            #if form.cleaned_data['zawod_wykres']:
            nazwa_zawodu_wykres = form.cleaned_data['zawod_wykres']
            params['what'] = str(nazwa_zawodu_wykres)

            wyczysc_filtr = request.POST.get('wyczysc_filtr')
            if wyczysc_filtr:
                nazwa_zawodu_wykres = ''

                params['what'] = str(nazwa_zawodu_wykres)

    form = MyForm_3()
    try:
        chart_tab = create_plot_offers(params)
        chart_1 = chart_tab[0]
        chart_2 = chart_tab[1]
    except:
        return render(request, "error.html", {'form': form})

    return render(request, "Analiza_ilosci_ofert.html", {'form': form, 'chart_1': chart_1, 'chart_2': chart_2})

def analiza_opisowa(request):

    # Adzuna - id i klucze
    API_ID = 'd85d9e1a'
    API_KEY = 'c36f6bf6947de94278ed9f035eddfc8e'

    # Ustawienia zapytania
    url = 'https://api.adzuna.com/v1/api/jobs/pl/search/1'
    params = {
        'app_id': API_ID,
        'app_key': API_KEY,
        'results_per_page': 300,
        'what': '',  # przykładowe wyszukiwanie
        'where': ''  # Możesz określić lokalizację, jeśli chcesz
    }

    if request.method == 'POST':
        form = MyForm_4(request.POST)
        if form.is_valid():

            # if form.cleaned_data['miasto_wykres']:
            nazwa_miasta_wykres = form.cleaned_data['miasto_wykres']
            # print(nazwa_miasta_wykres)
            params['where'] = str(nazwa_miasta_wykres)

            # if form.cleaned_data['zawod_wykres']:
            nazwa_zawodu_wykres = form.cleaned_data['zawod_wykres']
            params['what'] = str(nazwa_zawodu_wykres)

            wyczysc_filtr = request.POST.get('wyczysc_filtr')
            if wyczysc_filtr:
                nazwa_miasta_wykres = ''
                nazwa_zawodu_wykres = ''

                params['where'] = str(nazwa_miasta_wykres)
                params['what'] = str(nazwa_zawodu_wykres)

                # Wysłanie zapytania GET do API

    form = MyForm_4()

    try:
        zawartosc = analiza_opisowa_dane(url, params)
        zawartosc['form'] = form
    except:
        return render(request, "error.html", {'form': form})

    return render(request, "Analiza_opisowa.html", zawartosc)

def analiza_ilosc_2(request):

    def generate_dates(n, start_date, increment):
        dates = [start_date + dt.timedelta(days=(i // increment)) for i in range(n)]
        return dates

    file_path = "Job_analizer/Jobs_data.xlsx"
    saved_data = pd.read_excel(file_path, sheet_name='Sheet1')

    # Data początkowa
    start_date = datetime(2024, 12, 15)

    # Funkcja do generowania dat z przyrostem co 200 wierszy

    # Dodanie nowej kolumny do DataFrame
    saved_data['Data_pobrania'] = generate_dates(len(saved_data), start_date, 200)

    kategoria_col = np.array([])

    for i in range(len(saved_data)):
        testowe_dane = saved_data['category'][i]
        testowe_dane = testowe_dane.replace("'", '"')
        testowe_dane = json.loads(testowe_dane)['label']

        kategoria_col = np.append(kategoria_col, testowe_dane)

    saved_data['kategoria'] = kategoria_col

    kategoria_data = saved_data[['kategoria', 'Data_pobrania']]

    kategoria_data_filtered = kategoria_data[kategoria_data['kategoria'] != 'Unknown']
    kategoria_data_filtered = kategoria_data_filtered[kategoria_data_filtered['kategoria'] != 'Inna/ogólna']

    top_kategorie = ['Sprzedaż', 'Produkcja', 'Inżynieria']

    kategoria_data_filtered_top = kategoria_data_filtered[kategoria_data_filtered['kategoria'].isin(top_kategorie)]

    kategoria_data_grouped = kategoria_data_filtered_top.groupby(['Data_pobrania', 'kategoria']).size().unstack(
        fill_value=0)

    kategoria_data_grouped.index = pd.to_datetime(kategoria_data_grouped.index)
    colors = ['#37B0A1', '#4a209b', '#8b0000']
    fig, ax = plt.subplots(figsize=(10, 6))
    kategoria_data_grouped.plot(kind='line', marker='o', color=colors, ax=ax)
    ax.set_title('Liczba ofert w czasie dla 3 najczęściej występujących kategorii ofert pracy', fontsize=16)
    ax.set_xlabel('Grudzień 2024 - Styczeń 2025', fontsize=12)
    ax.set_ylabel('Liczba ofert', fontsize=12)
    plt.tight_layout()
    ax.legend(title="Kategoria", fontsize=12)
    ax.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    chart = 'data:image/png;base64,' + urllib.parse.quote(string)

    return render(request, "Analiza_ilosci_ofert_w_czasie.html", {'chart': chart})


