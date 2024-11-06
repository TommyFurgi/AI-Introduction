# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Klasyfikacja niezbalansowana, klasyfikatory zespołowe i wyjaśnialna AI

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Wykorzystanie Google Colab
#
# Jeśli korzystasz z Google Colab skopiuj plik `feature_names.json` do katalogu głównego projektu.
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apohllo/sztuczna-inteligencja/blob/master/lab2/lab_2.ipynb)

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Ładowanie i eksploracja danych

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# Na tym laboratorium wykorzystamy zbiór danych [Polish companies bankruptcy](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data). Dotyczy on klasyfikacji, na podstawie danych z raportów finansowych, czy firma zbankrutuje w ciągu najbliższych kilku lat. Jest to zadanie szczególnie istotne dla banków, funduszy inwestycyjnych, firm ubezpieczeniowych itp., które z tego powodu zatrudniają licznie data scientistów. Zbiór zawiera 64 cechy, obliczone przez ekonomistów, którzy stworzyli ten zbiór, są one opisane na podlinkowanej wcześniej stronie. Dotyczą one zysków, posiadanych zasobów oraz długów firm.
#
# Ściągnij i rozpakuj dane (`Data Folder` -> `data.zip`) do katalogu `data` obok tego notebooka. Znajduje się tam 5 plików w formacie `.arff`, wykorzystywanym głównie przez oprogramowanie Weka. Jest to program do "klikania" ML w interfejsie graficznym, jakiś czas temu popularny wśród mniej technicznych data scientistów. W Pythonie ładuje się je za pomocą bibliotek SciPy i Pandas.
#

# %% [markdown]
# Jeśli korzystasz z Linuksa możesz skorzystać z poniższych poleceń do pobrania i rozpakowania tych plików.

# %% editable=true slideshow={"slide_type": ""}
# !mkdir -p data
# !wget https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip -O data/data.zip

# %%
# !unzip data/data.zip -d data

# %% [markdown] editable=true slideshow={"slide_type": ""}
#
# W dalszej części laboratorium wykorzystamy plik `3year.arff`, w którym na podstawie finansowych firmy po 3 latach monitorowania chcemy przewidywać, czy firma zbankrutuje w ciągu najbliższych 3 lat. Jest to dość realistyczny horyzont czasowy.
#
# Dodatkowo w pliku `feature_names.json` znajdują się nazwy cech. Są bardzo długie, więc póki co nie będziemy z nich korzystać.

# %% editable=true pycharm={"name": "#%%\n"} slideshow={"slide_type": ""}
import json
import os

from scipy.io import arff
import pandas as pd

data = arff.loadarff(os.path.join("data", "3year.arff"))

with open("feature_names.json") as file:
    feature_names = json.load(file)

X = pd.DataFrame(data[0])

# %% [markdown] pycharm={"name": "#%% md\n"}
# Przyjrzyjmy się teraz naszym danym.

# %% editable=true pycharm={"name": "#%%\n"} slideshow={"slide_type": ""}
X.head()

# %% editable=true pycharm={"name": "#%%\n"} slideshow={"slide_type": ""}
X.dtypes

# %% pycharm={"name": "#%%\n"}
X.describe()

# %%
feature_names

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""} tags=["ex"]
# DataFrame zawiera 64 atrybuty numeryczne o zróżnicowanych rozkładach wartości oraz kolumnę `"class"` typu `bytes` z klasami 0 i 1. Wiemy, że mamy do czynienia z klasyfikacją binarną - klasa 0 to brak bankructwa, klasa 1 to bankructwo w ciągu najbliższych 3 lat. Przyjrzyjmy się dokładniej naszym danym.

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# ### Zadanie 1 (0.5 punktu)

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# 1. Wyodrębnij klasy jako osobną zmienną typu `pd.Series`, usuwając je z macierzy `X`. Przekonwertuj go na liczby całkowite.
# 2. Narysuj wykres słupkowy (bar plot) częstotliwości obu klas w całym zbiorze. Upewnij się, że na osi X są numery lub nazwy klas, a oś Y ma wartości w procentach.
#
# **Uwaga:** sugerowane jest użycie `if` w podpunkcie 1, żeby można było tę komórkę bezpiecznie odpalić kilka razy.

# %% editable=true pycharm={"name": "#%%\n"} slideshow={"slide_type": ""} tags=["ex"]
import matplotlib.pyplot as plt

if 'class' in X.columns:
    y = X.pop('class').apply(lambda x: int(x))  

classes = y.value_counts(normalize=True) * 100
classes = classes.reindex([0, 1], fill_value=0)

plt.figure(figsize=(8, 6))
bars =  plt.bar(classes.index, classes.values, color=['lightblue', 'darkgreen'])

plt.xlabel('Class')
plt.ylabel('Frequency (%)')
plt.title('Percentage distribution of classes (0 - No bankruptcy, 1 - Bankruptcy)')
plt.xticks(classes.index, ['No bankruptcy (0)', 'Bankruptcy (1)']) 

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

plt.show()

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# Jak widać, klasa pozytywna jest w znacznej mniejszości, stanowi poniżej 5% zbioru. Taki problem nazywamy **klasyfikacją niezbalansowaną (imbalanced classification)**. Mamy tu **klasę dominującą (majority class)** oraz **klasę mniejszościową (minority class)**. Pechowo prawie zawsze interesuje nas ta druga, bo klasa większościowa jest trywialna. Przykładowo, 99% badanych jest zdrowych, a 1% ma niewykryty nowotwór - z oczywistych przyczyn chcemy wykrywać właśnie sytuację rzadką (problem diagnozy jako klasyfikacji jest zasadniczo zawsze niezbalansowany). W dalszej części laboratorium poznamy szereg konsekwencji tego zjawiska i metody na radzenie sobie z nim.
#
# Mamy sporo cech, wszystkie numeryczne. Ciekawe, czy mają wartości brakujące, a jeśli tak, to ile. Policzymy to z pomocą biblioteki Pandas i metody `.isna()`. Domyślnie operuje ona na kolumnach, jak większość metod w Pandasie. Sumę wartości per kolumna zwróci nam metoda `.sum()`. Jeżeli podzielimy to przez liczbę wierszy `len(X)`, to otrzymamy ułamek wartości brakujących w każdej kolumnie.
#
# Pandas potrafi też stworzyć wykres, z pomocą funkcji np. `.plot.hist()` czy `.plot.bar()`. Przyjmują one opcje formatowania wykresu, z których korzysta pod spodem biblioteka matplotlib.

# %% editable=true pycharm={"name": "#%%\n"} slideshow={"slide_type": ""} tags=["ex"]
na_perc = X.isna().sum() / len(X)
na_perc.plot.bar(title="Fraction of missing values per column", figsize=(15, 5))

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""} tags=["ex"]
# Jak widać, cecha 37 ma bardzo dużo wartości brakujących, podczas gdy pozostałe cechy mają raczej niewielką ich liczbę. W takiej sytuacji najlepiej usunąć tę cechę, a pozostałe wartości brakujące **uzupełnić / imputować (impute)**. Typowo wykorzystuje się do tego wartość średnią lub medianę z danej kolumny. Ale uwaga - imputacji dokonuje się dopiero po podziale na zbiór treningowy i testowy! W przeciwnym wypadku wykorzystywalibyśmy dane ze zbioru testowego, co sztucznie zawyżyłoby wyniki. Jest to błąd metodologiczny - **wyciek danych (data leakage)**.
#
# Podział na zbiór treningowy i testowy to pierwszy moment, kiedy niezbalansowanie danych nam przeszkadza. Jeżeli zrobimy to czysto losowo, to są spore szanse, że w zbiorze testowym będzie tylko klasa negatywna - w końcu jest jej aż >95%. Dlatego wykorzystuje się **próbkowanie ze stratyfikacją (stratified sampling)**, dzięki któremu proporcje klas w zbiorze przed podziałem oraz obu zbiorach po podziale są takie same.

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# ### Zadanie 2 (0.75 punktu)

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# 1. Usuń kolumnę `"Attr37"` ze zbioru danych.
# 2. Dokonaj podziału zbioru na treningowy i testowy w proporcjach 80%-20%, z przemieszaniem (`shuffle`), ze stratyfikacją, wykorzystując funkcję `train_test_split` ze Scikit-learn'a.
# 3. Uzupełnij wartości brakujące średnią wartością cechy z pomocą klasy `SimpleImputer`.
#
# **Uwaga:**
# - jak wcześniej, sugerowane jest użycie `if` w podpunkcie 1
# - pamiętaj o uwzględnieniu stałego `random_state=0`, aby wyniki były **reprodukowalne (reproducible)**
# - `stratify` oczekuje wektora klas
# - wartości do imputacji trzeba wyestymować na zbiorze treningowym (`.fit()`), a potem zastosować te nauczone wartości na obu podzbiorach (treningowym i testowym)

# %% editable=true pycharm={"name": "#%%\n"} slideshow={"slide_type": ""} tags=["ex"]
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np

if 'Attr37' in X.columns:
    X.pop('Attr37')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, train_size=0.8, random_state=0, shuffle=True, stratify=y
)

inputer = SimpleImputer(missing_values=np.nan, strategy="mean")
inputer.fit(X_train)

X_train_imputed = inputer.transform(X_train) # transformacja
X_test_imputed = inputer.transform(X_test)

X_train = pd.DataFrame(X_train_imputed, columns=X.columns) # uzupełnianie danych
X_test = pd.DataFrame(X_test_imputed, columns=X.columns)

y_train = y_train.reset_index(drop=True)  
y_test = y_test.reset_index(drop=True) 


# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# ## Prosta klasyfikacja

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# Zanim przejdzie się do modeli bardziej złożonych, trzeba najpierw wypróbować coś prostego, żeby mieć punkt odniesienia. Tworzy się dlatego **modele bazowe (baselines)**.
#
# W naszym przypadku będzie to **drzewo decyzyjne (decision tree)**. Jest to drzewo binarne z decyzjami if-else, prowadzącymi do klasyfikacji danego przykładu w liściu. Każdy podział w drzewie to pytanie postaci "Czy wartość cechy X jest większa lub równa Y?". Trening takiego drzewa to prosty algorytm zachłanny, bardzo przypomina budowę zwykłego drzewa binarnego. W każdym węźle wykonujemy:
# 1. Sprawdź po kolei wszystkie możliwe punkty podziału, czyli każdą (unikalną) wartość każdej cechy, po kolei.
# 2. Dla każdego przypadku podziel zbiór na 2 kawałki: niespełniający warunku (lewe dziecko) i spełniający warunek (prawe dziecko).
# 3. Oblicz jakość podziału według pewnej wybranej funkcji jakości. Im lepiej nasz if/else rozdziela klasy od siebie (im "czystsze" są węzły-dzieci), tym wyższa jakość. Innymi słowy, chcemy, żeby do jednego dziecka poszła jedna klasa, a do drugiego druga.
# 4. Wybierz podział o najwyższej jakości.
#
# Taki algorytm wykonuje się rekurencyjnie, aż otrzymamy węzeł czysty (pure leaf), czyli taki, w którym są przykłady z tylko jednej klasy. Typowo wykorzystywaną funkcją jakości (kryterium podziału) jest entropia Shannona - im niższa entropia, tym bardziej jednolite są klasy w węźle (czyli wybieramy podział o najniższej entropii).
#
# Powyższe wytłumaczenie algorytmu jest oczywiście nieformalne i dość skrótowe. Doskonałe tłumaczenie, z interaktywnymi wizualizacjami, dostępne jest [tutaj](https://mlu-explain.github.io/decision-tree/). W formie filmów - [tutaj](https://www.youtube.com/watch?v=ZVR2Way4nwQ) oraz [tutaj](https://www.youtube.com/watch?v=_L39rN6gz7Y). Dla drzew do regresji - [ten film](https://www.youtube.com/watch?v=g9c66TUylZ4).
#
# <img src = https://miro.medium.com/max/1838/1*WyTsLwcAXivFCgNtF0OPqA.png width = "642" height = "451" >
#
# Warto zauważyć, że taka konstrukcja prowadzi zawsze do overfittingu. Otrzymanie liści czystych oznacza, że mamy 100% dokładności na zbiorze treningowym, czyli perfekcyjnie przeuczony klasyfikator. W związku z tym nasze predykcje mają bardzo niski bias, ale bardzo dużą wariancję. Pomimo tego drzewa potrafią dać bardzo przyzwoite wyniki, a w celu ich poprawy można je regularyzować, aby mieć mniej "rozrośnięte" drzewo. [Film dla zainteresowanych](https://www.youtube.com/watch?v=D0efHEJsfHo).
#
# W tym wypadku AI to naprawdę tylko zbiór if'ów ;)

# %% [markdown]
# Mając wytrenowany klasyfikator, trzeba oczywiście sprawdzić, jak dobrze on sobie radzi. Tu natrafiamy na kolejny problem z klasyfikacją niezbalansowaną - zwykła celność (accuracy) na pewno nie zadziała! Typowo wykorzystuje się AUC, nazywane też AUROC (Area Under Receiver Operating Characteristic), bo metryka ta "widzi" i uwzględnia niezbalansowanie klas. Wymaga ona przekazania prawdopodobieństwa klasy pozytywnej, a nie tylko binarnej decyzji.
#
# Bardzo dobre i bardziej szczegółowe wytłumaczenie, z interktywnymi wizualizacjami, można znaleć [tutaj](https://mlu-explain.github.io/roc-auc/). Dla preferujących filmy - [tutaj](https://www.youtube.com/watch?v=4jRBRDbJemM).
#
# Co ważne, z definicji AUROC, trzeba tam użyć prawdopodobieństw klasy pozytywnej (klasy 1). W Scikit-learn'ie zwraca je metoda `.predict_proba()`, która w kolejnych kolumnach zwraca prawdopodobieństwa poszczególnych klas.

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# ### Zadanie 3 (0.75 punktu)

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# 1. Wytrenuj klasyfikator drzewa decyzyjnego (klasa `DecisionTreeClassifier`). Użyj entropii jako kryterium podziału.
# 2. Oblicz i wypisz AUROC na zbiorze testowym dla drzewa decyzyjnego (funkcja `roc_auc_score`).
# 3. Skomentuj wynik - czy twoim zdaniem osiągnięty AUROC to dużo czy mało, biorąc pod uwagę możliwy zakres wartości tej metryki?
#
# **Uwaga:**
# - pamiętaj o użyciu stałego `random_state=0`
# - jeżeli drzewo nie wyświetli się samo, użyj `plt.show()` z Matplotliba
# - pamiętaj o tym, żeby przekazać do metryki AUROC prawdopodobieństwa klasy pozytywnej, a nie binarne predykcje!

# %% editable=true pycharm={"name": "#%%\n"} slideshow={"slide_type": ""} tags=["ex"]
from sklearn.tree import DecisionTreeClassifier, plot_tree

dtc = DecisionTreeClassifier(criterion='entropy', random_state=0)
dtc.fit(X_train, y_train)

plt.figure(figsize=(20, 10))  
plot_tree(dtc, filled=True, feature_names=X_train.columns, class_names=['Class 0', 'Class 1'])
plt.show()


# %% editable=true pycharm={"name": "#%%\n"} slideshow={"slide_type": ""} tags=["ex"]
from sklearn.metrics import roc_auc_score

prediction = dtc.predict_proba(X_test)
score = roc_auc_score(y_test, prediction[:,1])

print(f"AUROC: {score:.4f}")


# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""} tags=["ex"]
# AUROC przyjmuje wartości w przedziale od 0 do 1 oraz oczywiste jest żę, interesuje nas jak najwyższy wynik tej metryki. Otrzymana przez nas wartość jest jak najbardziej dobra, ale też mogła by być lepsza

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# ## Uczenie zespołowe, bagging, lasy losowe

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# Bardzo często wiele klasyfikatorów działających razem daje lepsze wyniki niż pojedynczy klasyfikator. Takie podejście nazywa się **uczeniem zespołowym (ensemble learning)**. Istnieje wiele różnych podejść do tworzenia takich klasyfikatorów złożonych (ensemble classifiers).
#
# Podstawową metodą jest **bagging**:
# 1. Wylosuj N (np. 100, 500, ...) próbek boostrapowych (boostrap sample) ze zbioru treningowego. Próbka boostrapowa to po prostu losowanie ze zwracaniem, gdzie dla wejściowego zbioru z M wierszami losujemy M próbek. Będą tam powtórzenia, średnio nawet 1/3, ale się tym nie przejmujemy.
# 2. Wytrenuj klasyfikator bazowy (base classifier) na każdej z próbek boostrapowych.
# 3. Stwórz klasyfikator złożony poprzez uśrednienie predykcji każdego z klasyfikatorów bazowych.
#
# <img src = https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Ensemble_Bagging.svg/440px-Ensemble_Bagging.svg.png width = "440" height = "248" >
#
# Typowo klasyfikatory bazowe są bardzo proste, żeby można było szybko wytrenować ich dużą liczbę. Prawie zawsze używa się do tego drzew decyzyjnych. Dla klasyfikacji uśrednienie wyników polega na głosowaniu - dla nowej próbki każdy klasyfikator bazowy ją klasyfikuje, sumuje się głosy na każdą klasę i zwraca najbardziej popularną decyzję.
#
# Taki sposób ensemblingu zmniejsza wariancję klasyfikatora. Intuicyjnie, skoro coś uśredniamy, to siłą rzeczy będzie mniej rozrzucone, bo dużo ciężej będzie osiągnąć jakąś skrajność. Redukuje to też overfitting.
#
# **Lasy losowe (Random Forests)** to ulepszenie baggingu. Zaobserwowano, że pomimo losowania próbek boostrapowych, w baggingu poszczególne drzewa są do siebie bardzo podobne (są skorelowane), używają podobnych cech ze zbioru. My natomiast chcemy zróżnicowania, żeby mieć niski bias - redukcją wariancji zajmuje się uśrednianie. Dlatego używa się metody losowej podprzestrzeni (random subspace method) - przy każdym podziale drzewa losuje się tylko pewien podzbiór cech, których możemy użyć do tego podziału. Typowo jest to pierwiastek kwadratowy z ogólnej liczby cech.
#
# Zarówno bagging, jak i lasy losowe mają dodatkowo bardzo przyjemną własność - są mało czułe na hiperparametry, szczególnie na liczbę drzew. W praktyce wystarczy ustawić 500 czy 1000 drzew i będzie dobrze działać. Dalsze dostrajanie hiperparametrów może jeszcze trochę poprawić wyniki, ale nie tak bardzo, jak przy innych klasyfikatorach. Jest to zatem doskonały wybór domyślny, kiedy nie wiemy, jakiego klasyfikatora użyć.
#
# Dodatkowo jest to problem **embarassingly parallel** - drzewa można trenować w 100% równolegle, dzięki czemu jest to dodatkowo wydajna obliczeniowo metoda.
#
# Głębsze wytłumaczenie, z interaktywnymi wizualizacjami, można znaleźć [tutaj](https://mlu-explain.github.io/random-forest/). Dobrze tłumaczy je też [ta seria filmów](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ&t=480s).

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Zadanie 4 (0.5 punktu)

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# 1. Wytrenuj klasyfikator Random Forest (klasa `RandomForestClassifier`). Użyj 500 drzew i entropii jako kryterium podziału.
# 2. Sprawdź AUROC na zbiorze testowym.
# 3. Skomentuj wynik w odniesieniu do drzewa decyzyjnego.
#
# **Uwaga:** pamiętaj o ustawieniu `random_state=0`. Dla przyspieszenia ustaw `n_jobs=-1` (użyje tylu procesów, ile masz dostępnych rdzeni procesora). Pamiętaj też o przekazaniu prawdopodobieństw do metryki AUROC.

# %% editable=true pycharm={"name": "#%%\n"} slideshow={"slide_type": ""} tags=["ex"]
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1, random_state=0)
rfc.fit(X_train, y_train)


# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
prediction = rfc.predict_proba(X_test)
score = roc_auc_score(y_test, prediction[:,1])

print(f"AUROC: {score:.4f}")

# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
assert score > 0.85

print("Solution is correct!")

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""} tags=["ex"]
# Otrzymany wynik jest dużo lepszy od poprzedniego. Model poprawnie klasyfikuje blisko 90% przypadków, co sugeruje, że jego predykcje są bardzo skuteczne.

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# Jak zobaczymy poniżej, wynik ten możemy jednak jeszcze ulepszyć!

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# ## Oversampling, SMOTE

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# W przypadku zbiorów niezbalansowanych można dokonać **balansowania (balancing)** zbioru. Są tutaj 2 metody:
# - **undersampling**: usunięcie przykładów z klasy dominującej
# - **oversampling**: wygenerowanie dodatkowych przykładów z klasy mniejszościowej
#
# Undersampling działa dobrze, kiedy niezbalansowanie jest niewielkie, a zbiór jest duży (możemy sobie pozwolić na usunięcie jego części). Oversampling typowo daje lepsze wyniki, istnieją dla niego bardzo efektywne algorytmy. W przypadku bardzo dużego niezbalansowania można zrobić oba.
#
# Typowym algorytmem oversamplingu jest **SMOTE (Synthetic Minority Oversampling TEchnique)**. Działa on następująco:
# 1. Idź po kolei po przykładach z klasy mniejszościowej
# 2. Znajdź `k` najbliższych przykładów dla próbki, typowo `k=5`
# 3. Wylosuj tylu sąsiadów, ile trzeba do oversamplingu, np. jeżeli chcemy zwiększyć klasę mniejszościową 3 razy (o 200%), to wylosuj 2 z 5 sąsiadów
# 4. Dla każdego z wylosowanych sąsiadów wylosuj punkt na linii prostej między próbką a tym sąsiadem. Dodaj ten punkt jako nową próbkę do zbioru
#
# <img src = https://miro.medium.com/max/734/1*yRumRhn89acByodBz0H7oA.png >
#
# Taka technika generuje przykłady bardzo podobne do prawdziwych, więc nie zaburza zbioru, a jednocześnie pomaga klasyfikatorom, bo "zagęszcza" przestrzeń, w której znajduje się klasa pozytywna.
#
# Algorytm SMOTE, jego warianty i inne algorytmy dla problemów niezbalansowanych implementuje biblioteka Imbalanced-learn.

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# ### Zadanie 5 (1 punkt)

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# Użyj SMOTE do zbalansowania zbioru treningowego (nie używa się go na zbiorze testowym!) (klasa `SMOTE`). Wytrenuj drzewo decyzyjne oraz las losowy na zbalansowanym zbiorze, użyj tych samych argumentów co wcześniej. Pamiętaj o użyciu wszędzie stałego `random_state=0`, `n_jobs=-1` oraz przekazaniu prawdopodobieństw do AUROC. Skomentuj wynik.

# %% editable=true pycharm={"name": "#%%\n"} slideshow={"slide_type": ""} tags=["ex"]
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=0, n_jobs=-1)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

dtc = DecisionTreeClassifier(criterion='entropy', random_state=0)
dtc.fit(X_train_sm, y_train_sm)

rfc = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1, random_state=0)
rfc.fit(X_train_sm, y_train_sm)


# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
prediction_dtc = dtc.predict_proba(X_test)
score_dtc = roc_auc_score(y_test, prediction_dtc[:,1])

prediction_rfc = rfc.predict_proba(X_test)
score_rfc = roc_auc_score(y_test, prediction_rfc[:,1])

print(f"AUROC dtc: {score_dtc:.4f}")
print(f"AUROC rtc: {score_rfc:.4f}")

# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
assert 0.6 < score_dtc < 0.8
assert 0.8 < score_rfc < 0.95

print("Solution is correct!")

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""} tags=["ex"]
# Jeśli chodzi o drzewo decyzyjne to nasza metryka jest trochę gorsza. Natomiast dla lasu losowego nasz wynik jest o drobinę lepszy. Na podstawie wyników można wyciągnąć wniosek, że oversampling może zarówno poprawić wynik jak i go pogorszyć, ale zmiany nie są znaczące.

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# W dalszej części laboratorium używaj zbioru po zastosowaniu SMOTE do treningu klasyfikatorów.

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Dostrajanie (tuning) hiperparametrów

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# Lasy losowe są stosunkowo mało czułe na dobór hiperparametrów - i dobrze, bo mają ich dość dużo. Można zawsze jednak spróbować to zrobić, a w szczególności najważniejszy jest parametr `max_features`, oznaczający, ile cech losować przy każdym podziale drzewa. Typowo sprawdza się wartości z zakresu `[0.1, 0.5]`.
#
# W kwestii szybkości, kiedy dostrajamy hiperparametry, to mniej oczywiste jest, jakiego `n_jobs` użyć. Z jednej strony klasyfikator może być trenowany na wielu procesach, a z drugiej można trenować wiele klasyfikatorów na różnych zestawach hiperparametrów równolegle. Jeżeli nasz klasyfikator bardzo dobrze się uwspółbieżnia (jak Random Forest), to można dać mu nawet wszystkie rdzenie, a za to wypróbowywać kolejne zestawy hiperparametrów sekwencyjnie. Warto ustawić parametr `verbose` na 2 lub więcej, żeby dostać logi podczas długiego treningu i mierzyć czas wykonania. W praktyce ustawia się to metodą prób i błędów.

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# ### Zadanie 6 (1 punkt)

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# 1. Dobierz wartość hiperparametru `max_features`:
#    - użyj grid search z 5 foldami
#    - wypróbuj wartości `[0.1, 0.2, 0.3, 0.4, 0.5]`
#    - wybierz model o najwyższym AUROC (argument `scoring`)
# 2. Sprawdź, jaka była optymalna wartość `max_features`. Jest to atrybut wytrenowanego `GridSearchCV`.
# 3. Skomentuj wynik. Czy warto było poświęcić czas i zasoby na tę procedurę?
#
# **Uwaga:**
# - pamiętaj, żeby jako estymatora przekazanego do grid search'a użyć instancji Random Forest, która ma już ustawione `random_state=0` i `n_jobs`

# %% editable=true pycharm={"is_executing": true, "name": "#%%\n"} slideshow={"slide_type": ""} tags=["ex"]
from sklearn.model_selection import GridSearchCV

paremeters = [{"max_features": [0.1,0.2,0.3,0.4,0.5]}]
rfc = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1, random_state=0)
grid_search = GridSearchCV(rfc, paremeters, cv=5, n_jobs=-1, scoring='roc_auc', verbose=2)
results = grid_search.fit(X_train_sm, y_train_sm)

results.best_params_


# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
random_forest_best_params = results.best_params_

# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
print("TRAIN results:")
print(results.best_score_)
print(results.best_estimator_)

# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
auroc = roc_auc_score(y_test, results.predict_proba(X_test)[:, 1])
print(f"AUROC on test data: {auroc:.4f}")

# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
assert 0.9 <= auroc <= 0.95

print("Solution is correct!")

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""} tags=["ex"]
# Otrzymany wynik dla metryki AUROC jest dobry, aczkolwiem uważam, że czekanie 7 minut z maksymalnym wykorzystaniem procesora nie było tego warte osiągnięcia tej wartości na poziomie 0.91. Bez dobierania hiperparametrów otrzymaliśmy podobny wynik.

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# W praktycznych zastosowaniach data scientist wedle własnego uznana, doświadczenia, dostępnego czasu i zasobów wybiera, czy dostrajać hiperparametry i w jak szerokim zakresie. Dla Random Forest na szczęście często może nie być znaczącej potrzeby, i za to go lubimy :)

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# **Random Forest - podsumowanie**
#
# 1. Model oparty o uczenie zespołowe
# 2. Kluczowe elementy:
#    - bagging: uczenie wielu klasyfikatorów na próbkach boostrapowych
#    - metoda losowej podprzestrzeni: losujemy podzbiór cech do każdego podziału drzewa
#    - uśredniamy głosy klasyfikatorów
# 3. Dość odporny na overfitting, zmniejsza wariancję błędu dzięki uśrednianiu
# 4. Mało czuły na hiperparametry
# 5. Przeciętnie bardzo dobre wyniki, doskonały wybór domyślny przy wybieraniu algorytmu klasyfikacji

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# ## Boosting

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# Drugą bardzo ważną grupą algorytmów ensemblingu jest **boosting**, też oparty o drzewa decyzyjne. O ile Random Forest trenował wszystkie klasyfikatory bazowe równolegle i je uśredniał, o tyle boosting robi to sekwencyjnie. Drzewa te uczą się na całym zbiorze, nie na próbkach boostrapowych. Idea jest następująca: trenujemy drzewo decyzyjne, radzi sobie przeciętnie i popełnia błędy na częsci przykładów treningowych. Dokładamy kolejne, ale znające błędy swojego poprzednika, dzięki czemu może to uwzględnić i je poprawić. W związku z tym "boostuje" się dzięki wiedzy od poprzednika. Dokładamy kolejne drzewa zgodnie z tą samą zasadą.
#
# Jak uczyć się na błędach poprzednika? Jest to pewna **funkcja kosztu** (błędu), którą chcemy zminimalizować. Zakłada się jakąś jej konkretną postać, np. squared error dla regresji, albo logistic loss dla klasyfikacji. Później wykorzystuje się spadek wzdłuż gradientu (gradient descent), aby nauczyć się, w jakim kierunku powinny optymalizować kolejne drzewa, żeby zminimalizować błędy poprzednika. Jest to konkretnie **gradient boosting**, absolutnie najpopularniejsza forma boostingu, i jeden z najpopularniejszych i osiągających najlepsze wyniki algorytmów ML.
#
# Tyle co do intuicji. Ogólny algorytm gradient boostingu jest trochę bardziej skomplikowany. Bardzo dobrze i krok po kroku tłumaczy go [ta seria filmów na YT](https://www.youtube.com/watch?v=3CC4N4z3GJc). Szczególnie ważne implementacje gradient boostingu to **XGBoost (Extreme Gradient Boosting)** oraz **LightGBM (Light Gradient Boosting Machine)**. XGBoost był prawdziwym przełomem w ML, uzyskując doskonałe wyniki i bardzo dobrze się skalując - był wykorzystany w CERNie do wykrywania cząstki Higgsa w zbiorze z pomiarów LHC mającym 10 milionów próbek. Jego implementacja jest dość złożona, ale dobrze tłumaczy ją [inna seria filmików na YT](https://www.youtube.com/watch?v=OtD8wVaFm6E).
#
# ![](xgboost.png)
#
# Obecnie najczęściej wykorzystuje się LightGBM. Został stworzony przez Microsoft na podstawie doświadczeń z XGBoostem. Został jeszcze bardziej ulepszony i przyspieszony, ale różnice są głównie implementacyjne. Różnice dobrze tłumaczy [ta prezentacja z konferencji PyData](https://www.youtube.com/watch?v=5CWwwtEM2TA) oraz [prezentacja Microsoftu](https://www.youtube.com/watch?v=5nKSMXBFhes). Dla zainteresowanych - [praktyczne aspekty LightGBM](https://www.kaggle.com/code/prashant111/lightgbm-classifier-in-python/notebook).

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Zadanie 7 (0.5 punktu)

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# 1. Wytrenuj klasyfikator LightGBM (klasa `LGBMClassifier`). Przekaż `importance_type="gain"` - przyda nam się to za chwilę.
# 2. Sprawdź AUROC na zbiorze testowym.
# 3. Skomentuj wynik w odniesieniu do wcześniejszych algorytmów.
#
# Pamiętaj o `random_state`, `n_jobs` i prawdopodobieństwach dla AUROC.

# %% editable=true pycharm={"is_executing": true, "name": "#%%\n"} slideshow={"slide_type": ""} tags=["ex"]
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(random_state=0, n_jobs=-1, importance_type='gain')
lgbm.fit(X_train_sm, y_train_sm)

auroc = roc_auc_score(y_test, lgbm.predict_proba(X_test)[:, 1])
print()
print(f"AUROC: {auroc:.4f}")


# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
assert 0.9 <= auroc <= 0.97

print("Solution is correct!")

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""} tags=["ex"]
# Użycie boostingu poprawiło wartość AUROC w stosunku tegi z poprzedniego zadania oraz kod wykoanł się dużo szybciej. Otrzymaliśmy my także dość dobry wynik AUROC.

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# Boosting dzięki uczeniu na poprzednich drzewach redukuje nie tylko wariancję, ale też bias w błędzie, dzięki czemu może w wielu przypadkach osiągnąć lepsze rezultaty od lasu losowego. Do tego dzięki znakomitej implementacji LightGBM jest szybszy.
#
# Boosting jest jednak o wiele bardziej czuły na hiperparametry niż Random Forest. W szczególności bardzo łatwo go przeuczyć, a większość hiperparametrów, których jest dużo, wiąże się z regularyzacją modelu. To, że teraz poszło nam lepiej z domyślnymi, jest rzadkim przypadkiem.
#
# W związku z tym, że przestrzeń hiperparametrów jest duża, przeszukanie wszystkich kombinacji nie wchodzi w grę. Zamiast tego można wylosować zadaną liczbę zestawów hiperparametrów i tylko je sprawdzić - chociaż im więcej, tym lepsze wyniki powinniśmy dostać. Służy do tego `RandomizedSearchCV`. Co więcej, klasa ta potrafi próbkować rozkłady prawdopodobieństwa, a nie tylko sztywne listy wartości, co jest bardzo przydatne przy parametrach ciągłych.
#
# Hiperparametry LightGBMa są dobrze opisane w oficjalnej dokumentacji: [wersja krótsza](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier) i [wersja dłuższa](https://lightgbm.readthedocs.io/en/latest/Parameters.html). Jest ich dużo, więc nie będziemy ich tutaj omawiać. Jeżeli chodzi o ich dostrajanie w praktyce, to przydatny jest [oficjalny guide](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html) oraz dyskusje na Kaggle.

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# ### Zadanie 8 (1.5 punktu)

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# 1. Zaimplementuj random search dla LightGBMa (klasa `RandomizedSearchCV`):
#    - użyj tylu prób, na ile pozwalają twoje zasoby obliczeniowe, ale przynajmniej 30
#    - przeszukaj przestrzeń hiperparametrów:
#     ```
#     param_grid = {
#         "n_estimators": [100, 250, 500],
#         "learning_rate": [0.05, 0.1, 0.2],
#         "num_leaves": [31, 48, 64],
#         "colsample_bytree": [0.8, 0.9, 1.0],
#         "subsample": [0.8, 0.9, 1.0],
#     }
#     ```
# 2. Wypisz znalezione optymalne hiperparametry.
# 3. Wypisz raporty z klasyfikacji (funkcja `classification_report`), dla modelu LightGBM bez i z dostrajaniem hiperparametrów.
# 4. Skomentuj różnicę precyzji (precision) i czułości (recall) między modelami bez i z dostrajaniem hiperparametrów. Czy jest to pożądane zjawisko w tym przypadku?
#
# **Uwaga:** 
# - koniecznie ustaw `verbose=-1` przy tworzeniu `LGBMClassifier`, żeby uniknąć kolosalnej ilości logów, która potrafi też wyłączyć Jupytera
# - pamiętaj o ustawieniu `importance_type`, `random_state=0` i `n_jobs`, oraz ewentualnie `verbose` w `RandomizedSearchCV` dla śledzenia przebiegu
# - istnieje możliwość, że ustawienie `n_jobs` dla grid searcha będzie szybsze niż dla samego LightGBM; odpowiada to tuningowi wielu klasyfikatorów równolegle, przy wolniejszym treningu pojedynczych klasyfikatorów
# - nie ustawiaj wszędzie `n_jobs=-1`, bo wtedy stworzysz więcej procesów niż rdzeni i spowodujesz thread contention

# %% editable=true pycharm={"is_executing": true, "name": "#%%\n"} slideshow={"slide_type": ""} tags=["ex"]
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
        "n_estimators": [100, 250, 500],
        "learning_rate": [0.05, 0.1, 0.2],
        "num_leaves": [31, 48, 64],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "subsample": [0.8, 0.9, 1.0],
    }
lgbm = LGBMClassifier(random_state=0, n_jobs=3, importance_type='gain')
rscv = RandomizedSearchCV(
    lgbm, param_grid, n_iter=30, scoring="roc_auc", verbose=0, n_jobs=4, random_state=0, return_train_score=True
)
results = rscv.fit(X_train_sm, y_train_sm)
lgbm.fit(X_train_sm, y_train_sm)

lightgbm_best_params = results.best_params_


# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
from sklearn.metrics import classification_report

y_test_pred_lgbm = lgbm.predict(X_test)
y_test_pred_randomized = rscv.predict(X_test)

print("Without tuning")
print(classification_report(y_test, y_test_pred_lgbm))

print("With tuning")
print(classification_report(y_test, y_test_pred_randomized))

auroc = roc_auc_score(y_test, rscv.predict_proba(X_test)[:, 1])
print()
print(f"AUROC with tuning: {auroc:.4f}")

# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
assert 0.9 <= auroc <= 0.99

print("Solution is correct!")

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""} tags=["ex"]
# Po tuningu modelu zauważalny jest istotny wzrost precyzji dla klasy mniejszościowej (klasy 1) z 0.60 do 0.80. Oznacza to, że model lepiej identyfikuje rzeczywiste przypadki klasy 1, zmniejszając liczbę fałszywych alarmów. Z drugiej strony, czułość dla tej klasy nieco spadła, z 0.60 do 0.54, co wskazuje, że model może przeoczyć niektóre przypadki klasy 1. Pomimo tego, wynik F1 wzrósł z 0.60 do 0.64, co wskazuje na lepszy balans między precyzją a czułością. Dodatkowo wzrost wartości wskaźnika AUROC do 0.9466 po tuningu dowodzi, że model osiągnął wyższą ogólną skuteczność.

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# **Boosting - podsumowanie**
#
# 1. Model oparty o uczenie zespołowe
# 2. Kolejne modele są dodawane sekwencyjnie i uczą się na błędach poprzedników
# 3. Nauka typowo jest oparta o minimalizację funkcji kosztu (błędu), z użyciem spadku wzdłuż gradientu
# 4. Wiodący model klasyfikacji dla danych tabelarycznych, z 2 głównymi implementacjami: XGBoost i LightGBM
# 5. Liczne hiperparametry, wymagające odpowiednich metod dostrajania

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# ## Wyjaśnialna AI

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""}
# W ostatnich latach zaczęto zwracać coraz większą uwagę na wpływ sztucznej inteligencji na społeczeństwo, a na niektórych czołowych konferencjach ML nawet obowiązkowa jest sekcja "Social impact" w artykułach naukowych. Typowo im lepszy model, tym bardziej złożony, a najpopularniejsze modele boostingu są z natury skomplikowane. Kiedy mają podejmować krytyczne decyzje, to musimy wiedzieć, czemu predykcja jest taka, a nie inna. Jest to poddziedzina uczenia maszynowego - **wyjaśnialna AI (explainable AI, XAI)**.
#
# Taka informacja jest cenna, bo dzięki temu lepiej wiemy, co robi model. Jest to ważne z kilku powodów:
# 1. Wymogi prawne - wdrażanie algorytmów w ekonomii, prawie etc. ma coraz częściej konkretne wymagania prawne co do wyjaśnialności predykcji
# 2. Dodatkowa wiedza dla użytkowników - często dodatkowe obserwacje co do próbek są ciekawe same w sobie i dają wiedzę użytkownikowi (często posiadającemu specjalistyczną wiedzę z dziedziny), czasem nawet bardziej niż sam model predykcyjny
# 3. Analiza modelu - dodatkowa wiedza o wewnętrznym działaniu algorytmu pozwala go lepiej zrozumieć i ulepszyć wyniki, np. przez lepszy preprocessing danych
#
# W szczególności można ją podzielić na **globalną** oraz **lokalną interpretowalność (global / local interpretability)**. Ta pierwsza próbuje wyjaśnić, czemu ogólnie model działa tak, jak działa. Analizuje strukturę modelu oraz trendy w jego predykcjach, aby podsumować w prostszy sposób jego tok myślenia. Interpretowalność lokalna z kolei dotyczy predykcji dla konkretnych próbek - czemu dla danego przykładu model podejmuje dla niego taką, a nie inną decyzję o klasyfikacji.
#
# W szczególności podstawowym sposobem interpretowalności jest **ważność cech (feature importance)**. Wyznacza ona, jak ważne są poszczególne cechy:
# - w wariancie globalnym, jak mocno model opiera się na poszczególnych cechach
# - w wariancie lokalnym, jak mocno konkretne wartości cech wpłynęły na predykcję, i w jaki sposób
#
# Teraz będzie nas interesować globalna ważność cech. Dla modeli drzewiastych definiuje się ją bardzo prosto. Każdy podział w drzewie decyzyjnym wykorzystuje jakąś cechę, i redukuje z pomocą podziału funkcję kosztu (np. entropię) o określoną ilość. Dla drzewa decyzyjnego ważność to sumaryczna redukcja entropii, jaką udało się uzyskać za pomocą danej cechy. Dla lasów losowych i boostingu sumujemy te wartości dla wszystkich drzew. Alternatywnie można też użyć liczby splitów, w jakiej została użyta dana cecha, ale jest to mniej standardowe.
#
# Warto zauważyć, że taka ważność cech jest **względna**:
# - nie mówimy, jak bardzo ogólnie ważna jest jakaś cecha, tylko jak bardzo przydatna była dla naszego modelu w celu jego wytrenowania
# - ważność cech można tylko porównywać ze sobą, np. jedna jest 2 razy ważniejsza od drugiej; nie ma ogólnych progów ważności
#
# Ze względu na powyższe, ważności cech normalizuje się często do zakresu [0, 1] dla łatwiejszego porównywania.

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# ### Zadanie 9 (0.5 punktu)

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# 1. Wybierz 5 najważniejszych cech dla drzewa decyzyjnego. Przedstaw wyniki na poziomym wykresie słupkowym. Użyj czytelnych nazw cech ze zmiennej `feature_names`.
# 2. Powtórz powyższe dla lasu losowego, oraz dla boostingu (tutaj znormalizuj wyniki - patrz uwaga niżej). Wybierz te hiperparametry, które dały wcześniej najlepsze wyniki.
# 3. Skomentuj, czy wybrane cechy twoim zdaniem mają sens jako najważniejsze cechy.
#
# **Uwaga:** Scikit-learn normalizuje ważności do zakresu [0, 1], natomiast LightGBM nie. Musisz to znormalizować samodzielnie, dzieląc przez sumę.

# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
random_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
random_tree = random_tree.fit(X_train_sm, y_train_sm)

random_forest = RandomForestClassifier(
    random_state=0, criterion='entropy', n_estimators=500, n_jobs=-1, max_features=random_forest_best_params['max_features']
)
random_forest = random_forest.fit(X_train_sm, y_train_sm)

lightgbm = LGBMClassifier(random_state=0, n_jobs=-1, importance_type='gain', **lightgbm_best_params)
lightgbm = lightgbm.fit(X_train_sm, y_train_sm)


# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
import numpy as np

top_features_tree = np.argpartition(random_tree.feature_importances_, -5)[-5:] # select features
top_features_forest = np.argpartition(random_forest.feature_importances_, -5)[-5:]

lightgbm_importance = lightgbm.feature_importances_ 
lightgbm_importance = lightgbm_importance / lightgbm_importance.sum() # normalization
top_features_lightgbm = np.argpartition(lightgbm_importance, -5)[-5:]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
ax1.barh(np.array(feature_names)[top_features_tree], random_tree.feature_importances_[top_features_tree], color='#83eb9f')
ax1.set_title("Decision Tree")
ax1.set_xlabel("Feature importance")

ax2.barh(np.array(feature_names)[top_features_forest], random_forest.feature_importances_[top_features_forest], color="#6deef2")
ax2.set_title("Random Forest")
ax2.set_xlabel("Feature importance")

ax3.barh(np.array(feature_names)[top_features_lightgbm], lightgbm_importance[top_features_lightgbm], color="#b7f58c")
ax3.set_title("LightGBM")
ax3.set_xlabel("Feature importance")

fig.suptitle("Top 5 features for each model")
fig.tight_layout()
fig.subplots_adjust(hspace=0.5)

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# Każdy klasyfikator wskazął bardzo podobne cechy za najważniejsze np. sales(n) / sales(n-1) mówiącą o przychodzie względem poprzedniego roku, co ma sens bo firmy nie przynoszące dochodów mają większą szansę na bankructwo,. Uważam, ze wskazane przez modele cechy mają prawo być naważniejszymy cechami.

# %% [markdown]
# ### Dla zainteresowanych
#
# Najpopularniejszym podejściem do interpretowalności lokalnych jest **SHAP (SHapley Additive exPlanations)**, metoda oparta o kooperatywną teorię gier. Traktuje się cechy modelu jak zbiór graczy, podzielonych na dwie drużyny (koalicje): jedna chce zaklasyfikować próbkę jako negatywną, a druga jako pozytywną. O ostatecznej decyzji decyduje model, który wykorzystuje te wartości cech. Powstaje pytanie - w jakim stopniu wartości cech przyczyniły się do wyniku swojej drużyny? Można to obliczyć jako wartości Shapleya (Shapley values), które dla modeli ML oblicza algorytm SHAP. Ma on bardzo znaczące, udowodnione matematycznie zalety, a dodatkowo posiada wyjątkowo efektywną implementację dla modeli drzewiastych oraz dobre wizualizacje.
#
# Bardzo intuicyjnie, na prostym przykładzie, SHAPa wyjaśnia [pierwsza część tego artykułu](https://iancovert.com/blog/understanding-shap-sage/). Dobrze i dość szczegółówo SHAPa wyjaśnia jego autor [w tym filmie](https://www.youtube.com/watch?v=-taOhqkiuIo).

# %% [markdown] pycharm={"name": "#%% md\n"}
# **Wyjaśnialna AI - podsumowanie**
#
# 1. Problem zrozumienia, jak wnioskuje model i czemu podejmuje dane decyzje
# 2. Ważne zarówno z perspektywy data scientist'a, jak i użytkowników systemu
# 3. Można wyjaśniać model lokalnie (konkretne predykcje) lub globalnie (wpływ poszczególnych cech)

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# ## Zadanie 10 dla chętnych (3 punkty)

# %% [markdown] editable=true pycharm={"name": "#%% md\n"} slideshow={"slide_type": ""} tags=["ex"]
# Dokonaj selekcji cech, usuwając 20% najsłabszych cech. Może się tu przydać klasa `SelectPercentile`. Czy Random Forest i LightGBM (bez dostrajania hiperparametrów, dla uproszczenia) wytrenowane bez najsłabszych cech dają lepszy wynik (AUROC lub innej metryki)?
#
# Wykorzystaj po 1 algorytmie z 3 grup algorytmów selekcji cech:
# 1. Filter methods - mierzymy ważność każdej cechy niezależnie, za pomocą pewnej miary (typowo ze statystyki lub teorii informacji), a potem odrzucamy (filtrujemy) te o najniższej ważności. Są to np. `chi2` i `mutual_info_classif` z pakietu `sklearn.feature_selection`.
# 2. Embedded methods - klasyfikator sam zwraca ważność cech, jest jego wbudowaną cechą (stąd nazwa). Jest to w szczególności właściwość wszystkich zespołowych klasyfikatorów drzewiastych. Mają po wytrenowaniu atrybut `feature_importances_`.
# 2. Wrapper methods - algorytmy wykorzystujące w środku używany model (stąd nazwa), mierzące ważność cech za pomocą ich wpływu na jakość klasyfikatora. Jest to np. recursive feature elimination (klasa `RFE`). W tym algorytmie trenujemy klasyfikator na wszystkich cechach, wyrzucamy najsłabszą, trenujemy znowu i tak dalej.
#
# Typowo metody filter są najszybsze, ale dają najsłabszy wynik, natomiast metody wrapper są najwolniejsze i dają najlepszy wynik. Metody embedded są gdzieś pośrodku.
#
# Dla zainteresowanych, inne znane i bardzo dobre algorytmy:
# - Relief (filter method) oraz warianty, szczególnie ReliefF, SURF i MultiSURF (biblioteka `ReBATE`): [Wikipedia](https://en.wikipedia.org/wiki/Relief_(feature_selection)), [artykuł "Benchmarking Relief-Based Feature Selection Methods"](https://www.researchgate.net/publication/321307194_Benchmarking_Relief-Based_Feature_Selection_Methods)
# - Boruta (wrapper method), stworzony na Uniwersytecie Warszawskim, łączący Random Forest oraz testy statystyczne (biblioteka `boruta_py`): [link 1](https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a), [link 2](https://danielhomola.com/feature%20selection/phd/borutapy-an-all-relevant-feature-selection-method/)

# %% editable=true pycharm={"name": "#%%\n"} slideshow={"slide_type": ""} tags=["ex"]
rfc = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1, random_state=0)
rfc.fit(X_train_sm, y_train_sm)

lgbm = LGBMClassifier(random_state=0, n_jobs=-1, importance_type='gain', verbose=-1)
lgbm.fit(X_train_sm, y_train_sm)

rfc_auroc = roc_auc_score(y_test, rfc.predict_proba(X_test)[:, 1])
lgbm_auroc = roc_auc_score(y_test, lgbm.predict_proba(X_test)[:, 1])

print(f"BEZ SELEKCJI:")
print(f"AUROC Random Forrest: {rfc_auroc:.4f}")
print(f"AUROC LightLGBM: {lgbm_auroc:.4f}")

# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
from sklearn.feature_selection import SelectPercentile, mutual_info_classif

select_percentile = SelectPercentile(mutual_info_classif, percentile=80)
X_train_res_reduced = select_percentile.fit_transform(X_train_sm, y_train_sm)
X_test_reduced = select_percentile.transform(X_test)

rfc = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1, random_state=0)
rfc.fit(X_train_res_reduced, y_train_sm)

lgbm = LGBMClassifier(random_state=0, n_jobs=-1, importance_type='gain', verbose=-1)
lgbm.fit(X_train_res_reduced, y_train_sm)

rfc_auroc = roc_auc_score(y_test, rfc.predict_proba(X_test_reduced)[:, 1])
lgbm_auroc = roc_auc_score(y_test, lgbm.predict_proba(X_test_reduced)[:, 1])

print(f"Filter method:")
print(f"AUROC Random Forrest: {rfc_auroc:.4f}")
print(f"AUROC LightLGBM: {lgbm_auroc:.4f}")

# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
from sklearn.feature_selection import SelectFromModel

threshold = np.quantile(random_forest_standard.feature_importances_, 0.2)

sm_forest = SelectFromModel(rfc, threshold=threshold)

X_train_forest_embedded = sm_forest.fit_transform(X_train_sm, y_train_sm)
X_test_forest_embedded = sm_forest.transform(X_test)

sm_lightgbm = SelectFromModel(lgbm, threshold=threshold)

X_train_lightgbm_embedded = sm_lightgbm.fit_transform(X_train_sm, y_train_sm)
X_test_lightgbm_embedded = sm_lightgbm.transform(X_test)

# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
rfc = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1, random_state=0)
rfc.fit(X_train_forest_embedded, y_train_sm)

lgbm = LGBMClassifier(random_state=0, n_jobs=-1, importance_type='gain', verbose=-1)
lgbm.fit(X_train_lightgbm_embedded, y_train_sm)

rfc_auroc = roc_auc_score(y_test, rfc.predict_proba(X_test_forest_embedded)[:, 1])
lgbm_auroc = roc_auc_score(y_test, lgbm.predict_proba(X_test_lightgbm_embedded)[:, 1])

print(f"Embedded method:")
print(f"AUROC Random Forrest: {rfc_auroc:.4f}")
print(f"AUROC LightLGBM: {lgbm_auroc:.4f}")

# %% editable=true slideshow={"slide_type": ""} tags=["ex"]
from sklearn.feature_selection import RFE

rfc = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1, random_state=0)
rfc_rfe = RFE(rfc, step=1, n_features_to_select=int(0.8*len(feature_names)))
rfc_rfe.fit(X_train_sm, y_train_sm)

lgbm = LGBMClassifier(random_state=0, n_jobs=-1, importance_type='gain', verbose=-1)
lgbm_rfe = RFE(lgbm, step=1, n_features_to_select=int(0.8*len(feature_names)))
lgbm_rfe.fit(X_train_sm, y_train_sm)

rfc_auroc = roc_auc_score(y_test, rfc_rfe.predict_proba(X_test)[:, 1])
lgbm_auroc = roc_auc_score(y_test, lgbm_rfe.predict_proba(X_test)[:, 1])

print(f"Wrapper method:")
print(f"AUROC Random Forrest: {rfc_auroc:.4f}")
print(f"AUROC LightLGBM: {lgbm_auroc:.4f}")

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["ex"]
# Wyniki bez selekcji oraz z selekcją są bardzo do siebie zbliżone, zatem można wniowskować, że 20% najmniej znaczące cechy nie wpływają zbytno na trening modeli. Nie są dla nich istnote.

# %% editable=true slideshow={"slide_type": ""}
