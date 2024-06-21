# Generacja muzyki na podstawie opisu

## Wprowadzenie

Projekt ten skupia się na generowaniu muzyki na podstawie dostarczonych do niej opisów tekstowych.
Celem jest opracowanie modelu, który będzie w stanie zrozumieć dany opis i stworzyć odpowiadający mu utwór muzyczny.
Projekt wykorzystuje techniki głębokiego uczenia, w tym przetwarzanie języka naturalnego (NLP) i modelowanie sekwencji, aby osiągnąć ten cel.

## Zbiór danych

### Pobranie

Wykorzystany zbiór danych do tego zadania zawierał 5,521 rekordów z czego w każdym znajdowały się następujące inforamcje:
- identyfikator filmu na youtube
- sekunda początku i końca fragmentu utworu
- lista cech opisująca muzykę - przykładowo `["pop", "tinny wide hi hats", "mellow piano melody", "high pitched female vocal melody", "sustained pulsating synth lead"]`
- opis tekstowy utworu - przykładowo `"A low sounding male voice is rapping over a fast paced drums playing a reggaeton beat along with a bass. Something like a guitar is playing the melody along. This recording is of poor audio-quality. In the background a laughter can be noticed. This song may be playing in a bar."`

Poprzez identyfikator oraz flagę startu i końca fragmentu z wykorzystaniem biblioteki `yt-dlp` pobierane zostały wszystkie pliki z rozszerzeniem `.wav`

### Preprocessing

Aby stworzyć model text-to-audio pobrane pliki audio musiały zostać przetwożone na postać nadającą się do tego rodzaju zadania.
W tym celu wszystkie pliki `.wav` zostały zamienione na spektogramy mel.
Spektrogram mel to reprezentacja krótkoterminowego widma mocy dźwięku, która rejestruje zawartość częstotliwości sygnału audio w czasie.

<p align="center">
  <img src="/images/transform.jpeg">
</p>

Reprezentacja ta jest odpowiednia do wprowadzania danych do modeli uczenia maszynowego, szczególnie tych zaprojektowanych do analizy i generowania dźwięku.

Przykładowy spektogram mel wygląda w następujący sposób:

<p align="center">
  <img src="/images/spectogram_og.png">
</p>

Spektogramy w procesie szkolenia są postaci macierzy w formacie `numpy` o kształcie `(256,420)`, gdzie 256 oznacza rozdzielczość podczas transformacji dźwięku na spektogramy (im wyższa wartość tym lepsza jakość ale też i większy rozmiar) a 420 oznacza długość próbki.

## Trening modelu

Podejście do tego zadania opierało się przede wszystkim na dwóch głównych elementach architektury tj. transformerach oraz sieciach LSTM oraz dwóch podejściach do wykorzystania dostępnego tekstu, czyli mniejsze wejście z wykorzystaniem listy cech lub wejście rozszerzone gdzie wprowadzany był cały tekst opisujący dany dźwięk.
W oparciu o te założenia sprawdzane oraz uczone było wiele architektur z wykorzystaniem akceleracji na GPU P100.

Żadna z sieci niestety nie przyniosła zakładanych wyników które pozowliłby wygenerować dźwięk nie będący szumem.

### Architektura modelu

Architekturą która osiągneła najlepsze wyniki jest `MusicGenerationModel` która łączy przetwarzanie języka naturalnego z modelowaniem sekwencji w celu generowania spektrogramów mel z opisów tekstowych. Poniżej znajduje się szczegółowy opis każdego komponentu:

1. Tokenizer i enkoder BERT:
    - **Tokenizer**: `BertTokenizer` służy do tokenizacji wejściowych opisów tekstowych. Konwertuje on tekst do formatu odpowiedniego dla modelu BERT.
    - **Enkoder**: `BertModel` jest wstępnie wytrenowanym modelem zbudowanym na transformerach, który enkoduje ztokenizowany tekst w osadzenia. Te osadzenia przechwytują semantyczne znaczenie tekstu.
2. W pełni połączona warstwa:
    - **Text to LSTM**: Dane wyjściowe modelu BERT są przekazywane przez w pełni połączoną warstwę (`fc_text_to_lstm`) w celu odwzorowania osadzenia tekstu na wymiar wejściowy LSTM.
3. LSTM (Long Short-Term Memory):
    - **Warstwy LSTM**: Główny moduł w generowaniu sekwencji jest obsługiwany przez sieć LSTM. LSTM przetwarza osadzenia tekstu i generuje sekwencję danych wyjściowych, które odpowiadają ramkom spektrogramu mel.
    - **Hidden and Cell States**: LSTM utrzymuje stany ukryte i komórkowe, aby śledzić zależności czasowe w całej sekwencji.
4. Warstwa wyjściowa:
    - **LSTM to Output**: Wyjścia LSTM są przekazywane przez kolejną w pełni połączoną warstwę (`fc_lstm_to_output`) w celu odwzorowania wyjść LSTM na końcowe wartości spektrogramu mel.
5. Warstwa dropout:
    - **Droput**: Warstwa dropout służy do zapobiegania nadmiernemu dopasowaniu poprzez losowe ustawienie ułamka jednostek wejściowych na zero podczas treningu.

### Proces uczenia

W procesie uczenia wykorzystane zostały następujące działania:
- Funkcja strarty: Funkcja straty używana w tym modelu **Mean Squared Error Loss (MSELoss)**. Jest ona powszechnie stosowana w zadaniach regresji, w których celem jest zminimalizowanie różnicy między wartościami przewidywanymi i rzeczywistymi. W tym przypadku pomaga ona zmierzyć, jak blisko rzeczywistych wartości znajdują się wygenerowane wartości spektrogramu Mel.
- Optimizer: Zastosowany optymalizator to **Adam**, który jest adaptacyjnym algorytmem optymalizacji szybkości uczenia się. Został on zaprojektowany do obsługi rzadkich gradientów w zaszumionych problemach. Współczynnik uczenia (lr) jest ustawiony na 1e-5, co kontroluje wielkość kroku podczas procesu optymalizacji.
- Cosine Annealing Learning Rate Scheduler: Harmonogram tempa uczenia dostosowuje tempo uczenia podczas szkolenia. Harmonogram `CosineAnnealingLR` zmniejsza szybkość uczenia się zgodnie z krzywą kosinusową, co pomaga w stopniowym zmniejszaniu szybkości uczenia się i może prowadzić do lepszej zbieżności. Parametr `T_max` jest ustawiony na liczbę epok (50), wskazując okres wyżarzania kosinusowego.
- Early Stopping: Jest to technika zapobiegająca nadmiernemu dopasowaniu poprzez zatrzymanie procesu uczenia, jeśli strata walidacyjna nie poprawi się przez określoną liczbę epok (`early_stopping_patience`). Zmienna `best_loss` jest inicjowana na dużą wartość, aby śledzić najniższą stratę walidacyjną, a licznik `early_stopping_counter` śledzi, ile epok minęło od ostatniej poprawy.

Ostatecznie model osiągnał wynik 198.376 na zbiorze walidacyjnym po 27 epokach traningu (~11,5h).

<p align="center">
  <img src="/images/losses.png">
</p>

### Predykcja modelu

Predykcja otrzymane dla przykładowego tekstu:

`'The low quality recording features a ballad song that contains sustained strings, mellow piano melody and soft female vocal singing over it. It sounds sad and soulful, like something you would hear at Sunday services.'`

Wygląda następująco:

<p align="center">
  <img src="/images/spectogram_model.png">
</p>
