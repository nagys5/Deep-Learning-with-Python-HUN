# **9. Haladó mély tanulás a gépi látáshoz**

Ez a fejezet ezekkel foglalkozik:
* A gépi látás különböző ágai: képosztályozás, képszegmentálás, tárgydetektálás
* Modern convnet architektúra minták: maradék kapcsolatok, köteg normalizálás, mélységben szétválasztható konvolúciók
* Technikák a convnet által tanultak megjelenítésére és értelmezésére

Az előző fejezetben először bemutattuk a gépi látás mély tanulását egyszerű modelleken (`Conv2D` és `MaxPooling2D` rétegek halmaza) és egy egyszerű használati eseten (bináris képosztályozás) keresztül. De a gépi látás többről szól, mint a képbesorolás! Ez a fejezet mélyebben belemerül a változatosabb alkalmazásokba és a legjobb haladó gyakorlatokba.

## 9.1 A három alapvető (számítógépes) gépi látási feladat

Eddig a képbesorolási modellekre koncentráltunk: egy kép bemegy, egy címke jön ki. „Ez a kép valószínűleg egy macskát tartalmaz; ezen a másikon valószínűleg egy kutya van." A képosztályozás azonban csak egy a gépi látás lehetséges mélytanulási alkalmazásai közül. Általában három alapvető gépi látási feladatot kell ismernie:
* _Képosztályozás_ – Ahol a cél egy vagy több címke hozzárendelése egy képhez. Ez lehet egycímkés besorolás (egy kép csak egy kategóriába tartozhat, a többit nem vesszük figyelembe), vagy többcímkés besorolás (minden kategória felcímkézése, amelyhez egy kép tartozik, amint az a 9.1. ábrán látható). Például, amikor kulcsszóra keresünk a Google Fotók alkalmazásban, a kulisszák mögött egy nagyon nagy, többcímkés osztályozási modellt kérdezünk le – egy olyan modellt, amely több mint 20 000 különböző osztályt tartalmaz, amelyek több millió képpel vannak betanítva.
* _Képszegmentálás_ – ahol a cél egy kép „szegmentálása” vagy „felosztása” különböző területekre, ahol minden terület általában egy kategóriát képvisel (amint a 9.1. ábrán látható). Például amikor a Zoom vagy a Google Meet egyéni hátteret jelenít meg mögötted egy videohívásban, akkor egy képszegmentációs modellt használ, hogy pixel pontossággal meg tudja különböztetni az arcot aattól ami mögötte van.
* _Tárgyérzékelés_ – A cél az, hogy téglalapokat (úgynevezett határolókereteket) rajzoljunk a kép érdekes objektumai köré, és minden téglalapot társítsunk egy osztályhoz. Egy önvezető autó objektumérzékelő modellt használhat például az autók, a gyalogosok és a táblák megfigyelésére a kameráinak a látképei alapján.

![Próba](figs/f9.1_.jpg)

**9.1. ábra:** A gépi látás három fő feladata: osztályozás, szegmentálás, észlelés

A gépi látás mély tanulása ezen a háromon kívül számos, némileg szűkebb feladatot is magában foglal, mint például a képhasonlósági pontozás (a két kép vizuális hasonlóságának becslése), a kulcspontok felismerése (a képen az érdeklődésre számot tartó attribútumok, például az arcvonások meghatározása), pózbecslés, 3D mesh becslés stb. Kezdetben azonban a képosztályozás, a képszegmentálás és az objektumészlelés jelenti azt az alapot, amelyet minden gépi tanulási mérnöknek ismernie kell. A legtöbb gépilátási alkalmazás e három egyikére csapódik le.

Az előző fejezetben már láttuk a képbesorolást működés közben. Ezután merüljünk el a képszegmentálásban. Ez egy nagyon hasznos és sokoldalú technika, és egyenesen meg lehet közelíteni az eddig tanultakkal.

Ne feledje, hogy nem térünk ki az objektumészlelésre, mert az túl speciális és túl bonyolult lenne egy bevezető könyvhöz. Azonban megtekintheti a RetinaNet példáját a keras.io webhelyen, amely bemutatja, hogyan lehet objektumészlelési modellt felépíteni a semmiből és betanítani a Kerasban körülbelül 450 kódsorral (https://keras.io/examples/vision/retinanet/ ).

## 9.2 Példa a képszegmentálásra

A képszegmentálás mély tanulással azt jelenti, hogy egy modell segítségével osztályt rendelünk a kép minden pixeléhez, így a képet különböző zónákra felosztva (például „háttér” és „előtér” vagy „út”, „autó” és „ járda"). A technikák ezen általános kategóriája számos hasznos alkalmazás működtetésére használható a kép- és videószerkesztésben, az önvezetésben, a robotikában, az orvosi képalkotásban és így tovább.

A képszegmentálásnak két különböző változata van, amelyekről tudnunk kell:
* _Szemantikai szegmentálás_, ahol minden képpont egymástól függetlenül egy jelentéstani kategóriába van besorolva, például „macska”. Ha két macska van a képen, a megfelelő pixelek ugyanahhoz az általános „macska” kategóriához vannak hozzárendelve (lásd a 9.2. ábrát).
* _Példányszegmentálás_, amely nem csak a képpixelek kategóriák szerinti osztályozására törekszik, hanem az egyes objektumpéldányok elemzésére is. Egy két macskát tartalmazó képen a példányszegmentálás az „1. macska”-t és a „2. macska”-t a pixelek két külön osztályaként kezelné (lásd a 9.2. ábrát).

Ebben a példában a szemantikai szegmentációra összpontosítunk: ismét macskák és kutyák képeit nézzük, és ezúttal megtanuljuk, hogyan lehet megkülönböztetni a fő témát és annak hátterét.

Az Oxford-IIIT Pets adatkészlettel (www.robots.ox.ac.uk/~vgg/data/pets/) fogunk dolgozni, amely 7390 képet tartalmaz különböző fajtájú macskákról és kutyákról, valamint előtér-háttér _szegmentáló maszkokat_ minden egyes képhez. A szegmentációs maszk a címke képszegmentálási megfelelője: ez egy olyan kép, amely megegyezik a bemeneti képpel, de csak egyetlen színcsatornával, ahol minden egész érték a bemeneti kép megfelelő pixel osztályának felel meg. Esetünkben a szegmentációs maszkjaink képpontjai a következő három egész érték valamelyikét vehetik fel:
* 1 (előtér)
* 2 (háttér)
* 3 (kontúr)
​

![](figs/f9.2_.jpg)

**9.2. ábra:** Szemantikus szegmentálás kontra példányszegmentálás

Kezdjük adatkészletünk letöltésével és kibontásával a wget és tar shell segédprogramok használatával:
```
!wget http:/ /www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
!tar -xf images.tar.gz
!tar -xf annotations.tar.gz
```
A bemeneti képek JPG fájlokként kerülnek tárolásra az images/ mappában (például images/Abyssinian_1.jpg), a megfelelő szegmentációs maszk pedig azonos nevű PNG-fájlként az annotations/trimaps/ mappában (például annotations/ trimaps/Abyssinian_1.png).

Készítsük el a bemeneti fájl útvonalak listáját, valamint a megfelelő maszkfájl útvonalak listáját:


```python
import os

input_dir = "images/"
target_dir = "annotations/trimaps/"

input_img_paths = sorted(
    [os.path.join(input_dir, fname)
     for fname in os.listdir(input_dir)
     if fname.endswith(".jpg")])

target_paths = sorted(
    [os.path.join(target_dir, fname)
     for fname in os.listdir(target_dir)
     if fname.endswith(".png") and not fname.startswith(".")])
```

Most hogyan néz ki az egyik ilyen bemenet és a maszkja? Vessünk rá egy gyors pillantást. Íme egy mintakép (lásd a 9.3. ábrát):

![](figs/f9.3_.jpg)

**9.3. ábra:** Egy példakép


```python
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array

plt.axis("off")
plt.imshow(load_img(input_img_paths[9]))    #<--- A 9-es számú bemeneti kép megjelenítése.

def display_target(target_array):
    normalized_array = (target_array.astype("uint8") - 1) * 127 #<--- Az eredeti címkék 1, 2 és 3. Kivonunk 1-et,
                                                                #     hogy a címkék 0 és 2 között legyenek,
                                                                #     majd megszorozzuk 127-tel, így a címkék 0-ra (fekete),
                                                                #     127-re (szürke), 254-re (majdnem fehérre) váltanak.
    plt.axis("off")
    plt.imshow(normalized_array[:, :, 0])

img = img_to_array(load_img(target_paths[9], color_mode="grayscale")) #<--- A color_mode="grayscale"-t használjuk, így
                                                                      #     a betöltött képet egyetlen színcsatornaként kezeljük.
display_target(img)
```

És itt van a megfelelő cél (lásd a 9.4. ábrát):

![](figs/f9.4_.jpg)

**9.4. ábra:** A megfelelő célmaszk

Ezután töltsük be a bemeneteinket és a célokat két NumPy tömbbe, és osszuk fel a tömböket egy képzési és egy érvényesítési halmazra. Mivel az adatkészlet nagyon kicsi, mindent be tudunk tölteni a memóriába:


```python
import numpy as np
import random

img_size = (200, 200)           #<--- Mindent átméretezünk 200 × 200-ra.
num_imgs = len(input_img_paths) #<--- Az adatokban szereplő minták teljes száma

random.Random(1337).shuffle(input_img_paths)  #<--- Keverje meg a fájl elérési útját (eredetileg fajták szerint rendezték őket).
random.Random(1337).shuffle(target_paths)     #     Mindkét utasításban ugyanazt a magot (1337) használjuk annak biztosítására,
                                              #     hogy a bemeneti útvonalak és a cél útvonalak ugyanabban a sorrendben maradjanak.
def path_to_input_image(path):
    return img_to_array(load_img(path, target_size=img_size))

def path_to_target(path):
    img = img_to_array(
        load_img(path, target_size=img_size, color_mode="grayscale"))
    img = img.astype("uint8") - 1             #<--- Vonjunk ki 1-et, hogy a címkéink 0, 1 és 2 legyenek.
    return img

input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32") #<--- Töltse be az input_imgs float32 tömbben található
targets = np.zeros((num_imgs,) + img_size + (1,), dtype="uint8")      #     összes képet és a maszkjaikat a targets uint8 tömbbe
for i in range(num_imgs):                                             #     (ugyanabban a sorrendben). A bemeneteknek három
    input_imgs[i] = path_to_input_image(input_img_paths[i])           #     csatornájuk van (RBG értékek), a célpontoknak pedig
    targets[i] = path_to_target(target_paths[i])                      #     egyetlen csatornájuk van (amely egész számokat tartalmaz).

num_val_samples = 1000                            #<--- Tartson fenn 1000 mintát az érvényesítéshez.
train_input_imgs = input_imgs[:-num_val_samples]  #<--- Ossza fel az adatokat egy képzési
train_targets = targets[:-num_val_samples]        #     és egy érvényesítési halmazra.
val_input_imgs = input_imgs[-num_val_samples:]    #
val_targets = targets[-num_val_samples:]          #
```

Most itt az ideje, hogy definiáljuk a modellünket:


```python
from tensorflow import keras
from tensorflow.keras import layers

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    x = layers.Rescaling(1./255)(inputs)        #<--- Ne felejtse el átméretezni a bemeneti képeket [0-1] tartományra.

    x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x) #<--- Jegyezze meg, hogy mindenhol a
                                                                              #     padding="same"-et használjuk, hogy
                                                                              #     elkerüljüka szegélyek kitöltésének
                                                                              #     hatását a jellemzőtérkép méretére.
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)

    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(
        256, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(
        128, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(
        64, 3, activation="relu", padding="same", strides=2)(x)

    outputs = layers.Conv2D(num_classes, 3, activation="softmax",   #<--- A modellt egy pixelenkénti háromutas softmax-szal
     padding="same")(x)                                             #     zárjuk, hogy minden kimeneti pixelt a három kategória
                                                                    #     valamelyikébe soroljunk.
    model = keras.Model(inputs, outputs)
    return model

model = get_model(img_size=img_size, num_classes=3)
model.summary()
```

Íme a `model.summary()` hívás kimenete:

```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 200, 200, 3)]     0
_________________________________________________________________
rescaling (Rescaling)        (None, 200, 200, 3)       0
_________________________________________________________________
conv2d (Conv2D)              (None, 100, 100, 64)      1792
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 100, 100, 64)      36928
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 50, 50, 128)       73856
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 50, 50, 128)       147584
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 25, 25, 256)       295168
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 25, 25, 256)       590080
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 25, 25, 256)       590080
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 50, 50, 256)       590080
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 50, 50, 128)       295040
_________________________________________________________________
conv2d_transpose_3 (Conv2DTr (None, 100, 100, 128)     147584
_________________________________________________________________
conv2d_transpose_4 (Conv2DTr (None, 100, 100, 64)      73792
_________________________________________________________________
conv2d_transpose_5 (Conv2DTr (None, 200, 200, 64)      36928
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 200, 200, 3)       1731
=================================================================
Total params: 2,880,643
Trainable params: 2,880,643
Non-trainable params: 0
_________________________________________________________________
```

A modell első fele nagyon hasonlít a képosztályozáshoz használt convnethez: `Conv2D` rétegek halmaza, fokozatosan növekvő szűrőmérettel. A képeinkből háromszor, egyenként kétszeres mintát veszünk, aminek eredményeként a `(25, 25, 256)` méret aktiválódik. Ennek az első félnek az a célja, hogy a képeket kisebb jellemzőtérképekre kódolja, ahol minden egyes térbeli hely (vagy pixel) az eredeti kép egy nagy térbeli darabjáról tartalmaz információt. Felfoghatod egyfajta tömörítésként is.

Az egyik fontos különbség ennek a modellnek az első fele és a korábban látott osztályozási modellek között a lemintavételezés módja: az utolsó fejezet osztályozási convnetjeiben `MaxPooling2D` rétegeket használtunk a jellemzőtérképek mintavételezésére. Itt a mintavételt úgy végezzük, hogy minden második konvolúciós réteghez _lépésközöket_ adunk (ha nem emlékszik a konvolúciós lépésközök működésének részleteire, lásd a 8.1.1. szakasz „A konvolúciós lépésköz értelmezése” című részt). Ezt azért tesszük, mert a képszegmentálásnál nagyon fontosnak tartjuk az információ _térbeli elhelyezkedését_ a képen, hiszen a modell kimeneteként pixelenkénti célmaszkokat kell készítenünk. Ha 2 × 2 max poolingot hajt végre, akkor teljesen megsemmisíti a helyinformációkat az egyes gyűjtőablakon belül: ablakonként egy skaláris értéket ad vissza, anélkül, hogy az ablakok négy helye közül melyikről származik az érték. Tehát bár a max pooling rétegek jól teljesítenek az osztályozási feladatoknál, egy szegmentálási feladatnál igencsak ártanak nekünk. Eközben a lépésközös konvolúciók jobb munkát végeznek a jellemzőtérképek mintavételezésében, miközben megtartják a helyinformációkat. Fel fog tűnni, hogy ebben a könyvben hajlamosak vagyunk a lépésköz használatára a maximális összevonás helyett minden olyan modellben, amely törődik a jellemzők elhelyezkedésével, például a 12. fejezetben található generatív modelleknél.

A modell második fele a `Conv2DTranspose` rétegek halma. Mik azok? Nos, a modell első felének kimenete egy `(25, 25, 256)` alakú térkép, de azt szeretnénk, hogy a végső kimenetünk a célmaszkokéval megegyező `(200, 200, 3)` alakú legyen. Ezért az eddig alkalmazott transzformációk egyfajta _inverzét_ kell alkalmaznunk – olyasvalamit, amely _felfelé mintavételezi_ a jellemzőtérképeket, ahelyett, hogy lefelé mintavételezné azokat. Ez a `Conv2DTranspose` réteg célja: egyfajta konvolúciós rétegként fogható fel, amely megtanulja a felmintavételezést. Ha van egy `(100, 100, 64)` alakú bemenetünk, és átfuttatjuk a `Conv2D(128, 3, strides=2, padding="same")` rétegen, akkor egy `(50, 50, 128)` alakú kimenetet kapunk. Ha ezt a kimenetet a `Conv2DTranspose(64, 3, strides=2, padding="same")` rétegen futtatja, visszakapja az eredetivel megegyező `(100, 100, 64)` alakú kimenetet. Tehát miután a bemeneteinket `(25, 25, 256)` alakú térképekké tömörítettük egy halom `Conv2D` rétegen keresztül, egyszerűen alkalmazhatjuk a megfelelő `Conv2DTranspose` rétegsorozatot, hogy visszatérjünk a `(200, 200, 3)` alakú képekhez.

Most már össze tudjuk állítani és illeszteni a modellünket:


```python
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.keras",
                                    save_best_only=True)
]

history = model.fit(train_input_imgs, train_targets,
                    epochs=50,
                    callbacks=callbacks,
                    batch_size=64,
                    validation_data=(val_input_imgs, val_targets))
```

Rajzoljuk fel a betanítási és érvényesítési veszteségünket (lásd a 9.5. ábrát):

![](figs/f9.5_.jpg)

**9.5. ábra:** Betanítási és érvényesítési veszteséggörbék megjelenítése


```python
epochs = range(1, len(history.history["loss"]) + 1)
loss = history.history["loss"]
val_loss = history.history["val_loss"]
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
```

Látható, hogy félúton kezdjük a túlillesztést, a 25. tanítási szakasz környékén. Töltsük újra a legjobban teljesítő modellünket az érvényesítési veszteségnek megfelelően, és mutassuk be, hogyan használjuk a szegmentációs maszk előrejelzésére (lásd a 9.6. ábrát):


```python
from tensorflow.keras.utils import array_to_img

model = keras.models.load_model("oxford_segmentation.keras")

i = 4
test_image = val_input_imgs[i]
plt.axis("off")
plt.imshow(array_to_img(test_image))

mask = model.predict(np.expand_dims(test_image, 0))[0]

def display_mask(pred):               #<--- Segédprogram a modell előrejelzésének megjelenítéséhez
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.axis("off")
    plt.imshow(mask)

display_mask(mask)
```

![](figs/f9.6_.jpg)

**9.6. ábra:** Egy tesztkép és a hozzá tartozó szegmentációs maszk

Az előrejelzett maszkunkban van néhány apró melléktermék, amelyeket az előtérben és a háttérben lévő geometriai alakzatok okoznak. Ennek ellenére úgy tűnik, hogy a modellünk jól működik.

Ekkorra már a 8. fejezetben és a 9. fejezet elején megtanultad a képosztályozás és képszegmentálás alapjait: már sok mindent képes vagy elérni azzal, amit tudsz. A tapasztalt mérnökök által a valós problémák megoldására kifejlesztett convnetek azonban nem olyan egyszerűek, mint azok, amelyeket eddigi bemutatóink során használtunk. Még mindig hiányoznak az alapvető mentális modellek és gondolkodási folyamatok, amelyek lehetővé teszik a szakértők számára, hogy gyors és pontos döntéseket hozzanak a legmodernebb modellek összeállításával kapcsolatban. E szakadék áthidalásához meg kell tanulnunk a _szerkezeti mintákat_. Merüljünk bele.

## 9.3 Modern convnet architektúra minták

A modell „architektúrája” a létrehozásához szükséges választási lehetőségek összessége: mely rétegeket használjuk, hogyan konfiguráljuk őket, és milyen elrendezésben kapcsoljuk össze őket. Ezek a választások határozzák meg a modell _hipotézisterét_: a lehetséges függvények terét, amelyen a gradiens ereszkedés kereshet, a modell súlyaival paraméterezve. A jellemzőtervezéshez hasonlóan a jó hipotézistér az adott problémával és annak megoldásával kapcsolatos _előzetes tudást_ kódolja. Például a konvolúciós rétegek használata azt jelenti, hogy előre tudja, hogy a bemeneti képekben található fontos minták fordításinvariánsak. Ahhoz, hogy hatékonyan tanulhasson az adatokból, feltételezéseket kell készítenie arról, hogy mit keres.


