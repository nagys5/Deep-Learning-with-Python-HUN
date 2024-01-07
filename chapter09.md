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

![osztályozás, szegmentálás, észlelés](figs/f9.1_.jpg)

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

A modellarchitektúra gyakran a különbség a siker és a kudarc között. Ha nem megfelelő architektúrát választ, előfordulhat, hogy a modell elakad a szuboptimális mérőszámoknál, és semmilyen betanítási adat sem fogja megmenteni. Ezzel szemben a jó modellarchitektúra felgyorsítja a tanulást, és lehetővé teszi, hogy a modell hatékonyan használja fel a rendelkezésre álló betanítási adatokat, csökkentve a nagy adathalmazok szükségességét. A jó modellarchitektúra az, amely _csökkenti a keresési terület méretét_, vagy más módon _megkönnyíti a keresési tér egy jó pontjához való konvergálást_. Csakúgy, mint a jellemzőtervezés és az adatkezelés, a modellarchitektúra is arról szól, hogy _egyszerűbbé tegye a problémát_ a gradiensereszkedés megoldásához. És ne feledje, hogy a gradiens süllyedés meglehetősen ostoba keresési folyamat, ezért minden segítségre szüksége van.

A modellépítészet inkább művészet, mint tudomány. A tapasztalt gépi tanulási mérnökök már első próbálkozásukra képesek intuitív módon összeállítani a nagy teljesítményű modelleket, míg a kezdőknek gyakran nehézséget okoz, hogy olyan modellt alkossanak, amely egyáltalán edz. A kulcsszó itt az _intuitív_: senki sem tud egyértelmű magyarázatot adni arra, hogy mi működik és mi nem. A szakértők a mintaillesztésre hagyatkoznak, amely képességre kiterjedt gyakorlati tapasztalat révén sajátítanak el. Ebben a könyvben fejlesztheti saját intuícióját. Azonban nem _minden_ az intuíción múlik – a tényleges tudománynak nincs sok módja, de mint minden mérnöki tudományágban, itt is vannak bevált gyakorlatok.

A következő szakaszokban áttekintünk néhány a gyakorlatban bevált alapvető convnet architektúrát: különösen a maradék kapcsolatokat, a kötegelt normalizálást és az elválasztható konvolúciókat. Miután elsajátította a használatukat, képes lesz rendkívül hatékony képmodelleket készíteni. Alkalmazni is fogjuk ezeket a macska vs. kutya osztályozási problémánkra. {249.o->:}

Kezdjük madártávlatból: a modularitás-hierarchia-újrafelhasználás (MHR) képletével a rendszerarchitektúrához.

### 9.3.1 Modularitás, hierarchia és újrafelhasználás

Ha egyszerűbbé szeretne tenni egy összetett rendszert, van egy univerzális recept, amelyet alkalmazhat: egyszerűen strukturálja _modulokba_ a komplexitás amorf levesét, rendezze a modulokat _hierarchiába_, és kezdje el _újra felhasználni_ ugyanazokat a modulokat több helyen is (az "újrafelhasználás" egy másik szó az _absztrakcióra_ ebben az összefüggésben). Ez az MHR képlet (modularity-hierarchy-reuse), és ez alapozza meg a rendszerarchitektúrát szinte minden olyan tartományban, ahol az „architektúra” kifejezést használják. Bármilyen értelmes komplexitású rendszer szervezetének középpontjában áll, legyen az egy katedrális, a saját tested, az amerikai haditengerészet vagy a Keras kódbázis (lásd a 9.7. ábrát).

![](figs/f9.7_.jpg)

**9.7. ábra:** Az összetett rendszerek hierarchikus struktúrát követnek, és különálló modulokba vannak szervezve, amelyeket többször is felhasználnak (például a négy végtag, amelyek mind ugyanannak a tervnek a változatai, vagy a 20 „ujjad”).

Ha ön szoftvermérnök, akkor már nagyon jól ismeri ezeket az alapelveket: a hatékony kódbázis az, amely moduláris, hierarchikus, és ahol nem hajtja végre ugyanazt a dolgot kétszer, hanem az újrafelhasználható osztályokra és függvényekre hagyatkozik. Ha ezeket az elveket követve szemléli a kódot, akkor azt mondhatja, hogy „szoftverarchitektúrát” csinál.

Maga a mélytanulás egyszerűen ennek a receptnek az alkalmazása a folyamatos optimalizálásra gradiensereszkedés útján: egy klasszikus optimalizálási technikát használ (gradiens süllyedés egy folytonos függvénytéren), és a keresési teret modulokba (rétegekbe) strukturálja, mély hierarchiába rendezve (ami gyakran csak egy verem, a hierarchia legegyszerűbb fajtája), ahol bármit újra felhasználhat (például a konvolúciók ugyanazt az információt különböző térbeli helyeken újra felhasználják).

Hasonlóképpen, a mély tanulási modell-architektúra elsősorban a modularitás, a hierarchia és az újrahasználat okos felhasználásáról szól. Észre fogja venni, hogy az összes népszerű convnet architektúra nem csak rétegekbe, hanem ismétlődő rétegcsoportokba (ezek „blokkok” vagy „modulok”) vannak strukturálva. Például az előző fejezetben használt népszerű VGG16 architektúra ismétlődő „conv, conv, max pooling” blokkokból áll (lásd a 9.8. ábrát).

Ezenkívül a legtöbb convnet gyakran piramisszerű struktúrákat (szolgáltatáshierarchiákat) tartalmaz. Emlékezzünk vissza például az előző fejezetben felépített első convnetben használt konvolúciós szűrők számának alakulására: 32, 64, 128. A szűrők száma a rétegmélységgel nő, míg a jellemzőtérképek mérete ennek megfelelően csökken. Ugyanezt a mintát láthatja a VGG16 modell blokkjaiban (lásd a 9.8. ábrát).

![](figs/f9.8_.jpg)

**9.8 ábra:** A VGG16 architektúra: vegye észre az ismétlődő rétegblokkokat és a jellemzőtérképek piramisszerű szerkezetét

A mélyebb hierarchiák alapvetően jók, mert ösztönzik a funkciók újrafelhasználását, és ezáltal az absztrakciót. Általánosságban elmondható, hogy a keskeny rétegekből álló mély halom jobban teljesít, mint a nagy rétegek sekély halmaza. Az eltűnő színátmenetek problémája miatt azonban korlátozottak a rétegek egymásra halmozásának mélységei. Ez elvezet bennünket az első lényeges modellarchitektúránkhoz: a maradék kapcsolatokhoz.

---

**Az ablációs vizsgálatok fontosságáról a mély tanulás kutatásában**

A mélytanulási architektúrák gyakran inkább _kifejlődtek_, minthogy megtervezték volna – úgy fejlesztették ki őket, hogy többször próbáltak dolgokat, és kiválasztották azt, ami működni látszott. A biológiai rendszerekhez hasonlóan, ha bármilyen bonyolult kísérleti mélytanulási beállítást választ, nagy eséllyel eltávolíthat néhány modult (vagy néhány betanított funkciót véletlenszerűre cserélhet) anélkül, hogy veszítene a teljesítményéből.

Ezt tovább rontják azok az ösztönzők, amelyekkel a mély tanulással foglalkozó kutatók szembesülnek: ha egy rendszert a szükségesnél bonyolultabbá tesznek, érdekesebbé vagy újszerűbbé tehetik azt, és így növelhetik az esélyeiket, hogy a szakértői értékelési folyamaton átjussanak. Ha sok mélyreható tanulmányt elolvas, észre fogja venni, hogy azok gyakran vannak optimalizálva szakértői értékeléshez mind stílusban, mind tartalomban oly módon, hogy az aktívan sérti a magyarázatok egyértelműségét és az eredmények megbízhatóságát. Például a matematikát a mélytanulási dolgozatokban ritkán használják fogalmak egyértelmű formába öntésére vagy nem nyilvánvaló eredmények levezetésére – inkább a _komolyság jelzéseként_ használják, mint a drága öltönyt az eladón.

A kutatás célja ne pusztán publikálás legyen, hanem megbízható tudás létrehozása. Létfontosságú, hogy az ok-okozati összefüggések megértése a rendszerben a legegyszerűbb módja a megbízható tudás létrehozásának. És van egy nagyon kevés erőfeszítést igénylő módszer az ok-okozati összefüggés vizsgálatára: az _ablációs_ vizsgálatok. Az ablációs vizsgálatok abból állnak, hogy szisztematikusan megpróbálják eltávolítani a rendszer egyes részeit – ezzel egyszerűbbé téve azt – annak azonosítására, hogy honnan származik a teljesítménye. Ha úgy találja, hogy az X + Y + Z jó eredményeket ad, próbálkozzon X, Y, Z, X + Y, X + Z és Y + Z esettel is, és nézze meg, mi történik.

Ha mélytanulás-kutató lesz, vágja le a kutatási folyamat zaját: végezzen ablációs vizsgálatokat modelljeihez. Mindig kérdezze meg: „Létezhet-e egyszerűbb magyarázat? Valóban szükséges ez a megnövelt bonyolultság? Miért?"

---

### 9.3.2 Maradék összefüggés

Valószínűleg ismeri a Telephone játékot, amelyet az Egyesült Királyságban kínai suttogásnak, Franciaországban pedig téléphone arabe-nak is neveznek, ahol a kezdeti üzenetet az egyik játékos fülébe suttogják, aki aztán a következő játékos fülébe súgja, és így tovább. Az utolsó üzenet végül alig hasonlít az eredeti verzióhoz. Ez egy szórakoztató metafora a zajos csatornán keresztüli szekvenciális átvitel során fellépő kumulatív hibákra.

Ahogy megtörténik, a szekvenciális mély tanulási modellben a visszaterjesztés, az nagyon hasonlít a Telefonos játékhoz. Van egy függvényláncunk, például ez:
```
y = f4(f3(f2(f1(x))))
```
A játék neve a lánc egyes függvényei paramétereinek beállítása az f4 kimenetén rögzített hiba (a modell vesztesége) alapján. Az f1 beállításához a hibainformációknak az f2-n, az f3-on és az f4-en keresztül kell átadnia. Mindazonáltal a láncban minden egymást követő függvény bizonyos mennyiségű zajt hoz be. Ha a függvénylánc túl hosszú, ez a zaj elkezdi elnyomni a gradiens információkat, és a visszaterjesztés leáll. A modelled egyáltalán nem fog tanulni. Ez az _eltűnő gradiensek_ problémája.

A javítás egyszerű: csak kényszerítse rá a lánc mindegyik függvényét, hogy legyen roncsolásmentes – hogy megőrizze az előző bemenetben található információ zajtalan változatát. Ennek legegyszerűbb módja a _maradék kapcsolat_ használata. Ez nagyon egyszerű: csak adja hozzá egy réteg vagy rétegblokk bemenetét a kimenethez (lásd a 9.9. ábrát). A maradék kapcsolat _információ gyorsításként_ működik a destruktív vagy zajos blokkok (például a `relu`-aktiválást vagy kihagyási rétegeket tartalmazó blokkok) körül, lehetővé téve a korai rétegekből származó hibagradiens információinak zajtalan terjedését egy mély hálózaton. Ezt a technikát 2015-ben vezették be a ResNet modellcsaláddal (amelyet He et al. fejlesztett ki a Microsoftnál).[1]

---

[1] Kaiming He et al., “Deep Residual Learning for Image Recognition,” Conference on Computer Vision and Pattern Recognition (2015), https://arxiv.org/abs/1512.03385 .

![](figs/f9.9_.jpg)

**9.9. ábra:** Egy feldolgozási blokk körüli maradék kapcsolat

A gyakorlatban a maradék kapcsolatot a következőképpen valósíthatjuk meg.

**9.1 lista: A maradék kapcsolat pszeudokódban**


```python
x = ...                 #<--- Valami bemeneti tenzor
residual = x            #<--- Mentsen el egy mutatót az eredeti bemenetre. Ezt hívjuk maradéknak.
x = block(x)            #<--- Ez a számítási blokk potenciálisan romboló vagy zajos lehet, és ez rendben van.
x = add([x, residual])  #<--- Hozzáadja az eredeti bemenetet a réteg kimenetéhez:
                        #     a végső kimenet így mindig megőrzi a teljes információt az eredeti bemenetről.
```

Vegye észre, hogy a bemenet hozzáadása egy blokk kimenetéhez azt jelenti, hogy a kimenetnek ugyanolyan alakúnak kell lennie, mint a bemenetnek. Ez azonban nem így van, ha a blokk konvolúciós rétegeket tartalmaz megnövelt számú szűrővel, vagy max pooling réteget. Ilyen esetekben használjon 1 × 1 `Conv2D` réteget aktiválás nélkül, hogy a maradékot lineárisan a kívánt kimeneti alakra vetítse (lásd a 9.2 listát). Általában a `padding=` `"same"` értéket használunk a célblokk konvolúciós rétegeiben, hogy elkerüljük a kitöltés miatti térbeli lemintavételezést, a maradék vetítésben pedig a lépésköz használatával illeszkedünk a max pooling réteg által okozott lemintavételhez (lásd a 9.3 listát).

**9.2 lista: Maradék blokk, ahol a szűrők száma változik**


```python
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation="relu")(inputs)
residual = x                                #<--- Tegyük félre a maradékot.
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)  #<--- Ez az a réteg, amely körül maradék kapcsolatot hozunk létre:
                                                                #     ez 32-ről 64-re növeli a kimeneti file-ok számát.
                                                                #     Vegye figyelembe, hogy a padding="same"-t használjuk,
                                                                #     hogy elkerüljük a padding miatti lemintavételezést.
residual = layers.Conv2D(64, 1)(residual)   #<--- A maradéknak csak 32 szűrője volt, ezért 1 × 1-es Conv2D-t használunk
                                            #     a megfelelő formára vetítéséhez.
x = layers.add([x, residual])               #<--- Most a blokk kimenete és a maradék azonos alakú, és összeadható.
```

**9.3 lista: Az az eset, amikor a célblokk max pooling réteget tartalmaz**


```python
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation="relu")(inputs)
residual = x                                #<--- Tegyük félre a maradékot.
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)  #<--- Ez az a két rétegből álló blokk, amely körül maradék kapcsolatot
x = layers.MaxPooling2D(2, padding="same")(x)                   #     hozunk létre: tartalmaz egy 2 × 2 max. pooling réteget.
                                                                #     Vegye figyelembe, hogy mind a konvolúciós rétegben,
                                                                #     mind a max pooling rétegben a padding="same" értéket használjuk,
                                                                #     hogy elkerüljük a padding miatti lemintavételezést.
residual = layers.Conv2D(64, 1, strides=2)(residual)            #<--- A maradék vetületben strides=2-t használunk, hogy megfeleljen a
                                                                #     max pooling réteg által létrehozott lemintavételezésnek.
x = layers.add([x, residual])               #<--- Most a blokk kimenete és a maradék azonos alakú, és összeadható.
```

Hogy konkrétabbá tegyük ezeket az elképzeléseket, íme egy példa egy egyszerű convnet-re, amely egy sor blokkból áll, amelyek mindegyike két konvolúciós rétegből és egy opcionális max pooling rétegből áll, mindegyik blokk körül egy maradék kapcsolattal:


```python
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Rescaling(1./255)(inputs)

def residual_block(x, filters, pooling=False):    #<--- Segédfüggvény a konvolúciós blokk alkalmazásához maradék kapcsolattal,
                                                  #     max pooling hozzáadásának lehetőségével
    residual = x
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    if pooling:
        x = layers.MaxPooling2D(2, padding="same")(x)
        residual = layers.Conv2D(filters, 1, strides=2)(residual) #<--- Ha max poolingot használunk, akkor lépésközös konvolúciót
                                                                  #     adunk hozzá, hogy a maradékot a kívánt alakra vetítsük.
    elif filters != residual.shape[-1]:
        residual = layers.Conv2D(filters, 1)(residual)  #<--- Ha nem használunk max poolingot, akkor
                                                        #     csak akkor vetítjük ki a maradékot, ha a csatornák száma megváltozott.
    x = layers.add([x, residual])
    return x

x = residual_block(x, filters=32, pooling=True)     #<--- Első blokk
x = residual_block(x, filters=64, pooling=True)     #<--- Második blokk; vegye észre a növekvő szűrőszámot mindegyik blokkban.
x = residual_block(x, filters=128, pooling=False)   #<--- Az utolsó blokkhoz nincs szükség max pooling rétegre,
                                                    #     mivel közvetlenül utána alkalmazzuk a globális átlag poolingot.

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
```

Ez a modell összefoglalója, amit kapunk:

```
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 32, 32, 3)]  0
__________________________________________________________________________________________________
rescaling (Rescaling)           (None, 32, 32, 3)    0           input_1[0][0]
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 32, 32, 32)   896         rescaling[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 32, 32)   9248        conv2d[0][0]
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 16, 16, 32)   0           conv2d_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 16, 16, 32)   128         rescaling[0][0]
__________________________________________________________________________________________________
add (Add)                       (None, 16, 16, 32)   0           max_pooling2d[0][0]
                                                                 conv2d_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 16, 16, 64)   18496       add[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 16, 16, 64)   36928       conv2d_3[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 8, 8, 64)     0           conv2d_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 8, 8, 64)     2112        add[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 8, 8, 64)     0           max_pooling2d_1[0][0]
                                                                 conv2d_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 8, 8, 128)    73856       add_1[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 8, 8, 128)    147584      conv2d_6[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 8, 8, 128)    8320        add_1[0][0]
__________________________________________________________________________________________________
add_2 (Add)                     (None, 8, 8, 128)    0           conv2d_7[0][0]
                                                                 conv2d_8[0][0]
__________________________________________________________________________________________________
global_average_pooling2d (Globa (None, 128)          0           add_2[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 1)            129         global_average_pooling2d[0][0]
==================================================================================================
Total params: 297,697
Trainable params: 297,697
Non-trainable params: 0
__________________________________________________________________________________________________
```
A maradék kapcsolatokkal tetszőleges mélységű hálózatokat építhetünk ki anélkül, hogy aggódni kellene az eltűnő gradiensek miatt.

Most térjünk át a következő alapvető convnet architektúra mintára: a _köteg(elt) normalizálásra_.

### 9.3.3 Kötegelt normalizálás

A _normalizálás_ azon módszerek tág kategóriája, amelyek arra törekszenek, hogy a gépi tanulási modellnek megmutatott különböző minták jobban hasonlítsanak egymásra, ami segít a modellnek megtanulni és jól általánosítani az új adatokat. Az adatnormalizálás legáltalánosabb formája az, amelyet ebben a könyvben már többször is láthattunk: az adatok nullára való központosítása úgy, hogy az átlagot kivonjuk az adatokból, és egységnyi szórást adunk az adatokhoz úgy, hogy az adatokat elosztjuk a szórással. Valójában ez azt feltételezi, hogy az adatok normál (vagy Gauss-) eloszlást követnek, és biztosítja, hogy ez az eloszlás legyen központosítva a 0-ra és átméretezve az egységnyi szórásra:
```
normalized_data = (data - np.mean(data, axis=...)) / np.std(data, axis=...)
```
A könyv korábbi példái normalizálták az adatokat, mielőtt betáplálták őket a modellekbe. Az adatok normalizálása azonban érdekes lehet minden hálózat által végrehajtott átalakítás után: még ha a `Dense` vagy `Conv2D` hálózatba belépő adatok 0 átlaggal és egységnyi szórással rendelkeznek is, nem lehet eleve elvárni, hogy ez az kiérkező adatok esetében is így legyen. A köztes aktiválások normalizálása segíthet?

A kötegelt normalizálás éppen ezt teszi. Ez is egyfajta réteg (a `BatchNormalization` a Kerasban), amelyet 2015-ben vezetett be az Ioffe és Szegedy;[2] ez képes adaptívan normalizálni az adatokat, még akkor is, ha az átlag és a variancia idővel változik a tanítás során. A betanítás során az aktuális adatköteg átlagát és szórását használja a minták normalizálására, a következtetés során pedig (amikor esetleg nem áll rendelkezésre elég nagy köteg reprezentatív adatokból) a képzés során látott adatok kötegenkénti átlagának és szórásának exponenciális mozgóátlagát használja. {256.o:->}

---

[2] Sergey Ioffe and Christian Szegedy, “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,” Proceedings of the 32nd International Conference on Machine Learning (2015), https://arxiv.org/abs/1502.03167.

Bár az eredeti dokumentum azt állította, hogy a kötegelt normalizálás a „belső kovariáns eltolódás csökkentésével” működik, senki sem tudja biztosan, miért segít a kötegelt normalizálás. Vannak különféle hipotézisek, de nincsenek bizonyosságok. Meg fogod tapasztalni, hogy ez sok mindenre igaz a mély tanulásban – a mélytanulás nem egzakt tudomány, hanem állandóan változó, empirikusan levezetett bevált mérnöki gyakorlatok halmaza, amelyeket megbízhatatlan narratívák fűznek össze. Néha úgy fogod érezni, hogy a kezedben lévő könyv megmondja, _hogyan_ kell csinálni valamit, de nem mondja meg kielégítően, hogy _miért_ működik: ez azért van, mert tudjuk, hogyan kell, de nem tudjuk, hogy miért. Ha megbízható magyarázat áll rendelkezésre, mindenképpen megemlítem. A kötegelt normalizálás nem tartozik ezek közé az esetek közé.

A gyakorlatban úgy tűnik, hogy a kötegelt normalizálás fő hatása az, hogy segíti a gradiens terjedését – hasonlóan a maradék kapcsolatokhoz –, és így mélyebb hálózatokat tesz lehetővé. Egyes nagyon mély hálózatokat csak akkor lehet betanítani, ha több `BatchNormalization` réteget tartalmaznak. A kötegelt normalizálást például bőségesen használják a Kerashoz csomagolt fejlett convnet architektúrákban, mint például a ResNet50, az EfficientNet és az Xception.

A `BatchNormalization` réteg bármely réteg után használható – `Dense, Conv2D` stb.:
```
x = ...
x = layers.Conv2D(32, 3, use_bias=False)(x) #<--- Mivel a Conv2D réteg kimenete normalizálódik,
                                            #     a rétegnek nincs szüksége saját torzítási vektorra.
x = layers.BatchNormalization()(x)
```

MEGJEGYZÉS
>Mind a `Dense`, mind a `Conv2D` tartalmaz torzítási (eltolási) vektort, egy tanult változót, amelynek az a célja, hogy a réteget affinná, nem pedig tisztán lineárissá tegye. Például a `Conv2D` vázlatosan az `y = conv(x, kernel) + bias`-t adja vissza, a `Dense` pedig az `y = pont(x, kernel) + bias`-t. Mivel a normalizálási lépés gondoskodik arról, hogy a réteg kimenet központja nullára kerüljön, az eltolási vektorra már nincs szükség `BatchNormalization` használatakor, és a réteg nélküle is létrehozható a `use_bias=False` paraméterrel. Ettől a réteg kissé karcsúbb lesz.

Fontos, hogy általában azt javaslom, hogy az előző réteg aktiválását a kötegelt normalizálási réteg után helyezze el (bár ez még mindig vita tárgya). Tehát ahelyett, hogy azt tenné, ami a 9.4-es listában látható, inkább a 9.5-ös listában leírtakat tegye.

**9.4. lista: Hogyan ne használjuk a kötegelt normalizálást**


```python
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.BatchNormalization()(x)
```

**9.5. lista: Így használjuk a kötegelt normalizálást: az aktiválás az utolsó**


```python
x = layers.Conv2D(32, 3, use_bias=False)(x)   #<--- Vegye észre az aktiválás hiányát.
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)    #<--- Az aktiválást a BatchNormalization réteg után helyezzük el.
```

Ennek a megközelítésnek az intuitív oka, hogy a kötegelt normalizálás a bemeneteket nullára állítja, míg a `relu` aktiválása nullát használ az aktivált csatornák megtartásához vagy eldobásához: az aktiválás előtti normalizálás maximalizálja a `relu` kihasználását. Ennek ellenére ez a sorrendre bevált gyakorlat nem éppen kritikus, így ha konvolúciót, majd aktiválást, majd kötegelt normalizálást végez, a modell továbbra is betanítható, és nem feltétlenül fog rosszabb eredményeket kapni.

---

**A kötegelt normalizálásról és finomhangolásról**

>A kötegelt normalizálásnak számos furcsasága van. Az egyik fő ilyen a finomhangoláshoz kapcsolódik: `BatchNormalization` rétegeket tartalmazó modell finomhangolásakor azt javaslom, hogy ezeket a rétegeket hagyjuk fagyasztva (a `trainable` attribútumot állítsuk `False` értékre). Ellenkező esetben folyamatosan frissítik belső átlagukat és szórásukat, ami megzavarhatja a környező `Conv2D` rétegekre alkalmazott nagyon kis frissítéseket.

---

Most pedig vessünk egy pillantást sorozatunk utolsó szerkezeti mintájára: a mélységben szétválasztható konvolúcióra.

### 9.3.4 Mélységben szétválasztható konvolúciók

Mi lenne, ha azt mondanám, hogy van egy réteg, amelyet a `Conv2D` helyettesítőjeként használhatsz, és amely kisebbé (kevesebb betanítható súlyparaméter) és karcsúbbá (kevesebb lebegőpontos művelet) teszi a modelledet, és néhány százalékponttal jobban teljesíti a feladatát? Pontosan ezt teszi a _mélységben szétválasztható konvolúciós_ réteg (a `SeparableConv2D` a Kerasban). Ez a réteg minden bemeneti csatornán térbeli konvolúciót hajt végre, függetlenül, mielőtt a kimeneti csatornákat pontszerű konvolúcióval (1 × 1 konvolúcióval) összekeverné, amint az a 9.10. ábrán látható.

![](figs/f9.10_.jpg)

**9.10. ábra:** Mélységben szétválasztható konvolúció: mélységi konvolúció, majd pontszerű konvolúció


