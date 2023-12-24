# **4. Ismerkedés a neurális hálózatokkal: Osztályozás és regresszió**

Ez a fejezet ezekkel foglalkozik:
* Az első példák a valós gépi tanulási munkafolyamatokra
* Osztályozási problémák kezelése vektoradatokon
* Folyamatos regressziós problémák kezelése vektoradatokon

Ez a fejezet arra szolgál, hogy elkezdje használni a neurális hálózatokat valódi problémák megoldására. A 2. és 3. fejezetben megszerzett ismereteit megszilárdítja, és a tanultakat három új feladatban alkalmazza, amelyek lefedik a neurális hálózatok három leggyakoribb használati esetét – a bináris osztályozást, a többosztályos osztályozást és a skaláris regressziót:
* Egy filmkritika pozitív vagy negatív besorolása (bináris osztályozás)
* Híradók osztályozása téma szerint (többosztályos osztályozás)
* Egy ház árának becslése ingatlanadatok alapján (skaláris regresszió)

Ezek a példák jelentik az első kapcsolatfelvételt a teljes körű gépi tanulási munkafolyamatokkal: megismerkedhet az adatok előfeldolgozásával, az alapvető modellarchitektúra elveivel és a modellértékeléssel. {96.o:}

**Osztályozási és regressziós szószedet**

Az osztályozás és a regresszió számos speciális kifejezést foglal magában. Néhányukkal már találkoztunk a korábbi példákban, és még többet fogunk látni belőlük a következő fejezetekben. Pontos, gépi tanulás-specifikus definícióik vannak, és ezeket ismernie kell:
* _Minta_ vagy _bemenet_ – Egy adatpont, amely bemegy a modellbe.
* _Előrejelzés_ vagy _kimenet_ – Ami kijön a modellből.
* _Cél_ – az igazság. Amit ideális esetben meg kellene jósolnia a modelljének egy külső adatforrás alapján.
* _Előrejelzési hiba_ vagy _veszteség érték_ – A modell előrejelzése és a cél közötti távolság mértéke.
* _Osztályok_ – Az osztályozási feladatban választható lehetséges címkék halmaza. Például a macska- és kutyaképek osztályozásakor a „kutya” és a „macska” a két osztály.
* _Címke_ – Az osztály megnevezése egy adott példányra az osztályozási problémában. Például, ha az 1234-es kép a „kutya” osztályba tartozik, akkor az  1234-es kép címkéje „kutya”.
* _Alapvető igazság_ vagy _magyarázatok_ – Egy adatkészlet összes cél értéke, amelyeket jellemzően emberek gyűjtenek össze.
* _Bináris osztályozás_ – Osztályozási feladat, ahol minden bemeneti mintát kizárólag két kategóriába kell besorolni.
* _Többosztályos osztályozás_ – Osztályozási feladat, ahol minden bemeneti mintát kettőnél több kategóriába kell besorolni: például kézzel írt számjegyek osztályozása.
* _Többcímkés osztályozás_ – Osztályozási feladat, ahol minden bemeneti mintához több címke is hozzárendelhető. Például egy adott kép macskát és kutyát is tartalmazhat, és a „macska” és a „kutya” címkével is meg kell jelölni. A képenkénti címkék száma általában változó.
* _Skalár regresszió_ – Olyan feladat, ahol a cél egy folytonos skalárérték. Jó példa erre a lakásárak előrejelzése: a különböző célárak egy folytonos teret alkotnak.
* _Vektor regresszió_ – Olyan feladat, ahol a cél folytonos értékek halmaza: például egy folytonos vektor. Ha több értékhez (például egy képen lévő határolókeret koordinátáihoz) képest végez regressziót, akkor vektoros regressziót hajt végre.
* _Mini-batch_ vagy _köteg_ – Minták egy kis halmaza (általában 8 és 128 között), amelyeket a modell egyidejűleg dolgoz fel. A minták száma gyakran 2 hatványa, hogy megkönnyítse a memóriakiosztást a GPU-n. A betanítás során egy mini köteggel számítanak ki egyetlen gradiens-süllyedés frissítést, amelyet a modell súlyaira alkalmaznak.

A fejezet végére képes lesz a neurális hálózatok segítségével egyszerű osztályozási és regressziós feladatokat kezelni vektoradatokon. Ezután készen áll arra, hogy az 5. fejezetben elkezdje a gépi tanulás elvibb, elméleti alapú megértését.

##4.1 Filmkritikák osztályozása: Példa bináris osztályozásra

A kétosztályos osztályozás vagy bináris osztályozás a gépi tanulási problémák egyik leggyakoribb fajtája. Ebben a példában meg fogjuk tanulni, hogy a filmértékeléseket hogyan pozitív vagy negatív kategóriába sorolja a vélemények szöveges tartalma alapján.


```python

```
