# **9. Haladó mély tanulás a gépi látáshoz**

Ez a fejezet ezekkel foglalkozik:
* A gépi látás különböző ágai: képosztályozás, képszegmentálás, tárgydetektálás
* Modern convnet architektúra minták: maradék kapcsolatok, köteg normalizálás, mélységben szétválasztható konvolúciók
* Technikák a convnet által tanultak megjelenítésére és értelmezésére

Az előző fejezetben először bemutattuk a gépi látás mély tanulását egyszerű modelleken (`Conv2D` és `MaxPooling2D` rétegek halmaza) és egy egyszerű használati eseten (bináris képosztályozás) keresztül. De a gépi látás többről szól, mint a képbesorolás! Ez a fejezet mélyebben belemerül a változatosabb alkalmazásokba és a legjobb haladó gyakorlatokba.

## 9.1 A három alapvető (számítógépes) gépi látási feladat

Eddig a képbesorolási modellekre koncentráltunk: egy kép bemegy, egy címke jön ki. „Ez a kép valószínűleg egy macskát tartalmaz; ezen a másikon valószínűleg egy kutya van." A képosztályozás azonban csak egy a gépi látás lehetséges mélytanulási alkalmazásai közül. Általában három alapvető gépi látási feladatot kell ismernie:


```python

```
