# Config.py

Конфигурационный файл, который показывает путь до модели и перенос её на девайс, путь, по которому будут записаны фотографии, максимальное количество распознаваемых конусов, по которым будет построено движение машины, время работы кода, угол обзора камеры и т.п.

# Img_processing.py

Файл с кодом, в котором написан алгоритм определения распознаваемого конуса по классам

![загрузка](https://user-images.githubusercontent.com/71214107/157867803-00b3b83e-35c1-4bf5-95b3-2ab447e43ce4.png)

Всего в симуляторе по мере движения машина может встретить 3 класса конуса: синий, оранжевый и большой красный. Картинки в симуляторе черно-белые.

Функция create_bitwise принимает на вход ограничивающую рамку, которая была распознана YOLOv5, затем накладывает на неё маску, которая оставляет только треугольник, где находится конус, убирая часть дороги органичивающей рамки.

Функция cone_classification принимает на вход полученный треугольник и выполняет алгорим классификации конусов. Синий конус имеет белый цвет по середине, в то время как оранжевый красный. По мере движения по картинке вниз белый становится синим, то есть яркость идёт вниз. В то время как у оранжевого яркость идёт наверх при переводе из чёрного в оранжевый. Это является ключевым признаком в определении цвета конуса. 

Функции contour и find_circle нужны для определения координат края скруглённого носа болида. Это необходимо для того, чтобы модель распознавала конусы по цветам только в тот момент, когда они полностью изображены на картинке, избегая те моменты, когда конус пропадает с камеры, его часть не попадает в кадр и алгоритм с переходом яркости работает неверно.

# Data collection.py

Главный файл, в написан код по взаимодействии с симулятором, написание алгоритма движения машины, запуска модели YOLOv5, классификации конусов и записи меток в текстовый файл.

# Results


https://user-images.githubusercontent.com/71214107/177034272-0be273a7-097e-4e56-ada4-cf866601f881.mp4

