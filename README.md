# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1 выполнил(а):
- Голованов Богдан Михайлович 
- ХИИ21
Отметка о выполнении заданий:

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Написать программы Hello World на Python и Unity.
- Задание 2.
- В разделе «ход работы» пошагово выполнить каждый пункт с описанием и примером реализации задачи по теме лабораторной работы.
- Задание 3.
- Изучить код на Python и ответить на вопросы.
- Выводы.
- ✨Magic ✨

## Цель работы
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.

## Задание 1
### Написать программы Hello World на Python и Unity.
- Для Python в отчете привести скриншоты с демонстрацией сохранения документа google.colab на свой диск с запуском программы, выводящей сообщение Hello World.
![Code_PhOvKECgUe](https://user-images.githubusercontent.com/75094394/192996638-3309424f-dd3d-4639-b18e-a6d5dd9e3c66.png)

- Для Unity в отчете привести скришноты вывода сообщения Hello World в консоль.
![Unity_XnZUy2z5ro](https://user-images.githubusercontent.com/75094394/192996621-4846b829-5773-4a48-9c82-5cd6aad31192.png)


## Задание 2
### Пошагово выполнить каждый пункт раздела "ход работы" с описанием и примерами реализации задач
Ход работы:
-  Произвести подготовку данных для работы с алгоритмом линейной регрессии. 10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. Данные преобразуются в формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.

```py

import numpy as np
import matplotlib.pyplot as plt

x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)

plt.scatter(x,y)

```
![Code_uNxykNiZIM](https://user-images.githubusercontent.com/75094394/193137734-8d444100-2abd-4a8b-94c6-408a70bfbe5f.png)

- Определите связанные функции. Функция модели: определяет модель линейной регрессии wx+b. Функция потерь: функция потерь среднеквадратичной ошибки. Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.

```py

def model(a, b, x):
    return a*x + b

def loss_function(a, b, x, y):
    num = len(x)
    prediction=model(a,b,x)
    return (0.5/num) * (np.square(prediction-y)).sum()

def optimize(a,b,x,y):
    num = len(x)
    prediction = model(a,b,x)
    da = (1.0/num) * ((prediction -y)*x).sum()
    db = (1.0/num) * ((prediction -y).sum())
    a = a - Lr*da
    b = b - Lr*db
    return a, b

def iterate(a,b,x,y,times):
    for i in range(times):
        a,b = optimize(a,b,x,y)
    return a,b

```
### Начать итерацию
- Шаг 1 Инициализация и модель итеративной оптимизации

```py

a = np.random.rand(1)
print(a)
b = np.random.rand(1)
print(b)
Lr = 0.000001

a,b = iterate(a,b,x,y,1)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```

![Code_X4lZJPp2OL](https://user-images.githubusercontent.com/75094394/193143156-7fdc4d9f-387b-48c8-a5ec-5397cfe9526a.png)

- Шаг 2 На второй итерации отображаются значения параметров, значения потерь и эффекты визуализации после итерации

```py

a,b = iterate(a,b,x,y,2)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```

![Code_xiGwUhAmtH](https://user-images.githubusercontent.com/75094394/193143125-bb7317ac-5b95-4960-90e6-b432eb5b7348.png)

- Шаг 3 Третья итерация показывает значения параметров, значения потерь и визуализацию после итерации

```py

a,b = iterate(a,b,x,y,3)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```

![Code_glx3o0sHpg](https://user-images.githubusercontent.com/75094394/193143080-2b3bd7d3-d44a-4cf7-8731-6191508d369f.png)

- Шаг 4 На четвертой итерации отображаются значения параметров, значенияпотерь и эффекты визуализации

```py

a,b = iterate(a,b,x,y,4)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```

![Code_5cBP3G3OLH](https://user-images.githubusercontent.com/75094394/193143040-73c4e33f-1b86-4e6c-b2a7-cce48d76e3ef.png)

- Шаг 5 Пятая итерация показывает значение параметра, значение потерь и эффект визуализации после итерации

```py

a,b = iterate(a,b,x,y,5)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```

![Code_9oDDSshbzO](https://user-images.githubusercontent.com/75094394/193143011-f52fff7d-65f2-4df3-9443-8c913d688048.png)

- Шаг 6 10000-я итерация, показывающая значения параметров, потери и визуализацию после итерации

```py

a,b = iterate(a,b,x,y,10000)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```

![Code_GuwBTy6e0k](https://user-images.githubusercontent.com/75094394/193142970-414108d5-4200-4e00-a90f-05457d3ac387.png)

## Задание 3
### Должна ли величина loss стремиться к нулю при изменении исходных данных? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ.
Да, должна, так как с каждой итерацией модель постепенно "обучается". А соответственно функция потерь стремится к нулю.

1 итерация

![Code_xiGwUhAmtH](https://user-images.githubusercontent.com/75094394/193143295-f276a302-84cd-4d3d-a907-159fd8e8f673.png)

1000 итерация 

![Code_GuwBTy6e0k](https://user-images.githubusercontent.com/75094394/193143330-adc8fea5-ce18-450c-be6f-286797a8620e.png)

### Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. В качестве эксперимента можете изменить значение параметра.

Lr - это learning rate, коэффициент скорости обучения. Чем больше этот параметр, тем быстрее модель обучается, а значит функция потерь меньше.

Lr = 0,000001
- Итерация 1

![Code_xiGwUhAmtH](https://user-images.githubusercontent.com/75094394/193143381-5e94452f-6524-448a-acc4-feda8f7e3548.png)

- Итерация 5

![Code_9oDDSshbzO](https://user-images.githubusercontent.com/75094394/193143424-6936a655-d723-4dbe-a1d8-6aac52f2acdd.png)


Lr = 0,0001
- Итерация 1

![Code_QJ2KVkuFGF](https://user-images.githubusercontent.com/75094394/193143727-90627428-a7a7-4cf5-bdb7-ff351258df3e.png)

- Итерация 2

![Code_iGPLcKk5Ef](https://user-images.githubusercontent.com/75094394/193143787-ef01c124-7b86-4344-94ce-29b7eca46be3.png)

Фнкция потерь заметно уменьшилась, уже на второй итерации, когда при Lr = 0,000001 изменений на 5 итерации не заметно.

## Выводы

В ходе лабароторной работы я установил необходимое программное обеспечение. Научился выводить в консоль Hello World. Я знакомилася с основными операторами языка Python на примере реализации линейной регрессии. Сделал выводы. При уменьшении исходных данных уменьшается величина потерь; параметр Lr отвечает за разницу значений после каждой итерации.

