Данные:

https://www.kaggle.com/c/nlp-getting-started/overview

Задачи:

Написать инференс модели (которую вы обучили в прошлый раз) на spark structured streaming:
1) Вход:
 - тип socket source
 - данные json пример {"id": 1, "text": "some text"} 

2) Выход файловая система в файл для сабмита (mode="append", repartition(1))

P.S Пул ревест и пжлст сделайте подпапку со своей фамилией 

Технологии:

Spark / Scala / Java