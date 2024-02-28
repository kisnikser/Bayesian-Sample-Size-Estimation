<table>
    <tr>
        <td align="center"> <b> Название исследуемой задачи </b> </td>
        <td> Байесовский подход к выбору достаточного размера выборки </td>
    </tr>
    <tr>
        <td align="center"> <b> Тип научной работы </b> </td>
        <td> НИР </td>
    </tr>
    <tr>
        <td align="center"> <b> Автор </b> </td>
        <td> Киселев Никита Сергеевич </td>
    </tr>
    <tr>
        <td align="center"> <b> Научный руководитель </b> </td>
        <td> к.ф.-м.н. Грабовой Андрей Валериевич </td>
    </tr>
</table>

# Аннотация

Исследуется задача выбора достаточного размера выборки. Рассматривается проблема определения достаточного размера выборки без постановки статистической гипотезы о распределении параметров модели. Предлагаются два подхода к определению достаточного размера выборки по значениям функции правдоподобия на подвыборках с возвращением. Эти подходы основываются на эвристиках о поведении функции правдоподобия при большом количестве объектов в выборке. Предлагаются два подхода к определению достаточного размера выборки на основании близости апостериорных распределений параметров модели на схожих подвыборках. Доказывается корректность представленных подходов при определенных ограничениях на используемую модель. Доказывается теорема о моментах предельного апостериорного распределения параметров в модели линейной регрессии. Предлагается метод прогнозирования функции правдоподобия в случае недостаточного размера выборки. Проводится вычислительный эксперимент для анализа свойств предложенных методов.

# Установка

Чтобы повторить результаты вычислительного эксперимента, рекомендуется установить все необходимые зависимости.
Файл ``requirements.txt`` находится в директории ``code``.
Для установки

- Сделайте ``git clone`` этого репозитория.
- Создайте новое ``conda`` окружение и активируйте его.
- Запустите ``pip install -r requirements.txt``.


# Демонстрация работы

Код со всеми проведенными вычислительными экспериментами [здесь](<https://github.com/kisnikser/Bayesian-Sample-Size-Estimation/blob/main/code/main.ipynb>).