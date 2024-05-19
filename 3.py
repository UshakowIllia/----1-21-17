import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from matplotlib.colors import ListedColormap


# Функція для візуалізації класифікатора
def visualize_classifier(classifier, X, y, title):
    # Визначаємо мінімальні та максимальні значення для кожної ознаки
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Створюємо сітку точок з кроком 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Використовуємо класифікатор для передбачення міток для кожної точки сітки
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Візуалізуємо поверхню прийняття рішень
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(('red', 'blue', 'lightgreen', 'gray')))

    # Відображаємо навчальні точки
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=ListedColormap(('red', 'blue', 'lightgreen', 'gray'))(idx),
                    marker='o', label=cl)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()


# Приклад даних для файлу 'data_multivar_nb.txt'
data = """3.1, 7.2, 0
4, 6.7, 0
2.9, 8, 0
5.1, 4.5, 1
6, 5, 1
5.6, 5, 1
3.3, 0.4, 2
3.9, 0.9, 2
2.8, 1, 2
0.5, 3.4, 3
1, 4, 3
0.6, 4.9, 3"""

# Запис даних у файл 'data_multivar_nb.txt'
with open('.venv/data_multivar_nb.txt', 'w') as f:
    f.write(data)

# Вхідний файл, який містить дані
input_file = '.venv/data_multivar_nb.txt'

# Завантаження даних із вхідного файлу
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Перший прогін
# Розбивка даних на навчальний та тестовий набори
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=3)

# Створення наївного байєсовського класифікатора
classifier1 = GaussianNB()
classifier1.fit(X_train1, y_train1)

# Прогнозування значень для тестових даних
y_test_pred1 = classifier1.predict(X_test1)

# Обчислення якості класифікатора
accuracy1 = 100.0 * (y_test1 == y_test_pred1).sum() / X_test1.shape[0]
print("Accuracy of the first classifier =", round(accuracy1, 2), "%")

# Візуалізація результатів роботи першого класифікатора
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
visualize_classifier(classifier1, X_test1, y_test1, "First Classifier")

# Другий прогін
# Розбивка даних на навчальний та тестовий набори (з іншим seed для random_state)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=7)

# Створення наївного байєсовського класифікатора
classifier2 = GaussianNB()
classifier2.fit(X_train2, y_train2)

# Прогнозування значень для тестових даних
y_test_pred2 = classifier2.predict(X_test2)

# Обчислення якості класифікатора
accuracy2 = 100.0 * (y_test2 == y_test_pred2).sum() / X_test2.shape[0]
print("Accuracy of the second classifier =", round(accuracy2, 2), "%")

# Візуалізація результатів роботи другого класифікатора
plt.subplot(1, 2, 2)
visualize_classifier(classifier2, X_test2, y_test2, "Second Classifier")
plt.show()

# Порівняння результатів
print("First classifier accuracy: ", round(accuracy1, 2), "%")
print("Second classifier accuracy: ", round(accuracy2, 2), "%")

# Виконання потрійної перехресної перевірки
num_folds = 3

accuracy_values = cross_val_score(classifier2, X, y, scoring='accuracy', cv=num_folds)
print("Cross-validated accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")

precision_values = cross_val_score(classifier2, X, y, scoring='precision_weighted', cv=num_folds)
print("Cross-validated precision: " + str(round(100 * precision_values.mean(), 2)) + "%")

recall_values = cross_val_score(classifier2, X, y, scoring='recall_weighted', cv=num_folds)
print("Cross-validated recall: " + str(round(100 * recall_values.mean(), 2)) + "%")

f1_values = cross_val_score(classifier2, X, y, scoring='f1_weighted', cv=num_folds)
print("Cross-validated F1: " + str(round(100 * f1_values.mean(), 2)) + "%")
