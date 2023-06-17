#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Воспоминания о загрузке данных
# -----------------------------------------------------

# In[2]:


import numpy as np

names = ("id", "n", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type")
data = np.genfromtxt('glass.csv', delimiter=',', names=names)
print(data[0:5])


# In[3]:


import pandas as pd

data = pd.read_csv('glass.csv', header=None, names=names, index_col=0)
data.head()


# In[4]:


import pandas as pd

data = pd.read_csv('concrete.csv')
data.head()


# Воспоминания о работе с массивами `numpy`
# ----------------------------------------------------

# In[5]:


import numpy as np


# In[6]:


x = np.array([1,2,3,5,8], dtype=np.float32)
print(x)
print(x.shape)


# In[7]:


x = np.array([1,2,3,5,8], dtype=np.int8)
print(x)
print(x.shape)


# In[8]:


x = np.array([[0,-1],[1,0]], dtype=np.float32)
print(x)
print(x.shape)


# In[9]:


x = np.array([[0,-1],[1,0]], dtype=np.float32)
z = np.array([1.0,0], dtype=np.float32)
print(x @ z)
print(x.dot(z))


# In[10]:


x = np.arange(10)
print(x)


# In[11]:


x = np.arange(10, -1, -1)
print(x)


# In[12]:


x = np.linspace(0, 2, 7)
print(x)


# In[13]:


z = np.zeros((3, 2), dtype=np.float32)
print(z)
print()

o = np.ones((3, 2), dtype=np.float32)
print(o)
print()

t = np.full(z.shape, 2, dtype=np.float32)
print(t)
print()

t = np.full_like(z, 2, dtype=np.float32)
print(t)


# ### Размерности массива

# In[14]:


x = np.arange(24)
print(x)
print(x.shape)
print()

x = x.reshape(4,6)
print(x)
print(x.shape)
print()

x = x.reshape(2,3,4)
print(x)
print(x.shape)
print()


# In[15]:


x = np.arange(5)
print(x)
print(x.shape)
print()

x = x.reshape(5,1)
print(x)
print(x.shape)
print()

x = x.reshape(1,5)
print(x)
print(x.shape)
print()


# In[16]:


x = np.arange(10)
print(x)
print(x.shape)
print()

x = x.reshape(-1,1)
print(x)
print(x.shape)
print()

x = x.reshape(-1,2)
print(x)
print(x.shape)
print()


# In[17]:


x = np.array([[1,2,3],[3,4,5]])
z = np.array([-1,0,2])
print(x)
print(z)
print()
print(x * z)
print()

x = np.array([[1],[2],[3]])
print(x)
print(z)
print()
print(x * z)

print(np.broadcast(x, z).shape)


# ### Срезы

# In[18]:


#x = np.linspace(0, 48, 16)
x = np.arange(16)
print(x)
print()
print("1.  x[1]        = {}".format(x[1]))
print("2.  x[-2]       = {}".format(x[-2]))
print("3.  x[2:9]      = {}".format(x[2:9]))
print("4.  x[:9]       = {}".format(x[:9]))
print("5.  x[2:]       = {}".format(x[2:]))
print("6.  x[:]        = {}".format(x[:]))
print("7.  x[2:9:3]    = {}".format(x[2:9:3]))
print("8.  x[9:2:-1]   = {}".format(x[9:2:-1]))
print("9.  x[::-1]     = {}".format(x[::-1]))
print("10. x[:-1]      = {}".format(x[:-1]))
print("11. x[-9:-2]    = {}".format(x[-9:-2]))
print("12. x[-9:-2:2]  = {}".format(x[-9:-2:2]))
print("13. x[-2:-9:-1] = {}".format(x[-2:-9:-1]))
print("14. x[-2:-9:-2] = {}".format(x[-2:-9:-2]))


# Воспоминания о работе со случайными числами
# -------------------------------------------

# In[19]:


x = np.arange(24)
x = x.reshape(2,3,4)
print(x)
print()
print("1. x[1] =\n{}".format(x[1]))
print("2. x[1,:,:] =\n{}".format(x[1,:,:]))
print("3. x[:,1,:] =\n{}".format(x[:,1,:]))
print("4. x[:,1] =\n{}".format(x[:,1]))
print("5. x[:,:,1] =\n{}".format(x[:,:,1]))
print("6. x[:,:2,:] =\n{}".format(x[:,:2,:]))
print("7. x[:,:,::-1] =\n{}".format(x[:,:,::-1]))


# In[20]:


import numpy as np
import numpy.random


# In[21]:


numpy.random.seed(42)

x = numpy.random.normal(2.0, 1.0, 1000)
print(x.shape)
print(np.mean(x))
print(np.var(x))
print()

x = numpy.random.normal(np.arange(4).reshape(2,2), 1.0, (1000,2,2))
print(x.shape)
print(np.mean(x, axis=0))
print(np.var(x, axis=0))
print()

x = numpy.random.normal(3.0, 1.0, 10000)
_ = plt.hist(x,25)


# In[22]:


numpy.random.seed(42)

x = numpy.random.poisson(6.0, 10000)
print(np.mean(x))
print(np.var(x))


# Метод наименьших квадратов
# ==========================

# **Задание 1.1**
# 
# Дан вектор `x` с равномерной сеткой значений от `0.0` до `10.0`, а так же константы `a` и `b`. В вектор `y` запишите реализацию случайного вектора удовлетворяющего следующему соотношению:
# 
# $$y_i = a  x_i + b + \epsilon_i,$$
# 
# где $\epsilon_i$ - независимые нормальные случайные величины с нулевым средним и стандартным отклонением $\sigma=2$.
# 
# Постройте график зависимости $y_i$ от $x_i$.

# In[23]:


numpy.random.seed(42)

x = np.linspace(0, 10, 50)
a, b = 2.5, 5.25
sigma = 2

## формулу записать сюда
y = a*x + b + np.random.normal(loc=0, scale=sigma, size=len(x))

plt.scatter(x,y)


# **Задание 1.2**
# 
# Используя вектора `x` и `y` из предыдущего задания и формулы метода наименьших квадратов из лекций, найдите оценки констант `a` и `b` для линейной модели.
# 
# Для этого используйте матричный формализм и векторные операции. Вычислите чему равна матрица задачи `A`. Для решения системы линейных алгебраических уравнений
# 
# $$(A^T A)\theta = A^T {\bf y}$$
# 
# используйте метод `np.linalg.solve`.
# Вычислите квадрат нормы вектора невязки используя метод `np.linalg.norm`. С его помощью вычислите ковариационную матрицу вектора параметров $\theta$:
# 
# $$\Sigma = \frac{(A^T A)^{-1} \left|\left|{\bf y} - A{\bf x}\right|\right|^2}{n -m},$$
# 
# где $n$ - количество измерений, $m$ - количество параметров модели. Для обращения матрицы можно использовать метод `np.linalg.inv`.
# 
# Нанесите на график данные и модельную прямую линию.

# In[24]:


import numpy.linalg

A = np.vstack([x, np.ones(len(x))]).T # матрица задачи
theta = np.linalg.solve(A.T @ A, A.T @ y) # оценка векторов параметров
res = np.linalg.norm(y - A @ theta)**2 / (len(y) - len(theta)) # квадрат нормы вектора невызки
theta_err = res * np.linalg.inv(A.T @ A) # ковариационная матрица
predicted = theta[0]*x + theta[1] # модельные значения y для сетки x


plt.scatter(x,y)
plt.plot(x, predicted, color='red')
print("a = {} +- {}".format(theta[0], np.sqrt(theta_err[0,0])))
print("b = {} +- {}".format(theta[1], np.sqrt(theta_err[1,1])))


# In[25]:


theta, res, _, _ = numpy.linalg.lstsq(A, y, rcond=None)
print(theta)


# In[26]:


import scipy
import scipy.linalg

theta, res, _, _ = scipy.linalg.lstsq(A, y)
print(theta)


# In[27]:


import sklearn
import sklearn.linear_model

r = sklearn.linear_model.LinearRegression()
r.fit(x.reshape(-1,1), y.reshape(-1,1))
predicted = r.predict(x.reshape(-1,1))
plt.scatter(x,y)
plt.plot(x, predicted, color='red')
print(np.hstack([r.coef_,[r.intercept_,]]))


# In[28]:


import statsmodels.api as sm

mod = sm.OLS(y, sm.add_constant(x))
res = mod.fit()
print(res.summary())


# **Задание 1.3**
# 
# В файле `glass.csv` хранится таблица в формате CSV, содержащая данные о химическом составе и показателе преломления $n$ нескольких марок стекла. Используя `pandas`, или любой удобный вам способ, загрузите эти данные и сформируйте массив `x` с "фичами" (содержанием каждого из девяти химических элементов в каждой марке стекла) и массив `y` содержащий значение показателя преломления $n$.
# 
# За отстутсвием луших идей будем считать, что для описания зависимости $n$ от химического состава можно использовать линейную модель. С использованием метода `sklearn.model_selection.train_test_split` раделите данные на учебную и тестовую выборки. Используя класс `sklearn.linear_model.LinearRegression` постройте линейную модель для описания зависимости.
# Подсчитайте среднеквадратичное отклонение модели для тестовой выборки и распечатайте его.
# Постройте график зависимости истинного значения $n$ от предсказанного модельню значения $n$ для тестовой выборки.

# In[29]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

names = ("id", "n", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type")

data = pd.read_csv("glass.csv", names=names, header=None)

x = data.iloc[:,2:-1].values # фичи
y = data.iloc[:,1].values # показатель преломления

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_test_pred = model.predict(x_test)

# квадратный корень из среднего квадрата ошибки модели на тестовой выборке
mse_test = mean_squared_error(y_test, y_test_pred)**0.5
print("RMSE on test set:", mse_test)

plt.scatter(y_test, y_test_pred)
plt.xlabel("True")
plt.ylabel("Predicted")
_ = plt.axis('equal')


# ### Представление о плохо обусловленных и некорректных задачах

# $$\alpha x^2 + \beta y^2 = 1$$

# In[30]:


from matplotlib.patches import Ellipse

def fit_ellipse(x, plot = False):
    X = x**2
    cond = numpy.linalg.cond(X)
    
    r = sklearn.linear_model.LinearRegression(fit_intercept=False)
    r.fit(X, np.ones((x.shape[0],1)))
    
    theta = np.asarray(r.coef_).reshape(-1)
    width, height = 2 * numpy.power(theta, np.full_like(theta, -0.5), out=np.full_like(theta, np.nan), where=theta>0)
    
    if plot:
        ellipse = Ellipse((0, 0), width=width, height=height, fill=False, edgecolor='red')
        plt.scatter(*x.T)
        plt.xlim(-2.5,2.5)
        plt.ylim(-2.5,2.5)
        plt.axis('equal')
        ax = plt.gca()
        ax.add_patch(ellipse)
    
    return width, height, cond

def generate_ellipse(phi, width, height, size):
    x = np.vstack([width / 2 * np.cos(phi), height / 2 * np.sin(phi)]).T
    X = x + np.random.normal(0.0, 0.0125, (size,*x.shape))
    return X
    


# In[31]:


numpy.random.seed(42)

width = 2.0
height = 4.0

lim = 7.5 / 180.0 * np.pi
phi = np.linspace(-lim,lim,4)
X = generate_ellipse(phi, width, height, 1000)

fit_ellipse(X[1], True)
fit_ellipse(X[3], True)

r = np.array([fit_ellipse(X[i]) for i in range(X.shape[0])])
print(np.nanmean(r, axis=0))
print(np.nanstd(r, axis=0))


# In[32]:


lim = 7.5 / 180.0 * np.pi
phi = np.linspace(-lim,lim,4)
phi = np.concatenate([phi, [1.5,]])
X = generate_ellipse(phi, width, height, 1000)

fit_ellipse(X[1], True)
fit_ellipse(X[3], True)

r = np.array([fit_ellipse(X[i]) for i in range(X.shape[0])])
print(np.nanmean(r, axis=0))
print(np.nanstd(r, axis=0))


# ### Представление о регуляризации

# Пусть нам кто-то сказал, что искомый эллипс очень похож на единичный круг:
# 
# $$\Delta \alpha x^2 + \Delta \beta y^2 = 1 - x^2 - y^2,$$
# 
# тогда $\Delta \alpha^2 + \Delta \beta^2 \rightarrow 0$.

# In[33]:


def fit_ellipse2(x, alpha, plot = False):
    X = x**2
    cond = numpy.linalg.cond(X)
    y = (1.0 - np.sum(x**2, axis=1)).reshape(-1)

    r = sklearn.linear_model.Ridge(fit_intercept=False, alpha=alpha)
    r.fit(X, y)
    
    theta = np.asarray(r.coef_).reshape(-1)
    width, height = 2 * numpy.power(theta + 1.0, np.full_like(theta, -0.5), out=np.full_like(theta, np.nan), where=theta>-1.0)
    
    if plot:
        ellipse = Ellipse((0, 0), width=width, height=height, fill=False, edgecolor='red')
        plt.scatter(*x.T)
        plt.xlim(-2.5,2.5)
        plt.ylim(-2.5,2.5)
        plt.axis('equal')
        ax = plt.gca()
        ax.add_patch(ellipse)
    
    return width, height, cond


# In[34]:


lim = 7.5 / 180.0 * np.pi
phi = np.linspace(-lim,lim,4)
X = generate_ellipse(phi, width, height, 1000)
alpha = np.geomspace(1e-6, 1e+3, 10)

for x in alpha:
    r = np.array([fit_ellipse2(X[i], x) for i in range(X.shape[0])])
    print("alpha = {}".format(x))
    print(np.nanmean(r, axis=0))
    print(np.nanstd(r, axis=0))
    print()


# **Задание 1.4** Логистическая регрессия.
# 
# В файле `magic04.csv` в формате `CSV` хранятся предварительно обработанные измерения эксперимента по ловле космического излучения в земной атмосфере с помощью излучения Черенкова. Используя `pandas`, или любой удобный вам способ, загрузите эти данные и сформируйте массив `x` с "фичами" (все которые найдутся в файле: это характеристики изображений, оставляемых отдельными событиями, на панорамном приемнике, такие как его размер, эллиптичность, и прочее...), и массив `y`, содержащий текстовый признак частицы: `"g"` - гамма квант, `"h"` - адрон.
# 
# Используя класс `sklearn.linear_model.LogisticRegression` постройте классификатор типа частицы по изображению с детектора.
# 
# С помощью обученной модели проделайте следующее:
# * с помощью метода `sklearn.metrics.confusion_matrix` постройте матрицу ошибок (confusion matrix),
# * рассчитайте AUC и постройте график ROC. вам помогут методы `sklearn.metrics.roc_curve`, `sklearn.metrics.roc_auc_score`, и метод `decision_function` объекта модели.

# In[35]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import sklearn.metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

names = ["length", "width", "size", "conc", "conc1", "asym", "m3long", "m3trans", "alpha", "dist", "class"]

data = pd.read_csv('magic04.csv')

x = data.iloc[:,:-1].values # фичи
y = data.iloc[:,-1].values # класс

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

c = LogisticRegression(random_state=42, solver="newton-cg")
c.fit(x_train, y_train.reshape(-1))

score_train = c.score(x_train, y_train)
score_test  = c.score(x_test, y_test)
print(score_train, score_test)

y_test_predicted = c.predict(x_test)
y_test_scores =  c.decision_function(x_test)

confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_test_predicted)

fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_test_scores, pos_label=c.classes_[-1])
auc = sklearn.metrics.roc_auc_score(y_test, y_test_scores)

plt.plot(fpr, tpr)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")


# Метод максимального правдоподобия
# ------------------------------------------------------------

# In[36]:


numpy.random.seed(42)
x = numpy.random.normal(1.5, 0.25, 1000)

mu_estimate = x.sum() / x.shape[0]
sigma2_estimate = ((x - mu_estimate)**2).sum() / x.shape[0]

mu_estimate, np.sqrt(sigma2_estimate)


# **Задание 1.5**
# 
# По аналогии с предыдущей ячейкой, сгенерируйте вектор `x` из $1000$ реализаций случайной величины, распределенной согласно закону Пуассона с параметром $\lambda = 2.45$.
# Используя формулу распределения Пуассона, а так же знания полученные на лекции, получите формулу оценки параметра $\lambda$ для выборки независимых величин. Рассчитайте по этой формуле значение оценки $\lambda$.

# In[37]:


numpy.random.seed(42)
x = np.random.poisson(2.45, 1000) # выборка случайной величины

lambda_estimate = x.mean() # оценка параметра

lambda_estimate


# **Задание 1.6**
# 
# Аналогично предыдущему заданию, сгенерируйте вектор `x` для лог-нормального распределения, с параметрами $\mu=1.5$, $\sigma=0.25$. Аналитически получите формулы оценок этих параметров по методу максимального правдоподобия, а затем проверьте себя вычислив числовые значения оценок параметров $\mu$ и $\sigma$.

# In[38]:


numpy.random.seed(42)
x = np.random.lognormal(1.5, 0.25, 1000) # выборка случайной величины

x_mean = x.mean()
x_var = x.var()

mu_estimate = np.log(x_mean) - 0.5 * np.log(x_var + x_mean**2) # оценки
sigma2_estimate = np.log(1 + x_var / x_mean**2) # параметров

mu_estimate, np.sqrt(sigma2_estimate)


# In[39]:


numpy.random.seed(42)
x = np.random.multivariate_normal([0.0, 0.0], [[1.0, 0.75],[0.75, 1.0]], 1000)
plt.plot(*x.T,'*')

mu_estimate = np.sum(x, axis=0) / x.shape[0]
dx = x - mu_estimate
sigma_estimate = np.tensordot(dx, dx, [0, 0]) / x.shape[0]

mu_estimate, sigma_estimate


# ### Линейные модели авторегрессии скользящего среднего

# In[40]:


import statsmodels.api as sm
from statsmodels import tsa
from statsmodels.tsa.statespace import sarimax
from statsmodels.graphics import tsaplots
from statsmodels.graphics import gofplots
from scipy.stats import boxcox

data = pd.read_csv('sunspots.csv')
x = np.asarray(data['YEAR']).reshape(-1,1)
y = np.asarray(data['SUNACTIVITY']).reshape(-1,1)

plt.plot(x, y)
plt.xlabel("Year")
plt.ylabel("Activity index")
plt.title("Solar activity")
plt.tight_layout()


# In[41]:


plt.figure()
gofplots.qqplot(y.reshape(-1), fit=True, line='45')
plt.axis('equal')

plt.figure()
y_log = boxcox(y+1, 0)
gofplots.qqplot(y_log.reshape(-1), fit=True, line='45')
_ = plt.axis('equal')

plt.figure()
y_1 = boxcox(y+1, 0.5)
gofplots.qqplot(y_1.reshape(-1), fit=True, line='45')
_ = plt.axis('equal')


# In[42]:


tsaplots.plot_acf(y_log, lags=np.arange(50))
_ = tsaplots.plot_pacf(y_log, lags=np.arange(50))


# In[43]:


arma_mod20 = sm.tsa.arima.ARIMA(y_log, order=(2,0,0)).fit()
print(arma_mod20.summary())

arma_mod30 = sm.tsa.arima.ARIMA(y_log, order=(3,0,0)).fit()
print(arma_mod30.summary())

arma_mods11 = sm.tsa.arima.ARIMA(y_log, order=(3,0,0), seasonal_order=(1,0,0,11)).fit()
print(arma_mods11.summary())


# In[44]:


plt.figure()
tsaplots.plot_acf(arma_mod20.resid, lags=np.arange(50))

plt.figure()
_ = tsaplots.plot_acf(arma_mod30.resid, lags=np.arange(50))

plt.figure()
_ = tsaplots.plot_acf(arma_mods11.resid, lags=np.arange(50))


# In[45]:


n_forecast = 40
x_pred = np.arange(x[-1], x[-1] + n_forecast)
y_pred_20 = arma_mod20.predict(len(x), len(x) + n_forecast - 1)
y_pred_s = arma_mods11.predict(len(x), len(x) + n_forecast - 1)

plt.plot(x, y_log)
plt.plot(x_pred, y_pred_20)
plt.plot(x_pred, y_pred_s)
plt.xlabel("Year")
plt.ylabel("Activity index")
plt.title("Solar activity")
plt.tight_layout()
plt.show()


# **Задание 1.7**
# 
# Используя данные о солнечной активности из файла `sunspots.csv` (по аналогии с предыдущими ячейками), найдите наиболее подходящую модель семейства ARIMA, опираясь на принцип максимального правдоподобия при сравнении моделей.
# * Руководствуясь принципом простоты, количество параметров модели не должно превышать 12
# * Попробуйте применить различные преобразования из семейства Бокса-Кокса перед обучением модели
# 
# Для найденной модели постройте график АКФ невязок, распечатайте `summary()`, и постройте график прогноза на ближайшие 40 точек в будущее.
# 
# *Бонусные баллы* полагаются тому, кто найдет удовлетворяющую условиям модель с наибольшим правдоподобием среди всей группы.

# In[51]:


data.head()


# ### Гауссовы процессы

# **Задание 1.8**
# 
# Ниже приведен пример кода использования гауссовых процессов для моделирования временных рядов. Используются всё те же данные солнечной активности. В этом задании можно получить *бонусные баллы*, для этого надо подобрать ядро преобразования `kernel` в примере таким образом, чтобы среднеквадратичная ошибка модели на тестовой выборке (посление 100 отсчетов по времени), стала лучше всех в группе.

# In[46]:


import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

data = pd.read_csv('sunspots.csv')
x = np.asarray(data['YEAR']).reshape(-1,1)
y = np.log(np.asarray(data['SUNACTIVITY']) + 1).reshape(-1,1)

x_train = x[:-100]
y_train = y[:-100]
x_test = x[-100:]
y_test = y[-100:]

kernel = 1 * ExpSineSquared(length_scale=2, periodicity=22, periodicity_bounds=(20,25)) + WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
gp.fit(x_train, y_train)
y_pred, y_std = gp.predict(x_test, return_std=True)
y_pred_lo = y_pred.reshape(-1) - y_std
y_pred_hi = y_pred.reshape(-1) + y_std

print("Kernel: {}".format(gp.kernel_))

plt.plot(x, y)
plt.plot(x_test, y_pred, color='red')
plt.fill_between(x_test.reshape(-1), y_pred_lo, y_pred_hi, alpha=0.3, color='red')
plt.xlabel("Year")
plt.ylabel("Activity index")
plt.title("Solar activity")
plt.tight_layout()
plt.show()

np.sqrt(mean_squared_error(y_test, y_pred))


# In[47]:


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

kernel = 1 * RBF() + 1 * ExpSineSquared(periodicity=11) + WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
gp.fit(x_train, y_train)
y_pred, y_std = gp.predict(x_test, return_std=True)
y_pred_lo = y_pred.reshape(-1) - y_std
y_pred_hi = y_pred.reshape(-1) + y_std

print("Kernel: {}".format(gp.kernel_))

plt.plot(x, y, '-')
plt.plot(x_test, y_pred, '*', color='red')
plt.xlabel("Year")
plt.ylabel("Activity index")
plt.title("Solar activity")
plt.tight_layout()
plt.show()

np.sqrt(mean_squared_error(y_test, y_pred))


# **Задание 1.9**
# 
# В файле `forestfires.csv` в формате CSV, находятся данные о лесных пожарах в некотором заповеднике.
# Ваша задача состоит в том, чтобы попробовать построить зависимость мощности пожара (выгоревшей площади `area`), от условной координаты парка (колонки `X` и `Y`).
# Неплохой идеей кажется использовать для моделирования двумерные гауссовы процессы, т.к. ожидается корреляция между соседними в пространстве областями.
# 
# Загрузите данные из файла: в вектор `x` колоки `X` и `Y`, в вектор `y` загрузите величину $\log(\mathtt{"area"} + 1)$
# 
# Используя класс `sklearn.gaussian_process.GaussianProcessRegressor`, постройте модель. Подберите ядро самостоятельно.
# 
# *Бонусные баллы* полагаются за наилучшую модель. Эффективность модели измеряется среднеквадратичной ошибкой на тестовой выборке.

# In[48]:


import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# загрузка данных
data = pd.read_csv('forestfires.csv')


x = data[['X', 'Y']].values
y = np.log(data['area'] + 1)

data.head()


# In[49]:


mse = mean_squared_error(y_test[:len(y_pred)], y_pred)
rmse = np.sqrt(mse)


# In[50]:


# разбиение на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# построение модели с различными ядрами
kernel_rbf = 1.0 * RBF(length_scale=1.0)
kernel_matern = 1.0 * Matern(length_scale=1.0, nu=1.5)
kernel_exp = 1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0)

gp_rbf = GaussianProcessRegressor(kernel=kernel_rbf, alpha=0.1, n_restarts_optimizer=10)
gp_matern = GaussianProcessRegressor(kernel=kernel_matern, alpha=0.1, n_restarts_optimizer=10)
gp_exp = GaussianProcessRegressor(kernel=kernel_exp, alpha=0.1, n_restarts_optimizer=10)

gp_rbf.fit(x_train, y_train)
gp_matern.fit(x_train, y_train)
gp_exp.fit(x_train, y_train)

# оценка качества моделей на тестовой выборке
y_pred_rbf = gp_rbf.predict(x_test)
y_pred_matern = gp_matern.predict(x_test)
y_pred_exp = gp_exp.predict(x_test)

mse_rbf = mean_squared_error(y_test, y_pred_rbf)
mse_matern = mean_squared_error(y_test, y_pred_matern)
mse_exp = mean_squared_error(y_test, y_pred_exp)

print('MSE (RBF kernel):', mse_rbf)
print('MSE (Matern kernel):', mse_matern)
print('MSE (ExpSineSquared kernel):', mse_exp)

