from abc import ABC, abstractmethod
import math
import numpy as np
import matplotlib.pyplot as plt


# Итерфейс для любой случайной величины
class RandomVariable(ABC):
  @abstractmethod
  def pdf(self, x):
    pass

  @abstractmethod
  def cdf(self, x):
    pass

  @abstractmethod
  def quantile(self, alpha):
    pass


# Класс нормальной случайной величны
class NormalRandomVariable(RandomVariable):
  def __init__(self, location=0, scale=1) -> None:
    super().__init__()
    self.location = location
    self.scale = scale

  # Плотность вероятности в точке x
  def pdf(self, x):
    z = (x - self.location) / self.scale
    return math.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * self.scale) # нормальная ф-ия плотности

  # Интегральная функция распределения в точке x
  def cdf(self, x):
    z = (x - self.location) / self.scale
    if z <= 0:
      return 0.852 * math.exp(-math.pow((-z + 1.5774) / 2.0637, 2.34))
    return 1 - 0.852 * math.exp(-math.pow((z + 1.5774) / 2.0637, 2.34))

  # Квантиль уровня alpha
  def quantile(self, alpha):
    return self.location + 4.91 * self.scale * (math.pow(alpha, 0.14) - math.pow(1 - alpha, 0.14))


# Равномерное распределение
class UniformRandomVariable(RandomVariable):
  def __init__(self, location=0, scale=1) -> None:
    super().__init__()
    self.location = location
    self.scale = scale

  def pdf(self, x):
    if x >= self.location and x <= self.scale:
      return 1 / (self.scale - self.location)
    else:
      return 0

  def cdf(self, x):
    if x <= self.location:
      return 0
    elif x >= self.scale:
      return 1
    else:
      return (x - self.location) / (self.scale - self.location)

  def quantile(self, alpha):
    return self.location + alpha * (self.scale - self.location)


# Экспоненциальное распределение
class ExponentialRandomVariable(RandomVariable):
  def __init__(self, rate=1):
    self.rate = rate

  def pdf(self, x):
    if x < 0:
      return 0
    else:
      return self.rate * math.exp(-self.rate * x)

  def cdf(self, x):
    if x < 0:
      return 0
    else:
      return 1 - math.exp(-self.rate * x)

  def quantile(self, alpha):
    return -math.log(1 - alpha) / self.rate


# Распределение Лапласа
class LaplaceRandomVariable(RandomVariable):
  def __init__(self, location=0, scale=1):
    self.location = location
    self.scale = scale

  def pdf(self, x):
    return 0.5 * self.scale * math.exp(-self.scale * abs(x - self.location))

  def cdf(self, x):
    if x < self.location:
      return 0.5 * math.exp((x - self.location) / self.scale)
    else:
      return 1 - 0.5 * math.exp(-(x - self.location) / self.scale)

  def quantile(self, alpha):
    if alpha == 0.5:
      return self.location
    elif alpha < 0.5:
      return self.location - self.scale * math.log(1 - 2 * alpha)
    else:
      return self.location + self.scale * math.log(2 * alpha - 1)


# Распределение Коши
class CauchyRandomVariable(RandomVariable):
  def __init__(self, location=0, scale=1):
    self.location = location
    self.scale = scale

  def pdf(self, x):
    return 1 / (math.pi * self.scale * (1 + ((x - self.location) / self.scale) ** 2))

  def cdf(self, x):
    return 0.5 + math.atan((x - self.location) / self.scale) / math.pi

  def quantile(self, alpha):
    return self.location + self.scale * math.tan(math.pi * (alpha - 0.5))


# Интерфейс для генератора псевдослучайных величин
class RandomNumberGenerator(ABC):
  def __init__(self, random_variable: RandomVariable):
    self.random_variable = random_variable

  @abstractmethod
  def get(self, N):
    pass


 # Генератор псевдослучайных величин
class SimpleRandomNumberGenerator(RandomNumberGenerator):
  def __init__(self, random_variable: RandomVariable):
    super().__init__(random_variable)

  # Возвращает выборку объема N
  def get(self, N):
    us = np.random.uniform(0, 1, N)
    return np.vectorize(self.random_variable.quantile)(us)


# Функция для рисования графиков
def plot(xs, ys, colors):
  for x, y, c in zip(xs, ys, colors):
    plt.plot(x, y, c)
  plt.show()


# Класс для всех оценок
class Estimation(ABC):
  def __init__(self, sample):
    self.sample = sample


# Эмпирическая функция распределения
class EDF(Estimation):
  def heaviside_function(x):
    if x > 0:
      return 1
    else:
      return 0

  def value(self, x):
    return np.mean(np.vectorize(EDF.heaviside_function)(x - self.sample))


# Непараметрическая случайная величина
class SmoothedRandomVariable(RandomVariable, Estimation):

  def _k(x): # епанЕчнеков
    if abs(x) <= 1:
      return 0.75 * (1 - x * x)
    else:
      return 0

  def _K(x):
    if x < -1:
      return 0
    elif -1 <= x < 1:
      return 0.5 + 0.75 * (x - x ** 3 / 3)
    else:
      return 1

  def __init__(self, sample, h):
    super().__init__(sample)
    self.h = h

  # Оценка Розенблатта-Парзена
  def pdf(self, x):
    return np.mean([SmoothedRandomVariable._k((x - y) / self.h) for y in self.sample]) / self.h

  # Сглаженная эмпирическая оценка функции распределения
  def cdf(self, x):
    return np.mean([SmoothedRandomVariable._K((x - y) / self.h) for y in self.sample])

  def quantile(self, alpha):
    raise NotImplementedError


# Класс для вычисления гистограммы
class Histogram(Estimation):

  class Interval:
    def __init__(self, a, b):
      self.a = a
      self.b = b

    def is_in(self, x):
      return x >= self.a and x <= self.b

    def __repr__(self):
      return f'({self.a}, {self.b})'

  def __init__(self, sample, m):
    super().__init__(sample)
    self.m = m

    self.init_intervals()

  def init_intervals(self):
    left_boundary_of_intervals = np.linspace(np.min(sample), np.max(sample), self.m + 1)[:-1]
    right_boundary_of_intervals = np.concatenate((left_boundary_of_intervals[1:], [np.max(sample)]))

    self.intervals = [ Histogram.Interval(a, b) for a,b in zip(left_boundary_of_intervals, right_boundary_of_intervals)]

    self.sub_interval_width = right_boundary_of_intervals[0] - left_boundary_of_intervals[0]

  def get_interval(self, x):
    for i in self.intervals:
      if i.is_in(x):
        return i
    return None

  def get_sample_by_interval(self, interval):
    return np.array(list(filter(lambda x: interval.is_in(x), self.sample)))

  def value(self, x):
    return len(self.get_sample_by_interval(self.get_interval(x))) / ( self.sub_interval_width * len(self.sample) )

N = int(input("Размер выборки: N = "))
m = int(input("К-во подинтервалов: m = "))
bandwidth = float(input("Значение параметра размытости: bandwidth = "))
location = int(input("Параметр сдвига: location = "))
scale = int(input("Параметр масштаба: scale = "))

print("\n1-Нормальное\n2-Экспоненциальное\n3-Коши\n4-Лапласа\n5-Равномерное")
while True:
  randvar = int(input("\nВыберите распределение: "))

  if randvar == 1:
    rv = NormalRandomVariable(location, scale)

  elif randvar == 2:
    rv = ExponentialRandomVariable()

  elif randvar == 3:
    rv = CauchyRandomVariable(location, scale)

  elif randvar == 4:
    rv = LaplaceRandomVariable(location, scale)

  elif randvar == 5:
    rv = UniformRandomVariable(location, scale)
  else: break

  generator = SimpleRandomNumberGenerator(rv)

  # Выборка из распределения с заданными параметрами и объемом
  sample = generator.get(N)

  M = 100
  X = np.linspace(np.min(sample), np.max(sample), M)

  # Узлы, через которые проходит истинная функци распределения
  Y_truth = np.vectorize(rv.cdf)(X)

  # Значения эмпирической функции распределения
  edf = EDF(sample)
  Y_edf = np.vectorize(edf.value)(X)

  # Две оценки функции распределения
  srv = SmoothedRandomVariable(sample, bandwidth)
  Y_kernel = np.vectorize(srv.cdf)(X)
  plot([X]*3, [Y_truth, Y_edf, Y_kernel], ['r', 'b', 'g'])

  # Истиные значения плотности на интервале
  P_1 = np.vectorize(rv.pdf)(X)

  # Оценка гистограммы
  hist = Histogram(sample, m)
  P_2 = np.vectorize(hist.value)(X)

  # Значения оценки плотности Розенблатта-Парзена
  P_3 = np.vectorize(srv.pdf)(X)

  plot([X]*3, [P_1, P_2, P_3], ['r', 'b', 'g'])
