import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd

f = pd.read_csv('data1.csv', sep='|')

# utworzyłam tablice wyników studentów dla 6 różnych dawek Gy:
dane1 = np.array(f['0 Gy']) # tworzenie tablicy liczb dla danej kolumny z wynikami
dane2 = np.array(f['0.5 Gy'])
dane3 = np.array(f['1 Gy'])
dane4 = np.array(f['2 Gy'])
dane5 = np.array(f['3 Gy'])
dane6 = np.array(f['4 Gy'])

dane_wzorcowe = np.array([4, 14, 15, 27, 67, 86]) # wyniki eksperta

# skorzystam z funkcji histfit do przedstawienia wykresów wyników studentów i ich
# funkcji gęstości prawdopodobieństwa rozkładu normalnego:
#(aby sprawdzić jak wyniki są zblizone do rozkładu normalnego i czy mogę opierać się na założeniu, że wyniki pochodzą z tego rozkładu)
def histfit(x, N_bins):
    n, bins, patches = plt.hist(x, N_bins, density=True, facecolor='blue', alpha=0.75)
    bincenters = 0.5 * (bins[:-1] + bins[1:])
    y = st.norm.pdf(bincenters, loc=np.mean(x), scale=np.std(x))
    l = plt.plot(bincenters, y, 'r--', linewidth=1)

plt.subplot(3, 3, 1)
histfit(dane1, 10)
plt.title('0 Gy')
plt.subplot(3, 3, 2)
histfit(dane2, 10)
plt.title('0,5 Gy')
plt.subplot(3, 3, 3)
histfit(dane3, 10)
plt.title('1 Gy')
plt.subplot(3, 3, 4)
histfit(dane4, 10)
plt.title('2 Gy')
plt.subplot(3, 3, 5)
histfit(dane5, 10)
plt.title('3 Gy')
plt.subplot(3, 3, 6)
histfit(dane6, 10)
plt.title('4 Gy')

# policzę 95% przedziały ufności ze średniej dla wyników studentów aby je porównać z wynikami eksperta
# na podstawie wykresów (nie wszytskie są podobne do rozkładu gaussa) okazuje się, że nie mogę liczyć przedziałów ufności korzystając z st.t.ppf() (t studenta)
# bo to byłyby przedziały ufnosci z założenia o normalności
# więc użyję bootstrapu, który nie zakłada, że dane pochodzą z rozkładu normalnego

# przedziały ufności z bootstrapu:
# (chcę sprawdzić jakie są przedziały ufności korzystając bootsrapu
# i zobaczyć czy wyniki wzorcowe doktoranta zawierają się w tych przedziałach)
def randsample(x, ile):
    ind = st.randint.rvs(0, len(x), size = ile)
    y = x[ind]
    return y

def f_bootstrap(danex,x): # danex czyli dla danej dawki Gy
    alfa = 0.05
    Nboot = 10000
    N = len(danex)
    M = np.empty(Nboot)
    for i in range(Nboot):
        m = randsample(danex, N)
        M[i] = np.mean(m)
    lo = st.scoreatpercentile(M, per=alfa / 2 * 100)
    hi = st.scoreatpercentile(M, per=(1 - alfa / 2) * 100)
    if lo<x<hi:
        z='zawiera się'
    else:
        z='nie zawiera się'
    return(lo,hi,x,z)

print('Przedziały ufności z Bootstrapu:')
print(f_bootstrap(dane1,dane_wzorcowe[0]))
print(f_bootstrap(dane2,dane_wzorcowe[1]))
print(f_bootstrap(dane3,dane_wzorcowe[2]))
print(f_bootstrap(dane4,dane_wzorcowe[3]))
print(f_bootstrap(dane5,dane_wzorcowe[4]))
print(f_bootstrap(dane6,dane_wzorcowe[5]))

# skorzystam z regrasji liniowej żeby dopasować prostą wyników studentów i doktoranta i porównać współczynniki prostych:
#(co pozwoli za pomocą wykresu lepeij zobrazować “Jak efektywnie niedoświadczeni studenci mogą ocenić CA po podstawowym wprowadzeniu
# oraz jak ich szacunki są zgodne z wynikami doświadczonego studenta”)
def regresja_liniowa(X, Y):
    N = len(X)
    x_sr = np.mean(X)
    y_sr = np.mean(Y)
    b1 = np.sum((X - x_sr) * (Y - y_sr)) / np.sum((X - x_sr) ** 2)
    b0 = y_sr - b1 * x_sr

    Y_reg = b0 + b1 * X
    residua = Y - Y_reg

    sse = np.sum(residua ** 2)
    v_e = sse / (N - 2)
    s_b0 = np.sqrt(v_e) * np.sqrt(1.0 / N + x_sr ** 2 / np.sum((X - x_sr) ** 2))
    s_b1 = np.sqrt(v_e) * np.sqrt(1.0 / np.sum((X - x_sr) ** 2))
    return (b0, b1, s_b0, s_b1, residua)


X_ref = np.array([0, 0.5, 1, 2, 3, 4])
Y_ref = dane_wzorcowe
(b0, b1, s_b0, s_b1, residua) = regresja_liniowa(X_ref, Y_ref)

print('Równanie prostej dla wynikow doktoranta: y = b0 + b1*x')
print('Współczynniki: b0 = %.3f, b1 = %.3f' % (b0, b1))

plt.subplot(3, 3, 7)
plt.errorbar(X_ref, Y_ref)
Y_reg = b0 + b1 * X_ref
plt.plot(X_ref, Y_reg)

x1 = np.zeros(29)
x2 = np.full(29, 0.5)
x3 = np.ones(29)
x4 = np.full(29, 2)
x5 = np.full(29, 3)
x6 = np.full(29, 4)

X_exp = np.array(np.concatenate([x1, x2, x3, x4, x5, x6]))
Y_exp = np.array(np.concatenate([dane1, dane2, dane3, dane4, dane5, dane6]))
(b2, b3, s_b2, s_b3, residua2) = regresja_liniowa(X_exp, Y_exp)

print('Równanie prostej dla wynikow studentów: y = b2 + b3*x')
print('Współczynniki: b2 = %.3f, b3 = %.3f' % (b2, b3))

plt.errorbar(X_exp, Y_exp)
Y_reg = b2 + b3 * X_exp
plt.plot(X_exp, Y_reg)

plt.show()
