# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import linalg

def fastICA(x, n_comp, alpha=1.0, maxit=200, tol=1e-4):
    n, p = x.shape

    #step.1 観測信号の中心化
    x = x - x.mean(axis=0)

    #step.2 白色化(主成分分析)
    v = np.dot(x.T,x) / n
    #特異値分解
    u, sigma, v = linalg.svd(v)
    D = np.diag(1/np.sqrt(sigma))
    #白色化行列の生成
    V = np.dot(D, u.T)
    #成分の個数
    V = V[0:n_comp, 0:p]
    #白色化を実行してベクトルzを得る
    z = np.dot(V, x.T).T

    #step.3 射影ベクトルの初期化
    w_init = np.random.normal(loc=0, scale=1, size=(n_comp, n_comp))

    #step.4 減次法によるFastICAの実行
    w = ica_deflation(z,n_comp, alpha, maxit, tol, w_init)
    #復元信号
    s = np.dot(w, z.T).T
    plt.scatter(s[:,0], s[:,1])
    plt.xlabel("$y_{1}$", fontsize=30)
    plt.ylabel("$y_{2}$", fontsize=30)
    plt.show()
    return s

#減次法によるFastICA
def ica_deflation(x,n_comp, alpha, maxit, tol, w_init):
    n, p = x.shape
    #求める射影ベクトルの結果を格納するオブジェクト
    w_res = np.zeros((n_comp, n_comp))

    #step4 各独立成分を求める
    for i in np.arange(n_comp):
        w = w_init[i,].T
        t = np.zeros((n_comp, np.size(w)))
        #既に求められている独立成分と直交するように変換
        if i>0:
            k = np.dot(w_res[0:i,], w)
            t = (w_res[0:i].T*k).T
        w -= t.sum(axis=0)
        w /= np.linalg.norm(w)
        #誤差、反復回数カウンタの初期化
        lim = np.repeat(1000.0, maxit)
        it = 0
        #独立成分の探索
        while lim[it] > tol and it < maxit:
            #step.4 wpの更新
            wx = np.dot(x,w)
            gwx = np.tanh(alpha*wx)
            gwx = np.repeat(gwx, n_comp).reshape(np.size(gwx), n_comp)
            xgwx = x * gwx
            v1 = xgwx.mean(axis=0)
            g_wx = alpha * (1-(np.tanh(alpha*wx))**2)
            v2 = np.dot(np.mean(g_wx), w)
            w1 = v1 - v2
            it += 1
            t = np.zeros((n_comp, np.size(w)))
            #step.5 Gram-Schmidtの正規直交化
            if i > 0:
                k = np.dot(w_res[0:i,], w1)
                t = (w_res[0:i,].T*k).T
            w1 -= t.sum(axis=0)
            w1 /= np.linalg.norm(w1)
            #step.6 収束判定
            lim[it] = abs(abs(sum(w1*w))-1.0)
            w = w1
        w_res[i,] = w
    return w_res


#http://shiten4.hatenablog.com/entry/2014/01/10/125704
#ICA, PCA, スパースコーディングの実装
from numpy import *
from matplotlib.pyplot import *
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import DictionaryLearning

# 混合ガウス分布に従う乱数の生成
def rand_gauss_mix(mu, sigma, p, N):
    d, K = mu.shape
    p_cumsum = cumsum(array(p) / sum(p))
    X = zeros([d, 0])
    for n in range(N):
        p_ = p_cumsum - random.rand()
        p_[p_<0] = 1
        k = p_.argmin()
        x = random.multivariate_normal(mu[:,k], sigma[:,:,k]).reshape(-1,1)
        X = hstack((X, x))
    return X

# 回転行列
Rot = lambda rad: array([[cos(rad), sin(rad)], [-sin(rad), cos(rad)]])


# データの生成
N = 1000
distribution_type = 'rectangle'

if distribution_type == 'gauss':
    mu = array([0, 0])
    sigma = array([[0.1,0.1],[0.2,0.3]])
    X = random.multivariate_normal(mu, sigma, N).T
elif distribution_type == 'rectangle':
    rad = (1./6)*pi
    ext = [1, 0.4]
    X = 2*random.rand(2,N)-ones([2,N])
    X = dot(Rot(rad), X)
    X = dot(diag(ext), X)
elif distribution_type == 'closs':
    mu = zeros([2,2])
    p = [0.5, 0.5]
    sigma0 = diag([0.3,0.003])
    rad = [1./8*pi, 7./8*pi]
    sigma = zeros([2, 2, 2])
    for i in range(2):
        sigma[:,:,i] = dot( Rot(rad[i]), dot(sigma0, Rot(rad[i]).T) )
    X = rand_gauss_mix(mu, sigma, p, N)

# 主成分分析
decomposer = PCA()
decomposer.fit(X.T)
Upca = decomposer.components_.T
Apca = decomposer.transform(X.T).T

# 独立成分分析
decomposer = FastICA()
decomposer.fit(X.T)
Uica = decomposer.mixing_ 
Aica = decomposer.transform(X.T).T

# スパースコーディング
decomposer = DictionaryLearning()
decomposer.fit(X.T)
Usc = decomposer.components_ .T
Asc = decomposer.transform(X.T).T


axis('equal')
plot(X[0],X[1],'xc')
Upca = Upca / sqrt((Upca**2).sum(axis=0))
Uica = Uica / sqrt((Uica**2).sum(axis=0))
Usc = Usc / sqrt((Usc**2).sum(axis=0))
for i in range(2):
    p_pca = plot([0, Upca[0,i]], [0, Upca[1,i]], '-r')
    p_ica = plot([0, Uica[0,i]], [0, Uica[1,i]], '-b')
    p_sc = plot([0, Usc[0,i]], [0, Usc[1,i]], '-g')
legend(('data', 'PCA', 'ICA', 'SC'))
legend(loc="best", prop=dict(size=12))
show()

subplot(1,3,1)
plot(Apca[0], Apca[1], 'xc')
title('PCA')
subplot(1,3,2)
plot(Aica[0], Aica[1], 'xc')
title('ICA')
subplot(1,3,3)
plot(Asc[0], Asc[1], 'xc')
title('SC')
show()
