import pyChemometrics
from sklearn.datasets import make_classification
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, classification_report
from pyChemometrics.ChemometricsPLS_Logistic import *

fake_data = make_classification(n_samples=5000, n_features=600, n_informative=10, n_classes=2, class_sep=100)

fake_x = fake_data[0]
fake_y = fake_data[1]

ples = ChemometricsPLS_Logistic(ncomps=5)
ples.fit(fake_x[0:1000], fake_y[0:1000])

ples2 = ChemometricsPLS(ncomps=5)
ples2.fit(fake_x[0:1000], pds.get_dummies(fake_y[0:1000]).values)
ples.modelParameters['PLS']
ples2.modelParameters

dummy_matrix = np.zeros((len(fake_y[0:1000]), 3))
for col in range(3):
    dummy_matrix[np.where(fake_y[0:1000] == col), col] = 1
"""
ples.cross_validation(fake_x[0:1000], fake_y[0:1000])

ples = pyChemometrics.ChemometricsPLS(ncomps=2)

ples.fit(fake_x[0:1000], fake_y[0:1000])
ples.cross_validation(fake_x[0:1000], fake_y[0:1000])


logmod = LogisticRegression(multi_class='ovr', solver='lbfgs')
logmod.fit(ples.scores_t, fake_y[0:1000])
y_pred = logmod.predict(ples.scores_t)
y_score = logmod.decision_function(ples.scores_t)
metrics.accuracy_score(fake_y[0:1000], y_pred)
#logmod.fit(fake_x[0:1000], fake_y[0:1000])

#ples.scores_t[fake_y == 1, 1] *= 0.1

#lda = LinearDiscriminantAnalysis()
#lda.fit(ples.scores_t, fake_y)

#qda = QuadraticDiscriminantAnalysis(store_covariances=True)
#qda.fit(ples.scores_t, fake_y)

#model = qda

testscore = ples.transform(x=fake_x[1000::])
y_pred = logmod.predict(testscore)
ole = logmod.decision_function(testscore)
fpr, tpr, _ = roc_curve(fake_y[1000::], ole)
roc_auc = auc(fpr, tpr)

roc = ples.cvParameters['Logistic']['CV_TestROC'][4]

fpr = roc[0]
tpr = roc[1]
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

plt.plot(ples.scores_t[:,0 ], ples.scores_t[:, 1], 'ro')
plt.plot(testscore[:, 0], testscore[:, 1], 'bo')
#ples.scores_t[fake_y == 1, 1] *= 0.1

#lda = LinearDiscriminantAnalysis()
#lda.fit(ples.scores_t, fake_y)

qda = QuadraticDiscriminantAnalysis(store_covariances=True)
qda.fit(ples.scores_t, fake_y)

#model = qda

testscore = ples.transform(x=fake_x[1000::])
y_pred = logmod.predict(testscore)

fpr, tpr, _ = roc_curve(fake_y[1000::], y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

plt.plot()


y_prob = model.predict_proba(ples.scores_t)

#ccmat = confusion_matrix(fake_y, y_pred, labels=None, sample_weight=None)
#ccurve = calibration_curve(fake_y, y_prob[:, 0])


xx, yy, zz = np.mgrid[-7:7:.01, -5:5:.01]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

f, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                      vmin=0, vmax=1)

ax_c = f.colorbar(contour)
ax_c.set_label("$P(y = 1)$")
ax_c.set_ticks([0, .25, .5, .75, 1])

ax.scatter(ples.scores_t[:, 0], ples.scores_t[:, 1], c=fake_y, s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.set(aspect="equal",
       xlim=(-5, 5), ylim=(-5, 5),
       xlabel="$X_1$", ylabel="$X_2$")


def plot_data(lda, X, y, y_pred, fig_index):
    splot = plt.subplot(2, 2, fig_index)
    if fig_index == 1:
        plt.title('Linear Discriminant Analysis')
        plt.ylabel('Data with fixed covariance')
    elif fig_index == 2:
        plt.title('Quadratic Discriminant Analysis')
    elif fig_index == 3:
        plt.ylabel('Data with varying covariances')

    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    alpha = 0.5

    # class 0: dots
    plt.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', alpha=alpha,
             color='red')
    plt.plot(X0_fp[:, 0], X0_fp[:, 1], '*', alpha=alpha,
             color='#990000')  # dark red

    # class 1: dots
    plt.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', alpha=alpha,
             color='blue')
    plt.plot(X1_fp[:, 0], X1_fp[:, 1], '*', alpha=alpha,
             color='#000099')  # dark blue

    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.))
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

    # means
    plt.plot(lda.means_[0][0], lda.means_[0][1],
             'o', color='black', markersize=10)
    plt.plot(lda.means_[1][0], lda.means_[1][1],
             'o', color='black', markersize=10)

    return splot

y_pred = model.predict(ples.scores_t)
splot = plot_data(model, ples.scores_t, fake_y, y_pred, 1)
plot_qda_cov(model, splot)
"""