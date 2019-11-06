import plotly.offline as plotly


def _lineplotsInteractive(mean, error=None, xaxis=None):

    fig, ax = plt.subplots()
    if xaxis is None:
        ax.plot(mean)
        xaxis = range(mean.size)
    else:
        ax.plot(xaxis, mean)
    if error is not None:
        ax.fill_between(xaxis, mean - error, mean + error, alpha=0.2, color='red')

    return fig, ax


def _barplotsInteractive(mean, error=None, xaxis=None):

    fig, ax = plt.subplots()
    if xaxis is None:
        xaxis = range(mean.size)

    if error is None:
        ax.bar(xaxis, height=mean)
    else:
        ax.bar(xaxis, height=mean, yerr=error)
    return fig, ax

