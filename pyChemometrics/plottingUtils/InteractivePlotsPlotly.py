from plotly.offline import iplot, plot
import plotly.graph_objs as go
import numpy as np


"""
General purpose interactive plots using plotly
"""

"""

x = np.random.randn(1000)
y = np.random.randn(10000)
c = np.random.randn(1000)
c_dic = np.random.randint(0, 10, (1000))

labels = ['le_ples'] * 1000
labels[500::] = ['wololoo']* 500

labels = np.array(labels)


pls = _plotly_lineplots(y, error=np.abs(y/10.0), xaxis=None,
                      colour='blue', colourmap='Portland', discrete_colour=False, label=None)

plot_url = plot(pls, filename='scatterplot_test2')



"""


def _plotly_lineplots(y, error=None, xaxis=None,
                      colour='blue', colourmap='Portland', discrete_colour=False, label=None):
    """

    :param mean:
    :param error:
    :param xaxis:
    :return:
    """

    plotly_line_plot = list()
    # Check if color array passed is numeric or strings and find nan's accordingly
    if xaxis is None:
        xaxis = np.arange(0, y.size, 1)

    # Main data series
    line_plot = go.Scatter(
        x=xaxis,
        y=y,
        mode='lines',
        line=dict(color=colour if colour is not None else 'blue', width=1))

    plotly_line_plot.append(line_plot)

    if error is not None:
        # Main data series
        upper_line_plot = go.Scatter(
            x=xaxis,
            y= y + error,
            mode='lines',
            line=dict(color=colour if colour is not None else 'blue', width=1)
        )
        plotly_line_plot.append(upper_line_plot)

        lower_line_plot = go.Scatter(
            x=xaxis,
            y=y - error,
            mode='lines',
            line=dict(color=colour if colour is not None else 'blue', width=1),
            fillcolor='rgba(0,100,80,0.2)',
            fill='tonexty',
            opacity=0.2
        )
        plotly_line_plot.append(lower_line_plot)

    data = go.Data(plotly_line_plot)
    layout = go.Layout(dict(hovermode="closest"))

    return go.Figure(data=data, layout=layout)


def _plotly_scatterplot(x, y, colour=None, colourmap='Portland', discrete_colour=False, labels=None, marker='circle'):
    """

    :param x:
    :param y:
    :param colour:
    :param colourmap:
    :param discrete_colour:
    :return:
    """

    # List containing the plots
    plotly_scatter_plot = list()

    # Check if color array passed is numeric or strings and find nan's accordingly
    if colour is not None:
        if np.issubdtype(colour.dtype.type, np.number):
            plotnans = np.isnan(colour)
        elif colour.dtype.type is np.str_:
            plotnans = (colour == 'nan')
    else:
        plotnans = np.zeros(shape=x.shape, dtype=bool)

    # Parse text labels
    # Check this properly, here
    # Parse markers
    # Plot NaN values in gray
    if np.any(plotnans):
        NaNplot = go.Scattergl(
            x=x[plotnans],
            y=y[plotnans],
            mode='markers',
            marker=dict(
                color='rgb(180, 180, 180)',
                symbol=marker,
            ),
            text=labels[~plotnans] if labels is not None else None,
            hoverinfo='text',
            showlegend=False
        )
        plotly_scatter_plot.append(NaNplot)

    if discrete_colour is False or colour is None:
        # Plot numeric values with a colorbar
        scatter_plot = go.Scattergl(x=x[~plotnans],
                                  y=y[~plotnans],
                                  mode='markers',
                                  marker=dict(colorscale=colourmap,
                                              color=colour[~plotnans] if colour is not None else None,
                                              symbol=marker,
                                              showscale=True if colour is not None else False),
                                  text=labels[~plotnans] if labels is not None else None,
                                  hoverinfo='text',
                                  showlegend=False
                                  )

        plotly_scatter_plot.append(scatter_plot)
    # Plot categorical values by unique groups
    else:
        uniq, indices = np.unique(colour, return_inverse=True)
        cmin = min(uniq)
        cmax = max(uniq)
        for level in uniq:
            scatter_plot = go.Scattergl(x=x[colour == level],
                                        y=y[colour == level],
                                        mode='markers',
                                        marker=dict(colorscale=colourmap,
                                                    color=level if colour is not None else None,
                                                    cmin=cmin,
                                                    cmax=cmax,
                                                    symbol=marker),
                                        text=labels[colour == level] if labels is not None else None,
                                        hoverinfo='text',
                                        name=level,
                                        showlegend=True)
            plotly_scatter_plot.append(scatter_plot)

    data = go.Data(plotly_scatter_plot)
    layout = go.Layout(dict(hovermode="closest"))

    return go.Figure(data=data, layout=layout)


def _plotly_add_circle(figure, x, y):
    """
    Add a plotly circle shape
    :param x:
    :param y:
    :return:
    """
    # Return or not?? this one is a layout, so tricky one
    figure.update(dict(layout={
        'shapes': [{'type': 'circle',
                    'xref': 'x',
                    'yref': 'y',
                    'x0': 0 - x,
                    'y0': 0 - y,
                    'x1': 0 + x,
                    'y1': 0 + y}]}))
    return figure


def _plotly_add_rectangle(figure, x1, x2, y1, y2):
    """
    Add a plotly circle shape
    :param x:
    :param y:
    :return:
    """
    # Return or not?? this one is a layout, so tricky one
    figure.update(dict(layout={
        'shapes': [{'type': 'rectangle',
                    'xref': 'x',
                    'yref': 'y',
                    'x0': x1,
                    'y0': y1,
                    'x1': x2,
                    'y1': y2}]}))
    return figure


def _plotly_draw_lorentzian(layout, mu, gamma, amplitude):
    return None


def _plotly_draw_gaussian(layout, mu, sigma, amplitude):
    return None