from src.visualization.evaluate import me, mae, mse, mpe, rmse, nrmse, calculate_metric_horizons, calculate_metric_horizons_all_models
import numpy as np
import pandas as pd
import datetime as datetime
pd.options.plotting.backend = "plotly"
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import n_colors

def plot_scatterplot(df, model):
    df = df[df.columns.drop(['Horizon'])]
    df = df.dropna()

    df_plot = df[['Y_true', model]]
    fig = df_plot.plot.scatter(x = model, y = 'Y_true')
    fig.update_layout(autosize=False, width=600, height=600,)

    # We add a straight line
    max_val = np.max(np.max(df_plot))
    ax = px.line(pd.DataFrame(dict(x = [0, max_val], y = [0, max_val])), x="x", y="y", title="Unsorted Input")
    ax.update_traces(line_color='#456987')
    fig.add_trace(ax.data[0])

    return(fig)

def plot_test_day(df, idx, width = 400, height = 600, title = ""):
    df = df[df.columns.drop(['Horizon'])]
    df = df.dropna()

    # assumes that Y_true is the last column
    n = len(df.columns)
    colors = list(px.colors.qualitative.Plotly[0:(n-1)]) + ['#565656']
    #if len(df.columns) > 4:
    #    n = len()
    #    colors = list(px.colors.qualitative.Plotly[0:n]) + ['#565656'] 

    forecast_days = df.index[df.index.time == datetime.time(0, 0)].strftime('%Y-%m-%d')
    forecast_day = forecast_days[idx]

    df_plot = df.loc[forecast_day]

    fig = df_plot.plot(width = width, height = height, color_discrete_sequence = colors)
    fig.update_layout(title=title, xaxis_title="", yaxis_title='Power')
    if len(df.columns) > 4:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.35, xanchor="left", x=0.01))
    else:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0.01))
    fig.update_layout(legend_title="Models", font=dict(family="Arial",size=22))

    return(fig)

def plot_horizons(df, method, models = None, vline = 0, title = "", plot_solar = False, plot_solar_range = [11,36]):
    df_plot = calculate_metric_horizons_all_models(df, method)
    if models is not None:
        df_plot = df_plot[models]

    if plot_solar:
        # We remove all inf which appear as a result of dividing by 0
        df_plot.replace([np.inf, -np.inf], np.nan, inplace=True)

        # We remove all values at morning / afternoon which divide by small ranges (max-min) resulting in large nrmse
        # We do this by looking at values within 05:30 to 18:00 and everything larger than the max value in this period are removed
        for model in models:
            max_val = np.max(df_plot.loc[plot_solar_range[0]:plot_solar_range[1], model])
            df_plot.loc[df_plot[model] > max_val, model] = np.nan

            # We manually remove some outlier to make the plot nicer for Nazeerabad
            if plot_solar_range[1] == 35:
                df_plot.loc[36:48] = np.nan

    # We return the plotly plot
    fig = df_plot.plot()
    #fig.update_layout(margin=dict(l=20, r=20, t=20, b=60),)
    fig.update_layout(title=title, xaxis_title="Lead time (30min resolution)", yaxis_title=method)
    fig.update_layout(xaxis = dict(tickmode = 'array',tickvals = df_plot.index, ticktext = df_plot.index))
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.27, xanchor="left", x=0.01))
    if vline != 0:
        fig.add_vline(x=vline, line_width=1, line_dash="dash", line_color="black")

    fig.update_layout(legend_title="Models", font=dict(family="Arial",size=18))

    return(fig)

def plot_mean_std_error_multiple_models(df, models, vline = 0):
    fill_colors = px.colors.qualitative.Plotly
    line_colors = ['black', '#7f7f7f']
    #fill_colors = ['#fed0fc', '#e86af0']

    fig = go.Figure()

    for cnt, model in enumerate(models):
        fig = plot_add_std_error(df, fig, model, fill_colors[cnt])
    for cnt, model in enumerate(models):
        fig = plot_add_mean_error(df, fig, model, line_colors[cnt])

    fig.update_layout(template = 'ggplot2',
                    title_text = 'Mean error and standard deviation', title_x=0,
                    yaxis_title = 'Error',
                    xaxis_title="Lead time (30min resolution)")
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="left", x=0.01))
    if vline != 0:
        fig.add_vline(x=vline, line_width=1, line_dash="dash", line_color="black")

    return(fig)

def plot_add_std_error(df, fig, model, fill_color):
    me = calculate_metric_horizons_all_models(df, 'me')[model]
    std = calculate_metric_horizons_all_models(df, 'std')[model]
    df_plot = pd.DataFrame({'me': me, 'std_low': me-std, 'std_high': me+std})

    # Add traces
    fig.add_trace(go.Scatter(x=df_plot.index,
                            y=df_plot['std_low'],
                            fill='tonexty', # fill area between trace0 and trace1
                            mode='lines', 
                            name=f'{model} Standard deviation',
                            line=dict(width=0.5, color=fill_color)))
    fig.add_trace(go.Scatter(x=df_plot.index,
                            y=df_plot['std_high'],
                            fill='tonexty', # fill area between trace0 and trace1
                            mode='lines', 
                            showlegend=False,
                            line=dict(width=0.5, color=fill_color)))
    return(fig)

def plot_add_mean_error(df, fig, model, line_color):
    me = calculate_metric_horizons_all_models(df, 'me')[model]
    std = calculate_metric_horizons_all_models(df, 'std')[model]
    df_plot = pd.DataFrame({'me': me, 'std_low': me-std, 'std_high': me+std})

    fig.add_trace(go.Scatter(x=df_plot.index, 
                            y=df_plot['me'],
                            mode='lines+markers',
                            name=f'{model} Mean error',
                            line_color=line_color))
    return(fig)

def plot_ridges_multiple_models(df, models):
    show_legend = [True,False,False]
    horizons = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 47]
    colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', len(horizons), colortype='rgb')

    fig = go.Figure()
    fig = make_subplots(rows=1, cols=len(models), subplot_titles = models)
    for cnt, model in enumerate(models):
        df_plot = df[[model,'Horizon', 'Y_true']]
        df_plot.loc[:,'metric_model1'] = df_plot[model] - df_plot['Y_true']

        for horizon, color in zip(horizons, colors):
            data_line = df_plot.loc[df_plot['Horizon'] == horizon]['metric_model1']
            fig.add_trace(go.Violin(x=data_line, line_color=color, name = horizon, showlegend = show_legend[cnt]), row = 1, col = cnt+1)
            
        fig.update_traces(orientation='h', side='positive', width=5, points='outliers')
        fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=True)
        fig.update_layout(template = 'ggplot2',
                            yaxis_title = 'Lead time (30min resolution)')
        
        fig.update_layout(title=go.layout.Title(text=f"Distribution of forecast errors against lead time",xref="paper",x=0))
    
    # We add the x axis text manually
    for i in range(1,len(models)+1): 
        fig['layout']['xaxis{}'.format(i)]['title']='Forecast errors'
    
    return(fig)

def plot_violins_multiple_models(df, models, horizons):
    show_legend = [True,False,False,False, False, False, False, False, False, False, False, False, False, False, False]

    model1 = 'Y_pred_LSTM(full day)'
    model2 = 'Y_pred_LSTM(full day)_sum'

    df_plot = df[[models[0],models[1],'Horizon', 'Y_true']]
    df_plot.loc[:,'metric_model1'] = df_plot[model1] - df_plot['Y_true']
    df_plot.loc[:,'metric_model2'] = df_plot[model2] - df_plot['Y_true']

    fig = go.Figure()
    for cnt, horizon in enumerate(horizons):
        # We calculate the metrics for the plot
        x_int = df_plot.loc[df_plot['Horizon'] == horizon]['Horizon']
        y1 = df_plot.loc[df_plot['Horizon'] == horizon]['metric_model1']
        y2 = df_plot.loc[df_plot['Horizon'] == horizon]['metric_model2']
        
        # We add the histograms
        fig.add_trace(go.Violin(x=x_int,
                                    y=y1,
                                    legendgroup=model1, scalegroup=model1, name=model1,
                                    side='negative',
                                    box_visible=True,
                                    line_color='lightseagreen',
                                    showlegend=show_legend[cnt])
                    )
        fig.add_trace(go.Violin(x=x_int,
                                    y=y2,
                                    legendgroup=model2, scalegroup=model2, name=model2,
                                    side='positive',
                                    box_visible=True,
                                    line_color='mediumpurple',
                                    showlegend=show_legend[cnt])
                    )
        
    # update characteristics shared by all traces
    fig.update_traces(meanline_visible=True,
                    #points='all', # show all points
                    jitter=0.05,  # add some jitter on points for better visibility
                    scalemode='count') #scale violin plot area with total count
    fig.update_layout(
        title_text="",
        violingap=0, violingroupgap=0, violinmode='overlay')
    fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=True)
    fig.update_layout(template = 'ggplot2',
                        yaxis_title = 'Forecast errors',
                        xaxis_title = 'Lead time (30min resolution)')
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0.01))
            
    fig.update_layout(title=go.layout.Title(text=f"Distribution of forecast errors against lead time",xref="paper",x=0))
    
    return(fig)

# used for uncertainty propagation
def plot_quantile_bands(df, df_quantiles, day, residual_quantile = False):
    df_plot = df[day]

    fig = go.Figure()
    keys = np.array(df_quantiles.columns)

    n_bands = int(len(df_quantiles.columns)/2)
    fill_colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', n_bands+1, colortype='rgb')
    line_colors = px.colors.qualitative.Plotly

    for cnt in range(0, n_bands):
        color = fill_colors[cnt]

        # If we give the argument residual_quantile we add the quantile to Y_true
        if residual_quantile:
            y_lower = df_plot[0].values + df_quantiles[keys[cnt]].values
            y_upper = df_plot[0].values + df_quantiles[keys[-(cnt+1)]].values

            # We fix the band above 0
            y_lower[y_lower < 0] = 0
            y_upper[y_upper < 0] = 0

            # We plot the bands
            fig.add_trace(go.Scatter(x=df_plot.index,
                                    y=y_lower ,
                                    mode='lines', 
                                    showlegend = False,
                                    line=dict(width=0, color=color)))
            fig.add_trace(go.Scatter(x=df_plot.index,
                                    y=y_upper,
                                    mode='lines', 
                                    fill='tonexty', # fill area between trace0 and trace1
                                    name=f'{keys[-(cnt+1)]*100} quantile',
                                    line=dict(width=0, color=color)))
        else:
            # We plot the bands
            fig.add_trace(go.Scatter(x=df_plot.index,
                                    y= df_quantiles[keys[cnt]],
                                    mode='lines', 
                                    showlegend = False,
                                    line=dict(width=0, color=color)))
            fig.add_trace(go.Scatter(x=df_plot.index,
                                    y=df_quantiles[keys[-(cnt+1)]],
                                    mode='lines', 
                                    fill='tonexty', # fill area between trace0 and trace1
                                    name=f'{keys[-(cnt+1)]*100} quantile',
                                    line=dict(width=0, color=color)))

    # We plot the deterministic part of the forecast
    fig.add_trace(go.Scatter(x=df_plot.index, 
                            y=df_plot['Y_true'],
                            mode='lines+markers',
                            name='Y_true',
                            line_color='black'))    

    fig.add_trace(go.Scatter(x=df_plot.index, 
                            y=df_plot[0],
                            mode='lines+markers',
                            name='Y_pred',
                            line_color=line_colors[1])) 

    fig.update_layout(#template = 'ggplot2',
                                yaxis_title = 'Power',
                                xaxis_title = 'Date')
    fig.update_layout(legend_title="", font=dict(family="Arial",size=18))
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0.01))

    # We fix the range
    y_max = np.max(y_upper)
    print(y_max)
    if y_max > 1:
        fig.update_yaxes(range = [0,y_max+0.25])
    else:
        fig.update_yaxes(range = [0,1])

    return(fig)

def plot_simulated_violin_horizons(names, df_vals_left, df_vals_right, std_sims, mean_sims, horizons, yaxis_title = "Power", solar_plot = False):
    show_legend = [True, False, False, False, False, False, False, False, False, False, False, False, False]

    name_left = names[0]
    name_right = names[1]

    fig = go.Figure()
    for cnt, horizon in enumerate(horizons):
        fig.add_trace(go.Violin(x=np.repeat(horizon, df_vals_left.shape[0]),
                                y=df_vals_left[horizon],
                                legendgroup=name_left, scalegroup=name_left, name=name_left,
                                side='negative',
                                box_visible=True,
                                line_color='lightseagreen',
                                spanmode = 'hard',
                                showlegend=show_legend[cnt]))

        fig.add_trace(go.Violin(x = np.repeat(horizon, df_vals_right.shape[0]),
                                y=df_vals_right[horizon],
                                legendgroup=name_right, scalegroup=name_right, name=name_right,
                                side='positive',
                                box_visible=True,
                                line_color='mediumpurple',
                                spanmode = 'hard',
                                showlegend=show_legend[cnt]))


    fig.update_traces(meanline_visible=True,
                        #points='all', # show all points
                        jitter=0.05,  # add some jitter on points for better visibility
                        scalemode='count') #scale violin plot area with total count
    fig.update_layout(violingap=0, violingroupgap=0, violinmode='overlay')
    fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=True)
    fig.update_layout(#template = 'ggplot2',
                        yaxis_title = yaxis_title,
                        xaxis_title = 'Lead time (30min resolution)')
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0.01))
    fig.update_layout(legend_title="", font=dict(family="Arial",size=18))

    # We add the standard deviations
    for cnt, horizon in enumerate(horizons):
        if solar_plot:
            x_horizon_pos = np.array([horizon+0.66, horizon+0.1])
            x_horizon_neg = np.array([horizon-0.66, horizon-0.1])
            shift = 0.66
        else:
            x_horizon_pos = np.array([horizon+1.1, horizon+0.1])
            x_horizon_neg = np.array([horizon-1.1, horizon-0.1])
            shift = 0.33

        std_col_left = std_sims.columns[0]
        std_col_right = std_sims.columns[1]
        mean_col_left = mean_sims.columns[0]
        mean_col_right = mean_sims.columns[1]

        fig.add_trace(go.Scatter(x=x_horizon_neg, 
                                 y=np.array([std_sims.loc[horizon, std_col_left], std_sims.loc[horizon, std_col_left]]),
                                 mode = 'lines+markers',
                                 name = std_col_left,
                                 showlegend=show_legend[cnt],
                                 opacity = 0.6,
                                 marker = dict(size = 8, color = 'seagreen')))
        fig.add_trace(go.Scatter(x=x_horizon_pos, 
                                 y=np.array([std_sims.loc[horizon, std_col_right], std_sims.loc[horizon, std_col_right]]),
                                 mode = 'lines+markers',
                                 name = std_col_right,
                                 showlegend=show_legend[cnt],
                                 opacity = 0.6,
                                 marker = dict(size = 8, color = 'purple', symbol = 'triangle-up')))

        # Add markers for the mean
        fig.add_trace(go.Scatter(x=[x_horizon_neg[0]], 
                                 y=np.array([mean_sims.loc[horizon, mean_col_left]]),
                                 mode = 'markers',
                                 name = mean_col_left,
                                 showlegend=show_legend[cnt],
                                 opacity = 0.9,
                                 marker = dict(size = 10, color = 'seagreen', symbol = 'triangle-down')))
        fig.add_trace(go.Scatter(x=[x_horizon_pos[0]], 
                                 y=np.array([mean_sims.loc[horizon, mean_col_right]]),
                                 mode = "markers",
                                 name = mean_col_right,
                                 showlegend=show_legend[cnt],
                                 opacity = 0.9,
                                 marker = dict(size = 10, color = 'purple', symbol = 'cross')))

        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),)
    
    return fig

def plot_single_day_simulations(df,day,title):
    df = df.drop(['Horizon'], axis=1)

    fig = df.loc[day].plot()
    fig.update_layout()
    fig.add_trace(go.Scatter(x=df.loc[day].index, 
                                y=df.loc[day]['Y_true'],
                                mode='lines+markers',
                                name='Y_true',
                                line_color=px.colors.qualitative.Plotly[1]))  
    fig.update_layout(#template = 'ggplot2',
                                yaxis_title = 'Power',
                                xaxis_title = 'Date')
    fig.update_layout(legend_title="", font=dict(family="Arial",size=18))

    return fig