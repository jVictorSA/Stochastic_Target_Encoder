import pandas as pd
import cv2
from copy import deepcopy
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def boxplot(csv_path, metric, model, dataset, img_height=700, show_legend=True, RMSE_range = None, MAE_range=None, ticks= None):
    y_legend = -0.4

    cvs = pd.read_csv(csv_path)

    cvs = cvs.loc[(cvs['Encoder'] == 'STE_com_smooth_2x') |\
                  (cvs['Encoder'] == 'TE_smooth_2x') |\
                  (cvs['Encoder'] == 'MEst_smooth_2x') |\
                  (cvs['Encoder'] == 'JSE') |\
                  (cvs['Encoder'] == 'QuantE_smooth_2x') |\
                  (cvs['Encoder'] == 'LOOE') |\
                  (cvs['Encoder'] == 'OE')].reset_index()

    change_names = {      
        "STE_com_smooth_2x": "Ours (smooth = 2x higher)",        
        "TE_smooth_2x": "Target Encoding (smooth = 2x higher)",        
        "MEst_smooth_2x": "M-Estimate (smooth = 2x higher)",
        "JSE": "James-Stein Encoding",        
        "QuantE_smooth_2x": "Quantile Encoding (smooth = 2x higher)",
        "LOOE": "Leave-One-Out Encoding",
        "OE": "Ordinal Encoding"
    }

    for key, value in change_names.items():
        cvs.loc[cvs.Encoder == key, 'Encoder'] = value

    encoders = cvs['Encoder'].unique()

    pallete = {"Ours (smooth = 100)": "steelblue",
            "Ours (smooth = 50% higher)": "deepskyblue",
            "Ours (smooth = 2x higher)":"steelblue",
            "Target Encoding (smooth = 100)": "brown",
            "Target Encoding (smooth = 50% higher)": "indianred",
            "Target Encoding (smooth = 2x higher)":"brown",
            "M-Estimate (smooth = 100)": "green",
            "M-Estimate (smooth = 50% higher)": "mediumseagreen",
            "M-Estimate (smooth = 2x higher)":"mediumseagreen",
            "James-Stein Encoding": "darkviolet",
            "Quantile Encoding (smooth = 100)": "darkorange",
            "Quantile Encoding (smooth = 50% higher)":"sandybrown",
            "Quantile Encoding (smooth = 2x higher)": "darkorange",
            "Leave-One-Out Encoding": "grey",
            "Ordinal Encoding":"red"
            }
    
    box_df = pd.DataFrame()
    box_df_mae = pd.DataFrame()
    for encoder in encoders:
        valores_metrica = cvs.loc[cvs['Encoder'] == encoder][metric]
        box_df[encoder] = valores_metrica.values

    for encoder in encoders:
        valores_metrica = cvs.loc[cvs['Encoder'] == encoder]["MAE"]
        box_df_mae[encoder] = valores_metrica.values


    fig = make_subplots(rows=1, cols=2)
    fig.update_annotations(font_size=22)

    layout = go.Layout(
        autosize=False,
        title = dict(text=f'{dataset}', font=dict(size=35)),
        xaxis = go.layout.XAxis(showticklabels=False),
        yaxis = go.layout.YAxis(type="log",
                                title = dict(text = f'{metric} (log)', font=dict(size=25)),
                                dtick=ticks
                                ),
        width=1400,
        height=img_height
        )

    for column in box_df.columns:
        legend_name = column
        legend_name = legend_name.replace(' (smooth = 2x higher)', '')

        if legend_name == "Ours":
          legend_name = "<b>" + legend_name + "</b>"

        fig.add_trace(go.Box(y=box_df[column],\
                             name=legend_name, \
                             fillcolor = pallete[column],\
                             line = {"color":"black", "width":0.5},\
                             marker=dict(color=pallete[column],\
                             line=dict(color='black')),
                             showlegend=show_legend),
                             row=1, col=1)

    for column in box_df_mae.columns:

      fig.add_trace(go.Box(y=box_df_mae[column],\
                            fillcolor = pallete[column],\
                            line = {"color":"black", "width":0.5},\
                            marker=dict(color=pallete[column],\
                            line=dict(color='black')),
                            showlegend=False),
                            row=1, col=2)

    fig.update_layout(layout, plot_bgcolor='white', legend = dict(font = dict(family = "Arial", size = 38, color = "black"),
                      orientation="h", yanchor="bottom", y=y_legend, xanchor="center", x=0.5))

    fig.update_yaxes(gridcolor='grey', griddash='dash')
    fig.update_yaxes(range=RMSE_range, row=1, col=1)
    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(showticklabels=False, title=dict(text="<b>Encoders</b>", font=dict(size=24)))
    fig.update_yaxes(type="log",
                      title = dict(text = 'MAE (log)',
                          font=dict(size=25)),
                     range=MAE_range, row=1, col=2)

    fig.write_image(f"plots/Plotly boxplot_{metric}s_{model}_{dataset}.png")

def subplot_boxplot_smooth(airbnb_path, bike_sharing_path, co2_emissions_path, sp_air_path, metric, model, dataset):

  a = ' (Airbnb Reviews)'
  b = ' (Bike Sharing Demand)'
  c = ' (CO2 Emissions)'
  d = ' (PM10 SP Air)'

  legend_names = [a,b,c,d]

  airbnb = pd.read_csv(airbnb_path)
  bike_sharing = pd.read_csv(bike_sharing_path)
  co2_emissions = pd.read_csv(co2_emissions_path)
  sp_air = pd.read_csv(sp_air_path)

  airbnb = airbnb.loc[(airbnb['Encoder'] == 'STE_com_smooth') |\
                      (airbnb['Encoder'] == 'STE_com_smooth_0_50x') |\
                      (airbnb['Encoder'] == 'STE_com_smooth_2x')].reset_index(drop=True)

  bike_sharing = bike_sharing.loc[(bike_sharing['Encoder'] == 'STE_com_smooth') |\
                                  (bike_sharing['Encoder'] == 'STE_com_smooth_0_50x') |\
                                  (bike_sharing['Encoder'] == 'STE_com_smooth_2x')].reset_index(drop=True)
  
  co2_emissions = co2_emissions.loc[(co2_emissions['Encoder'] == 'STE_com_smooth') |\
                                    (co2_emissions['Encoder'] == 'STE_com_smooth_0_50x') |\
                                    (co2_emissions['Encoder'] == 'STE_com_smooth_2x')].reset_index(drop=True)

  sp_air = sp_air.loc[(sp_air['Encoder'] == 'STE_com_smooth') |\
                      (sp_air['Encoder'] == 'STE_com_smooth_0_50x') |\
                      (sp_air['Encoder'] == 'STE_com_smooth_2x')].reset_index(drop=True)

  cvs = [airbnb, bike_sharing, co2_emissions, sp_air]

  change_names1 = {
      "STE_com_smooth": "Smooth = 100 (Airbnb Reviews)",
      "STE_com_smooth_0_50x": "Smooth = 50% higher (Airbnb Reviews)",
      "STE_com_smooth_2x": "Smooth = 2x higher (Airbnb Reviews)",
  }
  change_names2 = {
      "STE_com_smooth": "Smooth = 100 (Bike Sharing Demand)",
      "STE_com_smooth_0_50x": "Smooth = 50% higher (Bike Sharing Demand)",
      "STE_com_smooth_2x": "Smooth = 2x higher (Bike Sharing Demand)",
  }
  change_names3 = {
      "STE_com_smooth": "Smooth = 100 (CO2 Emissions)",
      "STE_com_smooth_0_50x": "Smooth = 50% higher (CO2 Emissions)",
      "STE_com_smooth_2x": "Smooth = 2x higher (CO2 Emissions)",
  }
  change_names4 = {
      "STE_com_smooth": "Smooth = 100 (PM10 SP Air)",
      "STE_com_smooth_0_50x": "Smooth = 50% higher (PM10 SP Air)",
      "STE_com_smooth_2x": "Smooth = 2x higher (PM10 SP Air)",
  }

  pallete1 = {"Smooth = 100 (Airbnb Reviews)": "skyblue",
            "Smooth = 50% higher (Airbnb Reviews)": "green",
            "Smooth = 2x higher (Airbnb Reviews)": "red"
            }
  pallete2 = {
            "Smooth = 100 (Bike Sharing Demand)": "skyblue",
            "Smooth = 50% higher (Bike Sharing Demand)": "green",
            "Smooth = 2x higher (Bike Sharing Demand)":"red"
  }
  pallete3 = {
            "Smooth = 100 (CO2 Emissions)": "skyblue",
            "Smooth = 50% higher (CO2 Emissions)": "green",
            "Smooth = 2x higher (CO2 Emissions)": "red"
  }
  pallete4 = {
            "Smooth = 100 (PM10 SP Air)": "skyblue",
            "Smooth = 50% higher (PM10 SP Air)": "green",
            "Smooth = 2x higher (PM10 SP Air)": "red"
  }

  names = []
  palletes = []
  datasets = ['Airbnb Reviews', 'Bike Sharing Demand', 'CO2 Emissions', 'PM10 SP Air']
  names.append(change_names1)
  names.append(change_names2)
  names.append(change_names3)
  names.append(change_names4)

  palletes.append(pallete1)
  palletes.append(pallete2)
  palletes.append(pallete3)
  palletes.append(pallete4)

  for i in range(len(names)):
    print()
    for key, value in names[i].items():
        cvs[i].loc[cvs[i].Encoder == key, 'Encoder'] = value

    encoders = cvs[i]['Encoder'].unique()

    print("=============================")

    box_df = pd.DataFrame()
    box_df_mae = pd.DataFrame()
    for encoder in encoders:
        if encoder in names[i].values():
          valores_metrica = cvs[i].loc[cvs[i]['Encoder'] == encoder]['RMSE']
          box_df[encoder] = valores_metrica.values

    for encoder in encoders:
        if encoder in names[i].values():
          valores_metrica = cvs[i].loc[cvs[i]['Encoder'] == encoder]['MAE']
          box_df_mae[encoder] = valores_metrica.values

    layout = go.Layout(
        autosize=False,
        title = dict(text=datasets[i], font=dict(size=35)),
        xaxis = go.layout.XAxis(showticklabels=False),
        yaxis = go.layout.YAxis(type="log",
            title = dict(text = f'<b>{metric} (log)</b>',
                          font=dict(size=25))
        ),        

        width =950,
        height=500
    )
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("<b>RMSE</b>","<b>MAE</b>")
                        )
    fig.update_annotations(font_size=22)


    for column in box_df.columns:
        legend_name = column
        for string in legend_names:
          legend_name = legend_name.replace(string, '')
        fig.add_trace(go.Box(y=box_df[column],\
                            name=f'<b>{legend_name}</b>', \
                            fillcolor = palletes[i][column],\
                            line = {"color":"black", "width":0.5},\
                            marker=dict(color=palletes[i][column],\
                            line=dict(color='black'))),
                            row=1,col=1
                      )
    for column in box_df_mae.columns:
        legend_name = column
        for string in legend_names:
          legend_name = legend_name.replace(string, '')

        fig.add_trace(go.Box(y=box_df_mae[column],\
                            fillcolor = palletes[i][column],\
                            line = {"color":"black", "width":0.5},\
                            marker=dict(color=palletes[i][column],\
                            line=dict(color='black')),
                            showlegend=False),
                            row=1,col=2
                      )

    fig.update_layout(layout)
    fig.update_yaxes(gridcolor='grey', griddash='dash')
    fig.update_yaxes(type="log",
                      title = dict(text = 'MAE (log)',
                          font=dict(size=25)),
                      row=1, col=2)

    fig.update_xaxes(showticklabels=False)

    if i == 2:
      fig.update_yaxes(range=[1.1,1.5], row=1, col=2)
    # fig.update_layout()
    fig.update_layout(plot_bgcolor='white', legend = dict(font = dict(family = "Arial", size = 26, color = "black"),
                      orientation="h", itemwidth=30, yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                      margin=dict(l=90, r=10, t=70, b=40)
                      )


    fig.layout.annotations[0].update(y=-0.1)
    fig.layout.annotations[1].update(y=-0.1)

    fig.show(renderer="png")

    fig.write_image(f"plots/Smooth_{metric}s_{datasets[i]}.png")

def concatenate_images(images_paths, title, plot_type):
  imgs = []

  for i in range(len(images_paths)):
    imgs.append(cv2.imread(images_paths[i]))



  print(imgs[2].shape, imgs[3].shape)
  im_h1 = cv2.hconcat([imgs[0], imgs[1]])
  
  im_h2 = cv2.hconcat([imgs[2], imgs[3]])

  im_v = cv2.vconcat([im_h1, im_h2])


  # Comparativo encoders
  if plot_type == 'encoders_comparison':
    #Crops
    img2 = deepcopy(im_v)

    ours = None
    target = None
    m_estimate = None
    james_stein = None
    quantile = None
    looe = None
    ordinal = None

    
    ours = im_v[1486:1533, 237:352]
    target = im_v[1486:1533, 710:1013]
    m_estimate = im_v[1533:1580, 237:455]
    james_stein = im_v[1533:1580, 710:1117]
    quantile = im_v[1580:1627, 237:575]
    looe = im_v[1580:1627, 710:1172]
    ordinal = im_v[1627:1678, 237:555]

    # Clear legend
    cv2.rectangle(img2,(120,1480),(2800,1878),(255,255,255),-1)
    print(img2.shape[:2])

    x_spacing = 60

    legend_width = ours.shape[1] + target.shape[1] + m_estimate.shape[1] + james_stein.shape[1] + x_spacing*4
    center = math.floor(im_v.shape[1]/2 - legend_width/2)

    #Pastes
    x_offset = center
    y_offset = 1550
    img2[y_offset:y_offset+47, x_offset:x_offset+ours.shape[1]] = ours
    x_offset += x_spacing + ours.shape[1]
    img2[y_offset:y_offset+47, x_offset:x_offset+target.shape[1]] = target
    x_offset += x_spacing + target.shape[1]
    img2[y_offset:y_offset+47, x_offset:x_offset+m_estimate.shape[1]] = m_estimate
    x_offset += x_spacing + m_estimate.shape[1]
    img2[y_offset:y_offset+47, x_offset:x_offset+james_stein.shape[1]] = james_stein

    legend_width = quantile.shape[1] + looe.shape[1] + ordinal.shape[1] + x_spacing*3
    center = math.floor(im_v.shape[1]/2 - legend_width/2)

    x_offset = center
    y_offset = 1597
    img2[y_offset:y_offset+47, x_offset:x_offset+quantile.shape[1]] = quantile
    x_offset += x_spacing + quantile.shape[1]
    img2[y_offset:y_offset+47, x_offset:x_offset+looe.shape[1]] = looe
    x_offset += x_spacing + looe.shape[1]
    img2[y_offset:y_offset+51, x_offset:x_offset+ordinal.shape[1]] = ordinal

    steel_blue = [180, 130, 70]
    cv2.rectangle(img2,(-40,-40),(1400, 700),(steel_blue), 8)
    cv2.rectangle(img2,(1400, -40), (3495, 700),(steel_blue), 8)
    cv2.rectangle(img2,(-40,700),(1400, 1500),(steel_blue), 8)
    cv2.rectangle(img2,(1400, 700), (3495, 1500),(steel_blue), 8)
    img2 = img2[0:img2.shape[0], 0:img2.shape[1]]
    
    cv2.imwrite(f'plots/{title}.jpg', img2)

  # Comparativo Smooth
  if plot_type == 'smooth_comparison':
    img2 = deepcopy(im_v)
    crop = im_v[956:982, 122:915]

    img2 = cv2.rectangle(img2,(80,452),(1900,500),(255,255,255),-1)
    img2 = cv2.rectangle(img2,(122,956),(1900,981),(255,255,255),-1)

    center = math.floor(im_v.shape[1]/2 - crop.shape[1]/2)
    img2[970:996, center:(center+793)] = crop

    steel_blue = [180, 130, 70]
    cv2.rectangle(img2,(-40,-40),(956, 475),(steel_blue), 4)
    cv2.rectangle(img2,(956, -40), (2000, 475),(steel_blue), 4)
    cv2.rectangle(img2,(-40, 475), (956, 961) ,(steel_blue), 4)
    cv2.rectangle(img2,(956, 475), (2000, 961),(steel_blue), 4)

    cv2.imwrite(f'plots/{title}.jpg', img2)

if __name__ == '__main__':
    print('Generating plots...')

    boxplot("resultados/CV_co2_emissions.csv", "RMSE", "KNN", "CO2 Emissions", show_legend=False)
    boxplot("resultados/CV_airbnb.csv", "RMSE", "KNN", "Airbnb Reviews", show_legend=False)
    boxplot("resultados/CV_bike_sharing.csv", "RMSE", "KNN", "Bike Sharing Demand", img_height=1000, show_legend=True)
    boxplot("resultados/CV_sp_air.csv", "RMSE", "KNN", "PM10 SP Air", img_height=1000, RMSE_range = [1.3,1.7], MAE_range = [1.1,1.65], show_legend=True)

    subplot_boxplot_smooth("resultados/CV_airbnb.csv", 'resultados/CV_bike_sharing.csv', 'resultados/CV_co2_emissions.csv', 'resultados/CV_sp_air.csv', "RMSE", "KNN", "Ours results varying the smooth values")

    encoders_images =  ['plots/Plotly boxplot_RMSEs_KNN_Airbnb Reviews.png',
                        'plots/Plotly boxplot_RMSEs_KNN_CO2 Emissions.png',
                        'plots/Plotly boxplot_RMSEs_KNN_Bike Sharing Demand.png',
                        'plots/Plotly boxplot_RMSEs_KNN_PM10 SP Air.png'
                       ]

    smooth_images = ['plots/Smooth_RMSEs_Airbnb Reviews.png',
                     'plots/Smooth_RMSEs_CO2 Emissions.png',
                     'plots/Smooth_RMSEs_Bike Sharing Demand.png',
                     'plots/Smooth_RMSEs_PM10 SP Air.png'
                    ]

    concatenate_images(encoders_images, title='Comparativo_encoders', plot_type='encoders_comparison')
    concatenate_images(smooth_images, title='Comparativo_smooths', plot_type='smooth_comparison')

    print('Plots generated')