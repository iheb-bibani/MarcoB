import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import date,timedelta
from plotly.subplots import make_subplots
from datetime import datetime as dt

sheet_name = ['USDCAD L150 Curncy','USDNOK L150 Curncy','USDSEK L150 Curncy','USDCHF L150 Curncy',
              'USDJPY L150 Curncy','USDBRL L150 Curncy','USDMXN L150 Curncy','USDHUF L150 Curncy',
              'USDKRW L150 Curncy','USDCNH L150 Curncy','EURUSD L150 Curncy','GBPUSD L150 Curncy',
              'NZDUSD L150 Curncy','AUDUSD L150 Curncy','EURSEK L150 Curncy','EURNOK L150 Curncy',
              'EURCHF L150 Curncy','EURNZD L150 Curncy','EURAUD L150 Curncy','EURGBP L150 Curncy',
              'EURCAD L150 Curncy','EURJPY L150 Curncy','EURHUF L150 Curncy','EURBRL L150 Curncy',
              'EURMXN L150 Curncy','CHFSEK L150 Curncy','CHFNOK L150 Curncy','CHFNZD L150 Curncy',
              'CHFAUD L150 Curncy','CHFGBP L150 Curncy','CHFCAD L150 Curncy','CHFJPY L150 Curncy',
              'CHFBRL L150 Curncy','CHFMXN L150 Curncy','GBPSEK L150 Curncy','GBPNOK L150 Curncy',
              'GBPNZD L150 Curncy','GBPAUD L150 Curncy','GBPCAD L150 Curncy','GBPJPY L150 Curncy',
              'GBPBRL L150 Curncy','GBPMXN L150 Curncy','SEKJPY L150 Curncy','NOKJPY L150 Curncy',
              'NZDJPY L150 Curncy','AUDJPY L150 Curncy','CADJPY L150 Curncy','JPYHUF L150 Curncy',
              'BRLJPY L150 Curncy','MXNJPY L150 Curncy','JPYKRW L150 Curncy','AUDNZD L150 Curncy']

sheet_name = st.sidebar.selectbox('Select Your Sheet Name',sheet_name)

df = pd.read_excel('fxnfilterfx.xlsx',sheet_name= sheet_name)
df.index = df['date']

start_date = st.sidebar.date_input('Enter START Date',min(df.index),min_value=min(df.index),max_value=max(df.index))
end_date = st.sidebar.date_input('Enter END Date',max(df.index),min_value=min(df.index),max_value=max(df.index))

assets = st.sidebar.radio("Asset",('FX','Rates','CMDTS'))

molecules = st.sidebar.radio('Molecules',(''))
atoms = st.sidebar.radio('Atoms',(''))

if assets == 'FX' :

    st.subheader('TABLE')
    st.table(df.mean(axis=0))

    # Graph 1
    st.subheader('Graph 1 : Asset vs Position vs DD')

    fig = make_subplots(rows=2,cols=1)

    fig.add_trace(go.Scatter(
        name="close",
        x=df["date"], y=df['close']
    ),row=1,col=1)

    fig.add_trace(go.Scatter(
        name="Buy",
        x=df[df['pos'] == 1].index,
        y=df[df['pos'] == 1]['close'],
        mode='markers',
        customdata = df[df['pos'] == 1]['close'],
        marker_symbol='triangle-up',
        marker_color='green'),row=1,col=1
    )

    fig.add_trace(go.Scatter(
        name="Sell",
        x=df[df['pos'] == -1].index,
        y=df[df['pos'] == -1]['close'],
        mode='markers',
        customdata = df[df['pos'] == -1],
        marker_symbol='x',
        marker_color='red'),row=1,col=1

    )

    fig.add_trace(go.Scatter(
        name="drawdown",
        x=df["date"], y=df["dd"],marker_color='red'
    ),row=2,col=1)

    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    st.plotly_chart(fig)

    # Graph 2
    st.subheader('Graph 2 : cumpnl vs dd ')
    fig = px.area(df[['dd','cumpnl']],facet_col_wrap=2, color_discrete_sequence=['#636EFA', 'red'])
    st.plotly_chart(fig)

    # Graph 3
    st.subheader('Graph 3 : Asset Price , Rolling Sharpe ')

    # create first trace
    trace1 = go.Scatter(x=df['date'], y=df['close'], name='close')

    # create second trace
    trace2 = go.Scatter(x=df['date'], y=df['sharpe3m'], name='sharpe', yaxis='y2')

    # create layout with two y axes
    layout = go.Layout(title='Two Axes Plot', yaxis=dict(title='close'), yaxis2=dict(title='sharpe', overlaying='y', side='right'))

    # create figure with both traces and layout
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # show figure
    st.plotly_chart(fig)

    # Graph 4
    st.subheader('Graph 4 : Rolling Sharpe VS Rolling Std ')

    # create first trace
    trace1 = go.Scatter(x=df['date'], y=df['sharpe3m'], name='rolling sharpe')

    # create second trace
    trace2 = go.Scatter(x=df['date'], y=df['std3m'], name='rolling std', yaxis='y2')

    # create layout with two y axes
    layout = go.Layout(title='Two Axes Plot', yaxis=dict(title='rolling sharpe'), yaxis2=dict(title='rolling std', overlaying='y', side='right'))

    # create figure with both traces and layout
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    st.plotly_chart(fig)

    # Graph 5 : Pnl , Sharpe , DD
    st.subheader('Graph 5 : Pnl , Sharpe , DD')

    fig = go.Figure()

    df.index = pd.to_datetime(df['date'],format='%Y-%m-%d')

    data = ['weekly', 'monthly','quarterly']

    fig = go.Figure()

    df.index = pd.to_datetime(df['date'],format='%Y-%m-%d')

    data = ['weekly', 'monthly','quarterly']

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    fig.add_trace(
        go.Bar(name='pnl', x=data, y=[df['pnl'].tail(5).sum(),
                                      df['pnl'].tail(21).sum(),
                                      df['pnl'].tail(62).sum()]),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(name='dd', x=data, y=[df['dd'].tail(5).sum(),
                                     df['dd'].tail(21).sum(),
                                     df['dd'].tail(62).sum()]),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(name='sharpe', x=data, y=[np.sqrt(252) * df['pnl'].tail(5).mean() / df['pnl'].tail(5).std(),
                                         np.sqrt(252) * df['pnl'].tail(21).mean() / df['pnl'].tail(21).std(),
                                         np.sqrt(252) * df['pnl'].tail(62).mean() / df['pnl'].tail(62).std()
                                         ]),
        row=2, col=1
    )

    fig.update_layout(height=600, width=800, title_text="PnL and DD vs. Sharpe")

    fig.update_yaxes(title_text="PnL and DD", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe", row=2, col=1)

    st.plotly_chart(fig)

    # Table : Returns
    st.subheader('Table : Returns')
    
    df['date'] = pd.to_datetime(df['date']) 

    x1 = df.resample('Y', on='date')['pnl'].sum()
    x1 = pd.DataFrame(pd.DataFrame(x1).transpose())
    x1.columns = [2022, 2023]

    # by month
    x2 = df.resample('m', on='date')['pnl'].sum()

    x3 = x2.iloc[:2]
    x3 = pd.DataFrame(x3)

    x4 = x2.iloc[2:]

    x5 = pd.concat([x3, x4])
    x5.columns = [2022, 2023]
    x5.index = x5.index.astype(str)

    st.table(pd.concat([x5,x1])) # stopped here 

    x6 = df.groupby(df.date.dt.year)['sharpe'].mean()
    x6 = pd.DataFrame(x6).T

    x7 = df.groupby(df.date.dt.year)['maxdd'].mean()
    x7 = pd.DataFrame(x7).T

    x8 = pd.concat([x6,x7])
    #st.table(pd.concat([x1,x9]))

    #def add_datepart(df):
    #    fld=df.index
    #    df['year'] = fld.year
    #    df['month'] = fld.month
    #   return df

    #pnl_df = add_datepart(20000000*df['pnl'])
    #st.dataframe(pnl_df)
    #pivot = np.round(pd.pivot_table(pnl_df,values='pnl',
    #index = ['year'],
    #columns = ['month'],
    #aggfunc = np.sum,
    #fill_value=0,margins=True,margins_name='Total'),2)

    #st.dataframe(pivot)
