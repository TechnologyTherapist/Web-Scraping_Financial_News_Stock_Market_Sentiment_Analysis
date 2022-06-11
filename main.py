# This is a sample Python script.
from bs4 import BeautifulSoup
from urllib.request import urlopen,Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
import matplotlib.pyplot as plt
import pandas as pd
def main():
    url='https://finviz.com/search.ashx?p='
    tickers=['AMZN','.AMD','FB']
    news_tables={}

    for ticker in tickers:
        url+=ticker
        req=Request(url=url,headers={'user-agent':'my-app'})
        response=urlopen(req)
        # print(response)
        html=BeautifulSoup(response,features="html.parser")
        # print(html)
        news_table=html.find(id='news-table')
        news_tables[ticker]=news_table
        # print(news_table)
        break
    # print(news_tables)
    parsed_data=[]
    amzn_data=news_tables['AMZN']
    amzn_rows=amzn_data.findAll('tr')
    # print(amzn_rows)
    # for index,row in enumerate(amzn_rows):
    #     title=row.a.text

    #     timestamp=row.td.text
    #     print(timestamp+" "+title)
    for ticker,news_table in news_tables.items():
        for row in news_table.findAll('tr'):
            title=row.a.get_text()
            date_data=row.td.text.split('')

            if len(date_data)==1:
                time=date_data[0]
            else:
                date=date_data[0]
                time=date_data[1]
            parsed_data.append(ticker,date,time,title)
        print(parsed_data)
    df=pd.DataFrame(parsed_data,columns=['ticker','time','date','title'])
    vadar=SentimentIntensityAnalyzer()
    print(vadar.polarity_scores("I dont thik apple is good company.I thry think they will do poorly this quater."))
    print(df.head())
    # print(df['title'])
    f=lambda title:vadar.polarity_score(title)['compound']
    df['compound']=df['title'].apply(f)
    print(df.head())
    df['date']=pd.to_datetime(df.date).dt.date
    plt.figure(figsize=(10,6))
    mean_df=df.groupby(['ticker','date']).mean()
    print(mean_df)
    mean_df=mean_df.unstack()
    mean_df=mean_df.xs('compound',axis="columns").transpose()
    mean_df.plot(kind='bar',)
    plt.show()


if __name__=="__main__":
    main()
