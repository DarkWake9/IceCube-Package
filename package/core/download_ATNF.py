import requests
import pandas as pd
import os

if 'ATNF.csv' not in os.listdir(os.getcwd() + '/data/'):
    # atnfurl = 'https://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php?version=1.70&Name=Name&RaJD=RaJD&DecJD=DecJD&P0=P0&Dist=Dist&Dist_DM=Dist_DM&startUserDefined=true&c1_val=&c2_val=&c3_val=&c4_val=&sort_attr=jname&sort_order=asc&condition=P0+%3C+30&pulsar_names=&ephemeris=short&coords_unit=raj%2Fdecj&radius=&coords_1=&coords_2=&style=Long+with+last+digit+error&no_value=*&fsize=3&x_axis=&x_scale=linear&y_axis=&y_scale=linear&state=query&table_bottom.x=55&table_bottom.y=23'

    # r = requests.get(atnfurl, allow_redirects=True)
    # data = html2text.html2text(r.text)
    # data = data.split('\n')
    # data = data[12:-18]
    # with open('temp', 'w') as f:
    #     for item in data:
    #         f.write(item + '\n')
    # cols = '     NAME        del1             RAJD      del2           DECJD       del3         P0                del4         sht5           DIST   DIST_DM  sht6'.split()
    # mspdata = pd.read_csv('temp', delim_whitespace=True, header = None, names=cols, index_col=0)
    # mspdata = mspdata.drop(['del1', 'del2', 'del3', 'del4', 'sht5', 'sht6'], axis=1)
    # mspdata.to_csv(os.getcwd() + '/data/ATNF.csv', index=False)
    # # os.remove('temp')

    r = requests.get('https://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php?version=1.70&table_top.x=48&table_top.y=29&Name=Name&RaJD=RaJD&DecJD=DecJD&P0=P0&S1400=S1400&Dist=Dist&Dist_DM=Dist_DM&startUserDefined=true&c1_val=&c2_val=&c3_val=&c4_val=&sort_attr=jname&sort_order=asc&condition=&pulsar_names=&ephemeris=short&coords_unit=raj%2Fdecj&radius=&coords_1=&coords_2=&style=Short+csv+without+errors&no_value=*&fsize=3&x_axis=&x_scale=linear&y_axis=&y_scale=linear&state=query')


    with open(os.getcwd() + '/data/temp.txt', 'w') as f:
        f.writelines(r.text)

    f.close()    
    temp = r.text.split('\n')
    temp = temp[30:-6]
    temp.pop(1)
    temp2 = [i.split(';') for i in temp if i!='']
    dat = pd.DataFrame(temp2[1:],columns=temp2[0])
    dat2 = dat['NAME;RAJD;DECJD;P0;S1400;DIST;DIST_DM;'.split(';')]
    dat2.to_csv(os.getcwd() + '/data/ATNF.csv', index=False)
    os.remove(os.getcwd() + '/data/temp.txt')

else:
    pass
