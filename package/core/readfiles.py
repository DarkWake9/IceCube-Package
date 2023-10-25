import pandas as pd
import os

#### IMPORTING AND SPLITTING ICDATA $$$
# master_path = os.getcwd()




class Data(): # type: ignore
    def __init__(self, master_path = "/media/darkwake/VIB2/Project-IceCube/"):
        # self.icdata = []
        # self.uptdata = []
        # self.eadata = []
        self.path = master_path + "icecube_10year_ps/events"
        self.t_eff_path = master_path + "icecube_10year_ps/uptime"
        self.irf_path = master_path + "icecube_10year_ps/irfs"
        self.filenames = ["IC40_exp.csv", "IC59_exp.csv","IC79_exp.csv", "IC86_I_exp.csv", "IC86_II_exp.csv",
        "IC86_III_exp.csv", "IC86_IV_exp.csv", "IC86_V_exp.csv", "IC86_VI_exp.csv", "IC86_VII_exp.csv"]


    # def readfiles(self):
        file = self.filenames[0]
        # f = open(os.path.join(self.path, file), 'r')
        # lines = f.readlines()
        # column=lines[0].split()
        # icdata = []
        # try:
        #     column.remove('#')
        # except:
        #     pass
        icdata = pd.read_csv(os.path.join(self.path, file), sep="\s+", comment="#", names="MJD[days]	log10(E/GeV)	AngErr[deg]	RA[deg]	Dec[deg]	Azimuth[deg]	Zenith[deg]".split("\t"), dtype=float, skiprows=1)
        content = []
        season_length = [len(icdata.index)]
        for file in self.filenames[1:]:
        #     f = open(os.path.join(self.path, file), 'r')
        #     lines = f.readlines()
        # #print(len(lines) - 1)
        #     for line in lines[1:]:
        #         content.append(line.split())
        #     f.close()
        #     season_length.append(len(content))
            content = pd.read_csv(os.path.join(self.path, file), sep="\s+", comment="#", names="MJD[days]	log10(E/GeV)	AngErr[deg]	RA[deg]	Dec[deg]	Azimuth[deg]	Zenith[deg]".split("\t"), dtype=float, skiprows=1)
            season_length.append(len(content.index))
            icdata = pd.concat([icdata, content], ignore_index=True)
        # icdata = pd.DataFrame(content, columns=column, dtype=float)#.convert_dtypes(infer_objects=True,convert_integer=True,convert_floating=True)
        
        icdata['log10(E/GeV)'] = [float(i) for i in icdata['log10(E/GeV)']]
        icdata['MJD[days]'] = [float(i) for i in icdata['MJD[days]']]

        print("read icdata")
        # f.close()
        self.icdata = icdata
        self.season_length = season_length

#Importing UPtime data
# def uptime(self):
        file = self.filenames[0]
        f = open(os.path.join(self.t_eff_path, file), 'r')
        lines = f.readlines()
        column=lines[0].split()
        column.pop(0)
        uptdata = []
        for file in self.filenames:
            content = []
            f = open(os.path.join(self.t_eff_path, file), 'r')
            lines = f.readlines()
            for line in lines[1:]:
                content.append(line.split())
            f.close()
            temp = pd.DataFrame(content, columns=column)
            temp['MJD_start[days]'] = [float(i) for i in temp['MJD_start[days]']]
            temp['MJD_stop[days]'] = [float(i) for i in temp['MJD_stop[days]']]
            uptdata.append(temp)
            temp = []
            content = []
        f.close()
        # return uptdata
        self.uptdata = uptdata
        print("read uptdata")

# def aeffdata(self):

#Importing Aeff data
        filenames = ["IC40_effectiveArea.csv", "IC59_effectiveArea.csv","IC79_effectiveArea.csv", "IC86_I_effectiveArea.csv", "IC86_II_effectiveArea.csv"]
        file = filenames[0]
        f = open(os.path.join(self.irf_path, file), 'r')
        lines = f.readlines()
        column=lines[0].split()
        column.pop(0)
        eadata = []
        for file in filenames:
            content = []
            f = open(os.path.join(self.irf_path, file), 'r')
            lines = f.readlines()
            for line in lines[1:]:
                content.append(line.split())
            f.close()
            temp = pd.DataFrame(content, columns=column, dtype=float)
            #temp['MJD_start[days]'] = [float(i) for i in temp['MJD_start[days]']]
            #temp['MJD_stop[days]'] = [float(i) for i in temp['MJD_stop[days]']]
            eadata.append(temp)
            temp = []
            content = []
        f.close()
        # return uptdata
        self.eadata = eadata
        print("read eadata")



        # #IMPORTING MSPDATA
        # f = open(master_path + "allpsr1.68.txt", 'r')
        # lines = f.readlines()
        # content=[]
        # column=lines.pop(0).replace('x', '').replace('#', '').split()
        # for line in lines[:]:
        #     content.append(line.split())
        #     #the INITAL DATABASE IS CLUTTERED SO WE REMOVE THE NULL COLUMNS AND OTHER CLUTTER
        # f.close()
        # mspdata = pd.DataFrame(content).drop(0, axis=1)#.dropna()#.drop_duplicates()#.drop(range(0,6)).dropna()

        # line = []
        # lines = []
        # mspdata.columns = column
        # column = []
        # content=[]
        # #mspdata = mspdata.sort_values('DECJD')
        # mspdata.dropna(inplace=True)
        # #mspdata = mspdata.reset_index()
        # #mspdata = mspdata.drop("index", axis=1)
        # self.mspdata = mspdata
        # print("read mspdata")

        #IMPORTING MSPDATA
        mspdata = pd.read_csv(master_path + "ATNF.csv")
        # mspdata = pd.read_csv(master_path + "ATNF_rem_psr_s1400_1050.csv") ##25102023 REMOVED PULSARS WITH S1400 = 1050
        # mspdata = pd.read_csv(master_path + "ATNF_max5.txt")
        # mspdata = mspdata[mspdata['DIST_DM'] != '*']
        # mspdata = mspdata[mspdata['S1400'] != '*']
        # mspdata = mspdata[abs(mspdata['DECJD'].values) > 3]
        self.mspdata = mspdata
        print("read mspdata")


# print(Data('/media/darkwake/VIB2/IceCube-Package/Project-IceCube/package/data/').season_length)