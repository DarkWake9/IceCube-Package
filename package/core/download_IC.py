############################################################################################################################
# This code downloads the 2008-2018 data release file from the IceCube website

# The data is stored in the data folder
############################################################################################################################

import requests
import os
import wget

from zipfile import ZipFile


if 'icecube_10year_ps' not in os.listdir(os.getcwd() + '/data/') or os.listdir(os.getcwd() + '/data/icecube_10year_ps') == []:
   url = 'http://icecube.wisc.edu/data-releases/20210126_PS-IC40-IC86_VII.zip'
   wget.download(url, os.getcwd() + '/data/data.zip')

   r = requests.get(url, allow_redirects=True)


   with ZipFile(os.getcwd() + '/data/data.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
      zipObj.extractall(os.getcwd() + '/data/')


   # Delete the zip file
   os.remove(os.getcwd() + '/data/data.zip')

   print('Data downloaded successfully')

else:
   pass

