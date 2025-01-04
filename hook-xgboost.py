from PyInstaller.utils.hooks import collect_all, collect_data_files
import os
import xgboost

datas, binaries, hiddenimports = collect_all('xgboost')

xgboost_dir = os.path.dirname(xgboost.__file__)
binaries.append((os.path.join(xgboost_dir, 'lib', 'xgboost.dll'), '.'))