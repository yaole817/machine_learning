import pandas as pd

class ReadData(object):
	def __init__(self,userInfoPath,dataNames):
		self._dataFrame = pd.read_table(userInfoPath, names = dataNames, sep=',')

	@property
	def dataFrame(self):
		return self._dataFrame
	@dataFrame.setter
	def dataFrame(self,value):
		self._dataFrame=self._dataFrame


userInfoPath= "../data/train/browse_history_train.txt"
if __name__ == '__main__':
	
	rd = ReadData(userInfoPath,['id','timeStamp','browse','browseChild'])
	df = rd.dataFrame
	print(df)

