import pandas as pd
from ReadData import ReadData

class BrowseHistory():
	def __init__(self,infoPath):
		rd = ReadData(infoPath,['id','timeStamp','browse','browseChild'])
		self.dataFrame = rd.dataFrame

	def parseData(self):
		print self.dataFrame

userInfoPath= "../data/train/browse_history_train.txt"

if __name__ == '__main__':
	browseHistory = BrowseHistory(userInfoPath)
	browseHistory.parseData()
