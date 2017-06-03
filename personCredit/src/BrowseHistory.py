import FileOperator
import pandas as pd
class BrowseHistory:
	def __init__(self,browseList):
		self.dataFrame = pd.read_table(userInfoPath,names = ['id','timeStamp','browse','browseChild'],sep=',')

	def parseData(self):
		print self.dataFrame

userInfoPath= "../data/train/browse_history_train.txt"

if __name__ == '__main__':
	browseHistory = BrowseHistory(userInfoPath)
	browseHistory.parseData()
