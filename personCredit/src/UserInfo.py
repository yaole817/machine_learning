import FileOperator
import pandas as pd
class UserInfo:
	def __init__(self,userInfoPath):
		self.dataFrame = pd.read_table(userInfoPath,names = ['id','gender','profession','eduction','marital','residence'],sep=',')

	def parseData(self):
		print self.dataFrame

userInfoPath= "../data/train/user_info_train.txt"

if __name__ == '__main__':
	#userInfoList = FileOperator.readFile(userInfoPath)
	userInfo = UserInfo(userInfoPath)
	userInfo.parseData()
