import FileOperator

class UserInfo:
	def __init__(self,userInfoList):
		self.userInfoList = userInfoList
		__id = 0
		__gender = 0
		__profession = 0
		__eduction = 0
		__marital =0
		__residence = 0
	def parseData(self):
		for line in self.userInfoList:
			line = line.strip()   # remove '\n' in each line
			[self.__id,self.__gender,self.__profession,self.__eduction,self.__marital,self.__residence] = line.split(',')
			print self.__residence

userInfoPath= "../data/train/user_info_train.txt"

if __name__ == '__main__':
	userInfoList = FileOperator.readFile(userInfoPath)
	userInfo = UserInfo(userInfoList)
	userInfo.parseData()
