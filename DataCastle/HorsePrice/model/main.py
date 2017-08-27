import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

dataPath = '../data/kc_train.csv'
headers = ['date','price','bedroomNum','bathroomNum','horseArea','carArea','floor','wholeArea',
         'structureArea','basementAres','structureYear','repairYear','col','row']
xVars = ['date','bedroomNum','bathroomNum','horseArea','carArea','floor','wholeArea',
         'structureArea','basementAres','structureYear','repairYear','col','row']
yVars = ['price']

if __name__ == "__main__":
   data = pandas.read_csv(dataPath, names = headers, index_col=None)
   X = data[xVars]
   Y = data[yVars]
   xTrain,xTest,yTrain,yTest = train_test_split(X, Y, random_state = 0)

   linreg = LinearRegression()
   model = linreg.fit(xTrain,yTrain)
   
   intercept = linreg.intercept_
   coef = linreg.coef_
   print(coef,intercept)

   yPred = linreg.predict(xTest)

   plt.figure()
   plt.plot(range(len(yPred)),yPred,'b',label="predict")
   plt.plot(range(len(yTest)),yTest,'r',label="test") 
   plt.legend(loc = 'upper right')
   plt.xlabel("sample numbers")
   plt.ylabel("values")
   plt.show()



    