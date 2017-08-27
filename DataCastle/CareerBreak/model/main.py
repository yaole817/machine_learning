import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np

dataPath = '../data/pfm_train.csv'
dataNewPath = "pfm_train_filter.csv"
headers = "Age,Attrition,BusinessTravel,Department,DistanceFromHome,Education,EducationField,EmployeeNumber,EnvironmentSatisfaction,Gender,JobInvolvement,JobLevel,JobRole,JobSatisfaction,MaritalStatus,MonthlyIncome,NumCompaniesWorked,Over18,OverTime,PercentSalaryHike,PerformanceRating,RelationshipSatisfaction,StandardHours,StockOptionLevel,TotalWorkingYears,TrainingTimesLastYear,WorkLifeBalance,YearsAtCompany,YearsInCurrentRole,YearsSinceLastPromotion,YearsWithCurrManager".split(",")
xVars = "Age,BusinessTravel,Department,DistanceFromHome,Education,EducationField,EmployeeNumber,EnvironmentSatisfaction,Gender,JobInvolvement,JobLevel,JobRole,JobSatisfaction,MaritalStatus,MonthlyIncome,NumCompaniesWorked,Over18,OverTime,PercentSalaryHike,PerformanceRating,RelationshipSatisfaction,StandardHours,StockOptionLevel,TotalWorkingYears,TrainingTimesLastYear,WorkLifeBalance,YearsAtCompany,YearsInCurrentRole,YearsSinceLastPromotion,YearsWithCurrManager".split(",")
yVars = ["Attrition"]
keyDict = {
    
    "Non-Travel":1,
    "Travel_Rarely":2,
    "Travel_Frequently":3,

    "LifeSciences":1,
    "Medical":2,
    "Marketing":3,
    "TechnicalDegree":4,
    "HumanResources":5,
    "Other":6,

    "Male":1,
    "Female":2,

    "SalesExecutive":1,
    "ResearchScientist":2,
    "LaboratoryTechnician":3,
    "ManufacturingDirector":4,
    "HealthcareRepresentative":5,
    "Manager":6,
    "SalesRepresentative":7,
    "ResearchDirector":8,
    "HumanResources":9,

    "Single":1,
    "Married":2,
    "Divorced":3,

    "Sales":1,
    "Research&Development":2,
    "HumanResources":3,

    "Yes":1,
    "No":2, 
    "Y":1,
    "N":2,
}

def filterData():
    data = open(dataPath).read()
    data = data.replace(" ",'')
    for key in keyDict:
        data = data.replace(key,str(keyDict[key]))
    with open("pfm_train_filter.csv",'w') as f:
        f.write(data)

if __name__ == "__main__":
    data = pd.read_csv(dataNewPath)
    X = data[xVars]
    Y = data[yVars]
    xTrain,xTest,yTrain,yTest = train_test_split(X, Y, random_state = 0)

    lr = LR()
    lr.fit(xTrain,yTrain)
    yPred = lr.predict(xTest)


    yTest = np.array(yTest.T)
    yPred = np.array(yPred)
    print(yTest)
    print(yPred)
    correct = [1 if a == b else 0 for (a,b) in zip(yPred,yTest[0])]
    accuracy = (sum(map(int,correct))/float(len(correct)))
    print("the accuracy is %f"%(accuracy))
    # print(abs(yTest-yPred).sum()/len(yPred))
    # print(yTest)
    # print(yPred)

    # plt.figure()
    # plt.scatter(range(len(yPred)),yPred)
    # plt.scatter(range(len(yTest)),yTest) 
    # # plt.legend(loc = 'upper right')
    # # plt.xlabel("sample numbers")
    # # plt.ylabel("values")
    # plt.show()

    # plt.figure()
    # plt.scatter(data.Age,data.Attrition)
    # # plt.plot(range(len(yTest)),yTest,'r',label="test") 
    # # plt.legend(loc = 'upper right')
    # # plt.xlabel("sample numbers")
    # # plt.ylabel("values")
    # plt.show()

        


