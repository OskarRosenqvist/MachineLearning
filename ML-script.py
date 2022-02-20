import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd

def read_data():
    d = pd.read_csv('investor.csv', sep=',')
    df = pd.DataFrame(d)
    return df


if __name__ == '__main__':
    df = read_data()
    x = df['week']
    y = df['value']
    mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
    print(f"r = {r2_score(y, mymodel(x))}\nHow well does the prediction-model fit the data? Pretty good...")
    myline = numpy.linspace(1, 10, 7)
    print(f"Stock value next week: {mymodel(8)}")
    plt.scatter(x, y)
    plt.plot(x, mymodel(myline))
    plt.show()
