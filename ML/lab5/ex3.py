import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model

data = pd.read_csv('cygage.txt', sep='\t')

age = data.iloc[:, 0]
depth = np.array(data.iloc[:, 1]).reshape(-1, 1)
weights = data.iloc[:, 2]

reg = linear_model.LinearRegression().fit(depth, age, weights)
print(f'С весами: {reg.score(depth, age, weights)}')
# Plot outputs
sizes = [500 * w for w in weights]
plt.plot(depth.reshape(-1, 1), reg.predict(depth), color="blue", linewidth=3)

# plt.xticks(())
# plt.yticks(())

reg = linear_model.LinearRegression().fit(depth, age)
print(f'Без весов: {reg.score(depth, age)}')
# Plot outputs
plt.plot(depth.reshape(-1, 1), reg.predict(depth), color="red", linewidth=3)

# plt.xticks(())
# plt.yticks(())
plt.legend(['С весами', 'Без весов'])
plt.scatter(depth.reshape(-1, 1), age, color="black", s=sizes)
plt.xlabel('depth')
plt.ylabel('age')
plt.show()
