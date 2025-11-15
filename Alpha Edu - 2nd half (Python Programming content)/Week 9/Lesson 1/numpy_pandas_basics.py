import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NumPyPandasBasics:
    """
    This class contains learning examples for example libraries Numpy, Pandas and MatplotLib.
    examples are oriented for new users and shows practical usage.
    """
    def __init__(self):
        print("Hello! Today we are learning Numpy, Pandas, Matplotlib")

    def numpy_example(self):
        """
        Example of using the library NumPy with simple
        :return:
        """
        print("\n Imagine, that you have temperatures in celcium for week: 20, 22, 19, 21, 23, 24, 20.")
        temperatures = np.array([20, 22, 19, 21, 23, 24, 20])
        print("Temperature for week: ", temperatures)

        #we can calculate the average easily
        print("\We can fnd mean value of temperatures for week.")
        average_temp = np.mean(temperatures)
        print(f" Mean temperature is: {average_temp:.2f} Celcium")
        # :.2f FOR rounding the result up to 2 decimal points

        # Global warming happened and increases by 5
        print("\n What if after 10 years temperature will rise by 5 degree every day?")
        future_temperatures = temperatures + 5
        print('Temperatures after 10 years: ', future_temperatures)

        #Example: find the maximum temperature
        print("\n Now we will find the maximum value in the week")
        max_temp = np.max(temperatures)
        print(f"Maximum temperature: ", max_temp)

a = NumPyPandasBasics()
a.numpy_example()