# Databricks notebook source
import numpy as np
import matplotlib.pyplot as plt

X = np.random.randn(100)
y = np.random.randn(100)

fig, ax = plt.subplots()
ax.scatter(X, y)
# display(fig)

# COMMAND ----------

