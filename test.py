from data_providers import CatsDogsDataProvider
import matplotlib.pyplot as plt
import numpy as np
 
data_prov = CatsDogsDataProvider()
for i in range(1000):
    x = data_prov.next()
