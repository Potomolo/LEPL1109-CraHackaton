import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import (AutoMinorLocator,MultipleLocator)

data = pd.read_csv("data_3d.csv")

def doubidibidi():
    for score in ["f1_score","recall_score"]:
        for regressor in ["Linear","Knn","Logistic"]:
            #pivot_z = data.pivot(index="Features",columns="Threshold",values="Linear_recall_score")
            #features = pivot_z.index.values
            features = data["Features"]
            #thresholds = pivot_z.columns.values
            thresholds = data["Threshold"]
            #z = np.array(z).reshape((2,-1))
            X, Y = (thresholds, features)
            lst_threshold = np.arange(0.1,0.51,0.01)
            
            X = X.to_numpy().reshape((22,41))
            Y = Y.to_numpy().reshape((22,41))

            z = data[f"{regressor}_{score}"].to_numpy().reshape((22,41))
            #z = np.array(z).reshape((-1,2))
            
            plt.figure(figsize = (12,8))
            plt.title(f"{score} with differents threshold for {regressor} regression", fontsize=14)

            plt.contourf(X, Y, z, 10, alpha=1, cmap='viridis')
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=16)
            plt.contour(X, Y, z, 10, alpha =1, cmap='hot', linewidths=1)
            plt.xlabel(r"Thresholds", fontsize=16)
            plt.ylabel(r"Features", fontsize=16)

            plt.show()
            plt.close()

            ############################################################
            ############################################################

            from matplotlib.ticker import LinearLocator

            fig, (ax1,ax2) = plt.subplots(1,2, subplot_kw={"projection": "3d"}, figsize = (30,15))
            plt.title(f"{score} with differents threshold for {regressor} regression", fontsize=14)

            surf = ax1.plot_surface(X, Y, z, cmap='viridis',
                                linewidth=0, antialiased=False)

            ax1.zaxis.set_major_locator(LinearLocator(10))
            ax1.zaxis.set_major_formatter('{x:.02f}')
            plt.ylabel(r"Thresholds", fontsize=16)
            plt.xlabel(r"Features", fontsize=16)

            surf = ax2.plot_surface(Y.T, X.T, z.T, cmap='viridis',
                                linewidth=0, antialiased=False)

            ax2.zaxis.set_major_locator(LinearLocator(10))
            ax2.zaxis.set_major_formatter('{x:.02f}')
            fig.colorbar(surf, shrink=0.5, aspect=5)

            plt.show()


def bidibididou():
    for regressor in ["Linear","Knn","Logistic"]:
        #pivot_z = data.pivot(index="Features",columns="Threshold",values="Linear_recall_score")
        #features = pivot_z.index.values
        features = data["Features"]
        #thresholds = pivot_z.columns.values
        thresholds = data["Threshold"]
        X, Y = (thresholds, features)
        X = X.to_numpy().reshape((22,41))
        Y = Y.to_numpy().reshape((22,41))

        zF = data[f"{regressor}_f1_score"].to_numpy().reshape((22,41))
        zR = data[f"{regressor}_recall_score"].to_numpy().reshape((22,41))
        

        for i in range(len(zF)):
                new_order = np.lexsort([zF[i], zR[i]])
                plt.plot(zF[i][new_order],zR[i][new_order])        
        
        plt.show()

bidibididou()
