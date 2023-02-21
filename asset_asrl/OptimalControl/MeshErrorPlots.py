import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns    # pip install seaborn if you dont have it
import matplotlib.animation as animation


def PhaseMeshErrorPlot(phase,show=True):
    fig,axs = plt.subplots(2,1)
    axs[0].set_ylabel("Error")
    axs[0].set_yscale("log")
    axs[1].set_xlabel("t (0-1)")
    axs[0].grid(True)
    axs[1].grid(True)
    Miters = phase.getMeshIters()
    
    cols=sns.color_palette("plasma",len(Miters))
    
    tol = phase.MeshTol
    axs[0].plot([0,1],[tol, tol],color='k',linestyle='dashed',label='tol={0:.2e}'.format(tol))
    
    for i,miter in enumerate(Miters):
        axs[0].plot(miter.times,miter.error,color=cols[i],label="Iter: {}".format(i))
        axs[1].plot(miter.times,miter.distribution/max(miter.distribution),color=cols[i])

    axs[0].legend()
    
    if(show):plt.show()
    
def OCPMeshErrorPlot(ocp,show=True):
    
    for phase in ocp.Phases:
        PhaseMeshErrorPlot(phase,show=False)
        
    if(show):plt.show()