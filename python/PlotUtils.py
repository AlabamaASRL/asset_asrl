import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots

colpal = sns.color_palette

class APlot:
    def __init__(self,rows=1,cols=1):
        self.Lines  ={}
        self.Points =[]
        self.rows=rows
        self.cols=cols
        
    def AddLine(self,name,x,y,z=None,color='blue',marker=None,linewidth=None,row=1,col=1):
        if(z==None):z=np.full_like(y,0)
        self.Lines[name] ={'name':name,'x':x,'y':y,'z':z,'row':row,'col':col}
        
    def plot2d_plotly(self,name="default",xlabels=['X'],ylabels=['Y']):
        fig = make_subplots(rows=self.rows, cols=self.cols)
        fig.update_layout(template="plotly_dark")

        for name in self.Lines.keys():
            D = self.Lines[name]
            fig.add_trace(go.Scatter(x=D['x'],y=D['y'],name=name),row=D['row'],col=D['col'])
            
        fig.show()
