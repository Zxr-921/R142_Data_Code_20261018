"""Add redundant line and hatch encodings to legacy scientific figures."""
def install_styles():
    from matplotlib.axes import Axes
    if getattr(Axes,'_r142_styled',False):return
    Axes._r142_styled=True
    bar0,barh0,plot0=Axes.bar,Axes.barh,Axes.plot
    def bar(self,*args,**kwargs):
        n=getattr(self,'_r142_bar_group',0);self._r142_bar_group=n+1
        kwargs.setdefault('hatch',['','///','xx','..','\\\\','++'][n%6]);kwargs.setdefault('edgecolor','white');kwargs.setdefault('linewidth',.4)
        return bar0(self,*args,**kwargs)
    def barh(self,*args,**kwargs):
        n=getattr(self,'_r142_hbar_group',0);self._r142_hbar_group=n+1
        kwargs.setdefault('hatch',['','///','xx','..','\\\\','++'][n%6]);kwargs.setdefault('edgecolor','white');kwargs.setdefault('linewidth',.4)
        return barh0(self,*args,**kwargs)
    def plot(self,*args,**kwargs):
        lines=plot0(self,*args,**kwargs)
        for line in lines:
            if len(line.get_xdata())<2:continue
            n=getattr(self,'_r142_line_group',0);self._r142_line_group=n+1
            line.set_linestyle(['-','--',':','-.'][n%4]);line.set_marker(['o','s','^','D','v','P','X','h'][n%8]);line.set_markersize(3)
            if len(line.get_xdata())>20:line.set_markevery(max(1,len(line.get_xdata())//8))
        return lines
    Axes.bar,Axes.barh,Axes.plot=bar,barh,plot
