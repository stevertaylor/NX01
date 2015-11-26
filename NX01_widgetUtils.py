from __future__ import print_function
import ipywidgets as widgets
from ipywidgets import *


def NX01opts():

    preamble = widgets.VBox(children=[Checkbox(description='HDF5 files'),
                                      Text(description='Info file:',
                                           value='/home/NX01/PsrListings_GWB.txt',font_size=15)])
    page0 = Box(children=[preamble], padding=4)
    page0.padding=20
    page0.font_size=20

    ############
    form = widgets.VBox()
    modes = widgets.Text(description="Number of Fourier modes:", value='30', font_size=20,width='20%')
    red = widgets.Checkbox(description="Red noise", value=False)
    dm = widgets.Checkbox(description="DM variations", value=False)

    red_info = widgets.HBox(visible=False, \
                            children=[widgets.Dropdown(description="Prior",\
                                                       options=['detect', 'limit'],
                                                       font_size=20),\
                                      widgets.Dropdown(description="Spectral model",\
                                                       options=['power-law', 'free-spectral'],
                                                       font_size=20)],\
                                                       padding=20)

    dm_info = widgets.HBox(visible=False, \
                            children=[widgets.Dropdown(description="Prior",\
                                                       options=['detect', 'limit'],
                                                       font_size=20),\
                                      widgets.Dropdown(description="Spectral model",\
                                                       options=['power-law', 'free-spectral'],
                                                       font_size=20)],\
                                                       padding=20)

    form.children = [modes,red,red_info,dm,dm_info]

    def on_red_toggle(name, value):
        if value:
            red_info.visible = True
        else:
            red_info.visible = False

    def on_dm_toggle(name, value):
        if value:
            dm_info.visible = True
        else:
            dm_info.visible = False
            

    red.on_trait_change(on_red_toggle, 'value')
    dm.on_trait_change(on_dm_toggle, 'value')

    page1 = Box(children=[form], padding=4)
    page1.padding=20
    page1.font_size=20
    ###########

    ###########
    form = widgets.VBox()

    gwb = Checkbox(description='Stochastic gravitational-wave background', value=False)
    gwb_preamble = widgets.HBox(visible=True, \
                            children=[widgets.Dropdown(description="Prior",\
                                                       options=['detect', 'limit'],font_size=20),\
                                      widgets.Dropdown(description="Spectral model",\
                                                       options=['power-law', 'free-spectral'],font_size=20)],padding=20)
    fixSlope = widgets.Checkbox(description='Fix PSD slope to 13/3', value=False)
    incCorr = widgets.Checkbox(description='Include correlations', value=False)
    corrOpts = widgets.Dropdown(visible=False, description='Type:', options=['Isotropic',
                                                                             'Spherical-harmonic anisotropy',
                                                                             'Direct cross-correlation recovery',
                                                                             'Stochastic point-source background'],
                                                                             font_size=20)
    anisLmax = widgets.Text(visible=False, description='lmax:',font_size=20)
    noPhysPrior = widgets.Checkbox(visible=False, description='Switch off physical prior')
    anisOpts = widgets.HBox(children=[anisLmax,noPhysPrior])

    gwb_info = widgets.VBox(visible=False, children=[gwb_preamble, fixSlope, incCorr,
                                                     corrOpts, anisOpts])

    form.children = [gwb,gwb_info]

    def on_gwb_toggle(name, value):
        if value:
            gwb_info.visible = True
        else:
            gwb_info.visible = False

    def on_incCorr_toggle(name, value):
        if value:
            corrOpts.visible = True
        else:
            corrOpts.visible = False
            anisLmax.visible = False
            noPhysPrior.visible = False

    def on_corrOpts_toggle(name, value):
        if value == 'Spherical-harmonic anisotropy':
            anisLmax.visible = True
            noPhysPrior.visible = True
        else:
            anisLmax.visible = False
            noPhysPrior.visible = False
        
    gwb.on_trait_change(on_gwb_toggle, 'value')
    incCorr.on_trait_change(on_incCorr_toggle, 'value')
    corrOpts.on_trait_change(on_corrOpts_toggle, 'value')
    
    page2 = Box(children=[form], padding=4)
    page2.padding=20
    page2.font_size=20
    ###############

    ###########
    form = widgets.VBox()
    detsig = Checkbox(description='Determinstic GW signal', value=False)

    detsig_info = widgets.Dropdown(visible=False, description="Signal type",\
                                                       options=['burst with memory', 'circular binary', 'eccentric binary'],
                                                       font_size=20)
    epochOpt = widgets.Checkbox(visible=False,description='Use epoch TOAs', value=False)
    binary_info = widgets.VBox(visible=False,\
                               children=[widgets.Checkbox(description='Pulsar term', value=False),\
                                         widgets.Checkbox(description='Periapsis evolution', value=False)])

    form.children = [detsig,detsig_info,binary_info,epochOpt]

    def on_detsig_toggle(name, value):
        if value:
            detsig_info.visible = True
            epochOpt.visible = True
        else:
            detsig_info.visible = False
            epochOpt.visible = False
            binary_info.visible = False

    def on_binary_toggle(name, value):
        if 'binary' in value:
            binary_info.visible = True
            binary_info.margin=10
        else:
            binary_info.visible = False
        
    detsig.on_trait_change(on_detsig_toggle, 'value')
    detsig_info.on_trait_change(on_binary_toggle, 'value')
    
    page3 = Box(children=[form], padding=4)
    page3.padding=20
    page3.font_size=20

    ##########
    tabs = widgets.Tab(children=[page0, page1, page2, page3])
    tabs.set_title(0, 'Pulsars')
    tabs.set_title(1, 'Pulsar properties')
    tabs.set_title(2, 'Stochastic GW signals')
    tabs.set_title(3, 'Deterministic GW signals')

    tabs.font_weight='bolder'
    tabs.font_style='italic'
    tabs.padding=30
    tabs.font_size=20


    return tabs
