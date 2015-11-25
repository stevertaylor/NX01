import ipywidgets as widgets
from ipywidgets import *



def NX01opts():

    preamble = widgets.VBox(children=[Checkbox(description='HDF5 files'),
                                      Text(description='Info file:',
                                           value='/home/NX01/PsrListings_GWB.txt')])
    page0 = Box(children=[preamble], padding=4)

    ############
    form = widgets.VBox()
    red = widgets.Checkbox(description="Red noise", value=False)
    dm = widgets.Checkbox(description="DM variations", value=False)

    red_info = widgets.HBox(visible=False, \
                            children=[widgets.Dropdown(description="prior",\
                                                       options=['detect', 'limit'],
                                                       padding=10),\
                                      widgets.Dropdown(description="spectral model",\
                                                       options=['power-law', 'free-spectral'],
                                                       padding=10)],\
                                                       padding=15)

    dm_info = widgets.HBox(visible=False, \
                            children=[widgets.Dropdown(description="prior",\
                                                       options=['detect', 'limit'],
                                                       padding=10),\
                                      widgets.Dropdown(description="spectral model",\
                                                       options=['power-law', 'free-spectral'],
                                                       padding=10)],\
                                                       padding=15)

    form.children = [red,red_info,dm,dm_info]

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
    ###########

    ###########
    form = widgets.VBox()

    gwb = Checkbox(description='Stochastic gravitational-wave background', value=False)
    gwb_preamble = widgets.HBox(visible=True, \
                            children=[widgets.Dropdown(description="prior",\
                                                       options=['detect', 'limit'],
                                                       padding=10),\
                                      widgets.Dropdown(description="spectral model",\
                                                       options=['power-law', 'free-spectral'],
                                                       padding=10)],\
                                                       padding=15)
    incCorr = widgets.Checkbox(description='Include correlations', value=False)
    corrOpts = widgets.Dropdown(visible=False, description='Type:', options=['Spherical-harmonic anisotropy',
                                                                             'Direct cross-correlation recovery',
                                                                             'Stochastic point-source background'])

    gwb_info = widgets.VBox(visible=False, children=[gwb_preamble, incCorr, corrOpts])

    form.children = [gwb,gwb_info]

    def on_gwb_toggle(name, value):
        if value:
            gwb_info.visible = True
        else:
            gwb_info.visible = False

    def on_corr_toggle(name, value):
        if value:
            corrOpts.visible = True
        else:
            corrOpts.visible = False
        
    gwb.on_trait_change(on_gwb_toggle, 'value')
    incCorr.on_trait_change(on_corr_toggle, 'value')
    
    page2 = Box(children=[form], padding=4)
    ###############

    ###########
    form = widgets.VBox()
    detsig = Checkbox(description='Determinstic GW signal', value=False)

    detsig_info = widgets.VBox(visible=False, \
                            children=[widgets.Dropdown(description="signal type",\
                                                       options=['circular binary', 'eccentric binary', 'burst with memory'],
                                                       padding=5),\
                                      widgets.Checkbox(description='Use epoch TOAs', value=False)],\
                                      padding=15)

    form.children = [detsig,detsig_info]

    def on_detsig_toggle(name, value):
        if value:
            detsig_info.visible = True
        else:
            detsig_info.visible = False
        
    detsig.on_trait_change(on_detsig_toggle, 'value')
    page3 = Box(children=[form], padding=4)

    ##########
    tabs = widgets.Tab(children=[page0, page1, page2, page3])
    tabs.set_title(0, 'Pulsars')
    tabs.set_title(1, 'Pulsar properties')
    tabs.set_title(2, 'Stochastic GW signals')
    tabs.set_title(3, 'Deterministic GW signals')

    tabs.font_weight='bolder'
    tabs.font_style='italic'
    tabs.padding=30


    return tabs
