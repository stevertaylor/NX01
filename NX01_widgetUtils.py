import ipywidgets as widgets
from ipywidgets import *


class NX01gui(object):

    from_h5 = None
    psrlist = None
    nmodes = None
    cadence = None
    dmVar = None
    sampler = None
    writeHotChains = None
    resume = None
    incGWB = None
    gwbSpecModel = None
    incCorr = None
    gwbPointSrc = None
    redSpecModel = None
    dmSpecModel = None
    dirExt = None
    num_gwfreq_wins = None
    LMAX = None
    noPhysPrior = None
    miCorr = None
    use_gpu = None
    fix_slope = None
    limit_or_detect_gwb = None
    limit_or_detect_red = None
    limit_or_detect_dm = None
    anis_modefile = None
    fullN = None
    fixRed = None
    fixDM = None
    psrStartIndex = None
    psrEndIndex = None
    psrIndices = None
    det_signal = None
    bwm_search = None
    bwm_antenna = None
    bwm_model_select = None
    cgw_search = None
    ecc_search = None
    epochTOAs = None
    psrTerm = None
    periEv = None
    
    def __init__(self):

        self.from_h5 = False
        self.psrlist = None
        self.nmodes = None
        self.cadence = None
        self.dmVar = False
        self.sampler = 'ptmcmc'
        self.writeHotChains = False
        self.resume = False
        self.incGWB = False
        self.gwbSpecModel = 'powerlaw'
        self.incCorr = False
        self.gwbPointSrc = False
        self.redSpecModel = 'powerlaw'
        self.dmSpecModel = 'powerlaw'
        self.dirExt = None
        self.num_gwfreq_wins = 1
        self.LMAX = 0
        self.noPhysPrior = False
        self.miCorr = False
        self.use_gpu = False
        self.fix_slope = False
        self.limit_or_detect_gwb = 'limit'
        self.limit_or_detect_red = 'limit'
        self.limit_or_detect_dm = 'limit'
        self.anis_modefile = None
        self.fullN = True
        self.fixRed = False
        self.fixDM = False
        self.psrStartIndex = 0
        self.psrEndIndex = 18
        self.psrIndices = None
        self.det_signal = False
        self.bwm_search = False
        self.bwm_antenna = 'quad'
        self.bwm_model_select = False
        self.cgw_search = False
        self.ecc_search = False
        self.epochTOAs = False
        self.psrTerm = False
        self.periEv = False


    def makeGUI(self):

        fromh5Check = widgets.Checkbox(description='HDF5 files')
        psrlistText = widgets.Text(description='Info file:',
                                   value='/home/NX01/PsrListings_GWB.txt',font_size=15)
        preamble = widgets.VBox(children=[fromh5Check,psrlistText])

        def change_fromh5(name, value):
            self.from_h5 = value
        fromh5Check.on_trait_change(change_fromh5, 'value')

        def change_psrlist(name, value):
            self.psrlist = value
        psrlistText.on_trait_change(change_psrlist, 'value')
        
        page0 = Box(children=[preamble], padding=4)
        page0.padding=20
        page0.font_size=20

        ############
        form = widgets.VBox()
        
        modes = widgets.Text(description="Number of Fourier modes:", font_size=20,width='20%')

        def change_nmodes(name, value):
            self.nmodes = value
        modes.on_trait_change(change_nmodes, 'value')

        red = widgets.Checkbox(description="Red noise", value=False)
        red_prior = widgets.Dropdown(description="Prior",\
                                     options=['detect', 'limit'],\
                                     font_size=20)
        red_specmodel = widgets.Dropdown(description="Spectral model",\
                                         options=['power-law', 'free-spectral'],\
                                         font_size=20)
        red_info = widgets.HBox(visible=False, children=[red_prior,red_specmodel], padding=20)

        def change_red(name, value):
            if value:
                self.fixRed = False
            else:
                self.fixRed = True
        red.on_trait_change(change_red, 'value')

        def change_redprior(name, value):
            self.limit_or_detect_red = value
        red_prior.on_trait_change(change_redprior, 'value')

        def change_redspecmodel(name, value):
            self.redSpecModel = value
        red_specmodel.on_trait_change(change_redspecmodel, 'value')
        
        dm = widgets.Checkbox(description="DM variations", value=False)
        dm_prior = widgets.Dropdown(description="Prior",\
                                    options=['detect', 'limit'],\
                                    font_size=20)
        dm_specmodel = widgets.Dropdown(description="Spectral model",\
                                        options=['power-law', 'free-spectral'],\
                                        font_size=20)
        dm_info = widgets.HBox(visible=False, children=[dm_prior,dm_specmodel], padding=20)

        def change_dm(name, value):
            self.dmVar = value
        dm.on_trait_change(change_dm, 'value')
        
        def change_dmprior(name, value):
            self.limit_or_detect_dm = value
        dm_prior.on_trait_change(change_dmprior, 'value')

        def change_dmspecmodel(name, value):
            self.dmSpecModel = value
        dm_specmodel.on_trait_change(change_dmspecmodel, 'value')


        form.children = [modes,red,red_info,dm,dm_info]

        def on_red_toggle(name, value):
            if value:
                red_info.visible = True
            else:
                red_info.visible = False
        red.on_trait_change(on_red_toggle, 'value')

        def on_dm_toggle(name, value):
            if value:
                dm_info.visible = True
            else:
                dm_info.visible = False
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
