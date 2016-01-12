import ipywidgets as widgets
from ipywidgets import *
import json
import cPickle as pickle

from IPython import get_ipython
ipython = get_ipython()

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
    typeCorr = None
    redSpecModel = None
    dmSpecModel = None
    dirExt = None
    nwins = None
    LMAX = None
    noPhysPrior = None
    use_gpu = None
    fixSlope = None
    gwbPrior = None
    redPrior = None
    dmPrior = None
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
    incGWline = None
    gwlinePrior = None
    
    def __init__(self, *args, **kwargs):  

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
        self.typeCorr = 'spharmAnis'
        self.redSpecModel = 'powerlaw'
        self.dmSpecModel = 'powerlaw'
        self.dirExt = './chains_nanoAnalysis/'
        self.nwins = 1
        self.LMAX = 0
        self.noPhysPrior = False
        self.use_gpu = False
        self.fixSlope = False
        self.gwbPrior = 'uniform'
        self.redPrior = 'uniform'
        self.dmPrior = 'uniform'
        self.anis_modefile = None
        self.fullN = True
        self.fixRed = False
        self.fixDM = False
        self.psrStartIndex = 0
        self.psrEndIndex = 18
        self.psrIndices = None
        self.det_signal = False
        self.bwm_search = True
        self.bwm_antenna = 'quad'
        self.bwm_model_select = False
        self.cgw_search = False
        self.ecc_search = False
        self.epochTOAs = False
        self.psrTerm = False
        self.periEv = False
        self.incGWline = False
        self.gwlinePrior = 'uniform'


    def makeGUI(self):

        fromh5Check = widgets.Checkbox(description='HDF5 files')
        psrlistText = widgets.Text(description='Info file:',
                                   value='/home/NX01/PsrListings_GWB.txt',font_size=15)
        numpsrs = widgets.IntText(description='Number of pulsars:',
                               value=18,font_size=15,width='100%')
        preamble = widgets.VBox(children=[fromh5Check,psrlistText,numpsrs])

        def change_fromh5(name, value): self.from_h5 = value
        fromh5Check.on_trait_change(change_fromh5, 'value')

        def change_psrlist(name, value): self.psrlist = value
        psrlistText.on_trait_change(change_psrlist, 'value')

        def change_numpsrs(name, value): self.psrEndIndex = value
        numpsrs.on_trait_change(change_numpsrs, 'value')
        
        page0 = Box(children=[preamble], padding=4)
        page0.padding=20
        page0.font_size=20

        ############
        form = widgets.VBox()
        
        modes = widgets.IntText(description="Number of Fourier modes:", width='100%')

        def change_nmodes(name, value): self.nmodes = value
        modes.on_trait_change(change_nmodes, 'value')

        red = widgets.Checkbox(description="Red noise", value=False)
        red_prior = widgets.Dropdown(description="Prior",\
                                     options=['uniform', 'loguniform'],\
                                     font_size=20)
        red_specmodel = widgets.Dropdown(description="Spectral model",\
                                         options=['powerlaw', 'freespectral'],\
                                         font_size=20)
        red_info = widgets.HBox(visible=False, children=[red_prior,red_specmodel], padding=20)

        def change_red(name, value):
            if value: self.fixRed = False
            else: self.fixRed = True
        red.on_trait_change(change_red, 'value')

        def change_redprior(name, value): self.redPrior = value
        red_prior.on_trait_change(change_redprior, 'value')

        def change_redspecmodel(name, value): self.redSpecModel = value
        red_specmodel.on_trait_change(change_redspecmodel, 'value')
        
        dm = widgets.Checkbox(description="DM variations", value=False)
        dm_prior = widgets.Dropdown(description="Prior",\
                                    options=['uniform', 'loguniform'],\
                                    font_size=20)
        dm_specmodel = widgets.Dropdown(description="Spectral model",\
                                        options=['powerlaw', 'freespectral'],\
                                        font_size=20)
        dm_info = widgets.HBox(visible=False, children=[dm_prior,dm_specmodel], padding=20)

        def change_dm(name, value): self.dmVar = value
        dm.on_trait_change(change_dm, 'value')
        
        def change_dmprior(name, value): self.dmPrior = value
        dm_prior.on_trait_change(change_dmprior, 'value')

        def change_dmspecmodel(name, value): self.dmSpecModel = value
        dm_specmodel.on_trait_change(change_dmspecmodel, 'value')


        form.children = [modes,red,red_info,dm,dm_info]

        def on_red_toggle(name, value):
            if value: red_info.visible = True
            else: red_info.visible = False
        red.on_trait_change(on_red_toggle, 'value')

        def on_dm_toggle(name, value):
            if value: dm_info.visible = True
            else: dm_info.visible = False
        dm.on_trait_change(on_dm_toggle, 'value')

        page1 = Box(children=[form], padding=4)
        page1.padding=20
        page1.font_size=20
        ###########

        ###########
        form = widgets.VBox()
    
        gwb = Checkbox(description='Stochastic gravitational-wave background', value=False)
        def change_incGWB(name, value): self.incGWB = value
        gwb.on_trait_change(change_incGWB, 'value')

        gwb_prior = widgets.Dropdown(description="Prior",\
                                     options=['uniform', 'loguniform', 'sesana', 'mcwilliams'],\
                                     font_size=20)
        gwb_specmodel = widgets.Dropdown(description="Spectral model",\
                                         options=['powerlaw', 'freespectral', 'turnover'],\
                                         font_size=20)
        gwb_preamble = widgets.HBox(visible=True, \
                                    children=[gwb_prior,gwb_specmodel],padding=20)

        def change_gwbprior(name, value): self.gwbPrior = value
        gwb_prior.on_trait_change(change_gwbprior, 'value')

        def change_gwbspecmodel(name, value): self.gwbSpecModel = value
        gwb_specmodel.on_trait_change(change_gwbspecmodel, 'value')
                                    
        fixSlope = widgets.Checkbox(description='Fix PSD slope to -13/3', value=False)
        def change_fixslope(name, value): self.fixSlope = value
        fixSlope.on_trait_change(change_fixslope, 'value')
        
        incCorr = widgets.Checkbox(description='Include correlations', value=False)
        def change_inccorr(name, value): self.incCorr = value
        incCorr.on_trait_change(change_inccorr, 'value')

        corrOpts = widgets.Dropdown(visible=False, description='Type:', options=['Isotropic',
                                                                                'Spherical-harmonic anisotropy',
                                                                                'Direct cross-correlation recovery',
                                                                                'Stochastic point-source background'],
                                                                                font_size=20)
        def change_corrOpts(name, value):
            if corrOpts.value == 'Direct cross-correlation recovery':
                self.typeCorr == 'modelIndep'
            elif corrOpts.value == 'Stochastic point-source background':
                self.typeCorr == 'pointSrc'
            elif corrOpts.value == 'Isotropic':
                self.LMAX = 0
                self.typeCorr == 'spharmAnis'
            elif corrOpts.value == 'Spherical-harmonic anisotropy':
                self.typeCorr == 'spharmAnis'
        corrOpts.on_trait_change(change_corrOpts, 'value')
        
        anisLmax = widgets.IntText(visible=False, description='lmax:',width='100%',font_size=20)
        def change_anisLmax(name, value): self.LMAX = value
        anisLmax.on_trait_change(change_anisLmax, 'value')
        
        noPhysPrior = widgets.Checkbox(visible=False, description='Switch off physical prior')
        def change_nophysprior(name, value): self.noPhysPrior = value
        noPhysPrior.on_trait_change(change_nophysprior, 'value')
        
        anisOpts = widgets.HBox(children=[anisLmax,noPhysPrior])
        gwb_info = widgets.VBox(visible=False, children=[gwb_preamble, fixSlope, incCorr,
                                                        corrOpts, anisOpts])

        form.children = [gwb,gwb_info]

        def on_gwb_toggle(name, value):
            if value: gwb_info.visible = True
            else: gwb_info.visible = False

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
        def change_detsig(name, value): self.det_signal = value
        detsig.on_trait_change(change_detsig, 'value')

        detsig_info = widgets.Dropdown(visible=False, description="Signal type",\
                                        options=['burst with memory', 'circular binary', 'eccentric binary'],
                                        font_size=20)
        def change_detsiginfo(name, value):
            if detsig_info.value == 'circular binary':
                self.cgw_search = True
                self.ecc_search = False
                self.bwm_search = False
            elif detsig_info.value == 'eccentric binary':
                self.cgw_search = True
                self.ecc_search = True
                self.bwm_search = False
            elif detsig_info.value == 'burst with memory':
                self.cgw_search = False
                self.ecc_search = False
                self.bwm_search = True
        detsig_info.on_trait_change(change_detsiginfo, 'value')
        
        epochOpt = widgets.Checkbox(visible=False,description='Use epoch TOAs', value=False)
        def change_epochopt(name, value): self.epochTOAs = value
        epochOpt.on_trait_change(change_epochopt, 'value')

        psrtermOpt = widgets.Checkbox(description='Pulsar term', value=False)
        def change_psrtermopt(name, value): self.psrTerm = value
        psrtermOpt.on_trait_change(change_psrtermopt, 'value')

        perievOpt = widgets.Checkbox(description='Periapsis evolution', value=False)
        def change_perievopt(name, value): self.periEv = value
        perievOpt.on_trait_change(change_perievopt, 'value')
        
        binary_info = widgets.VBox(visible=False,children=[psrtermOpt,perievOpt])
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

        samplerOpt = widgets.Dropdown(description='Sampler',
                                   options=['PTMCMC','MultiNest'],font_size=20)
        hotchains = widgets.Checkbox(visible=True,description='Write hot chains')
        resume = widgets.Checkbox(visible=True,description='Resume')
        dirtext = widgets.Text(description='Parent results directory:',
                               value='./chains_nanoAnalysis/',font_size=20)
        
        bookkeeping = widgets.VBox(children=[samplerOpt,hotchains,resume,dirtext],width='150%')

        def change_sampler(name, value):
            if value == 'PTMCMC':
                self.sampler = 'ptmcmc'
            elif value == 'MultiNest':
                self.sampler = 'mnest'
        samplerOpt.on_trait_change(change_sampler, 'value')

        def change_hotchains(name, value): self.writeHotChains = value
        hotchains.on_trait_change(change_hotchains, 'value')

        def change_resume(name, value): self.resume = value
        resume.on_trait_change(change_resume, 'value')

        def change_dirtext(name, value): self.dirExt = value
        dirtext.on_trait_change(change_dirtext, 'value')

        def on_sampler_toggle(name, value):
            if value == 'PTMCMC':
                hotchains.visible = True
                resume.visible = True
            else:
                hotchains.visible = False
                resume.visible = False
        samplerOpt.on_trait_change(on_sampler_toggle, 'value')

        page4 = Box(children=[bookkeeping], padding=4)
        page4.padding=20
        page4.font_size=20

        ##########
        page0.border_color='#F08080'
        page1.border_color='#F08080'
        page2.border_color='#F08080'
        page3.border_color='#F08080'
        page4.border_color='#F08080'
        
        tabs = widgets.Tab(children=[page0, page1, page2, page3, page4])
        tabs.set_title(0, 'Pulsars')
        tabs.set_title(1, 'Pulsar properties')
        tabs.set_title(2, 'Stochastic GW signals')
        tabs.set_title(3, 'Deterministic GW signals')
        tabs.set_title(4, 'Sampling')

        tabs.font_weight='bolder'
        tabs.font_style='italic'
        tabs.padding=30
        tabs.font_size=20
        tabs.border_color='#F08080'
        tabs.border_width=3

        return tabs


    def makeJSON_button(self):

        json_click = widgets.Button(description='STORE MODEL',font_weight='bolder',
                                    color='#F08080',font_size=30,border_color='#F08080')
        
        def on_button_clicked(b):
            with open('mymodel.json', 'w') as fp:
                json.dump(self.__dict__, fp)
            fp.close()

        json_click.on_click(on_button_clicked)

        return json_click

    def makeRUN_button(self):

        run_click = widgets.Button(description='ENGAGE',font_weight='bolder',
                                    color='#F08080',font_size=30,border_color='#F08080',padding=30)
        
        def on_button_clicked(b):
            ipython.magic("run NX01_master.py --jsonModel=mymodel.json")

        run_click.on_click(on_button_clicked)

        return run_click
