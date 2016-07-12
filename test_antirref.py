#!/usr/bin/env python
#-*- coding: utf-8 -*-
""" 
Adapted from test2.py to include transmisison/reflection spectra
using the function get_s_parameters
"""

import time, sys, os
import numpy as np
from scipy.constants import c, epsilon_0, mu_0

import meep_utils, meep_materials
from meep_utils import in_sphere, in_xcyl, in_ycyl, in_zcyl, in_xslab, in_yslab, in_zslab, in_ellipsoid
import meep_mpi as meep
#import meep

class Antirref_model(meep_utils.AbstractMeepModel):
    def __init__(self, simtime=0.5e-13, resolution = 8e-9,size_x = 0.3e-6, size_y = 0.3e-6,
                 size_z = .75e-6,thickness_mgf2 = 50.8e-9,**other_args):
    #def __init__(self, simtime=1e-13, resolution = 5e-9,size_x = 0.2e-6, size_y = 0.2e-6,
     #            size_z = 0.5e-6,thickness_mgf2 = 99.8e-9,**other_args):
	meep_utils.AbstractMeepModel.__init__(self) #Inizialitation of the class
	

	self.simulation_name = "AntirrefFilm30deg"    
        self.src_freq = 545e12     # [Hz] (note: srcwidth irrelevant for continuous_source)
	self.src_width= 130e12
	self.interesting_frequencies=(430e12,650e12)
        self.pml_thickness = 5.0e-8
        self.size_x = size_x 
        self.size_y = size_y
        self.size_z = size_z
        self.simtime = simtime      # [s]
	self.monitor_z1 = - 3e-7
	self.monitor_z2 =  thickness_mgf2/2 + 2.5e-7
        self.Kx = (2*np.pi*self.src_freq/c)*np.sin(np.pi/6)#6.8017e6 # 30 degrees
	self.Ky = 0
	self.padding=0
        self.register_locals(locals(), other_args)          ## Remember the parameters
        ## Define materials
        f_c = c / np.pi/self.resolution/meep_utils.meep.use_Courant()

        self.materials   = [meep_materials.material_dielectric(eps=1.904,where=self.where_MgF2)] #eps=1.904
	self.materials +=  [meep_materials.material_dielectric(eps=3.24, where = self.where_sust)] 
	#self.materials = [] # uncomment if no materials present
        for material in self.materials: self.fix_material_stability(material, f_c=2e15, verbose=1)
        meep_utils.plot_eps(self.materials, plot_conductivity=True, 
                draw_instability_area=(self.f_c(), 3*meep.use_Courant()**2), mark_freq={self.f_c():'$f_c$'})
        self.test_materials()

    def where_MgF2(self, r):
        if in_zslab(r,cz=0,d=self.thickness_mgf2):
            return self.return_value   #remove comment if film present          # (do not change this line)
        return 0
    def where_sust(self,r):
	if (r.z()>=self.thickness_mgf2/2):
	    return self.return_value
        return 0
   
 # Model selection
model_param = meep_utils.process_param(sys.argv[1:])
model = Antirref_model(**model_param)

## Initialize volume, structure and the fields according to the model
vol = meep.vol3d(model.size_x, model.size_y, model.size_z, 1./model.resolution)
vol.center_origin()
s = meep_utils.init_structure(model=model, volume=vol, pml_axes=meep.Z)

## Create fields with Bloch-periodic boundaries 
f = meep.fields(s)
# Define the Bloch-periodic boundaries (any transversal component of k-vector is allowed)
f.use_bloch(meep.X, getattr(model, 'Kx', 0) / (-2*np.pi)) 
f.use_bloch(meep.Y, getattr(model, 'Ky', 0) / (-2*np.pi))

# Add the field source (see meep_utils for an example of how an arbitrary source waveform is defined)
src_time_type = meep.gaussian_src_time(model.src_freq/c, model.src_width/c)
#src_time_type = meep.continuous_src_time(model.src_freq/c)
srcvolume = meep.volume(
        meep.vec(-model.size_x/2, -model.size_y/2, -model.size_z/2+model.pml_thickness),
        meep.vec(+model.size_x/2, +model.size_y/2, -model.size_z/2+model.pml_thickness))

#f.add_volume_source(meep.Ex, src_time_type, srcvolume)

class SrcAmplitudeFactor(meep.Callback): 
    ## The source amplitude is complex -> phase factor modifies its direction
    ## todo: implement in MEEP: we should define an AmplitudeVolume() object and reuse it for monitors later
    def __init__(self, Kx=0, Ky=0): 
        meep.Callback.__init__(self)
        (self.Kx, self.Ky) = Kx, Ky
    def complex_vec(self, vec):   ## Note: the 'vec' coordinates are _relative_ to the source center
        # (oblique) plane wave source:
        return np.exp(-1.0j*(self.Kx*vec.x() + self.Ky*vec.y()))
        # (oblique) Gaussian beam source:
        #return np.exp(-1j*(self.Kx*vec.x() + self.Ky*vec.y()) - (vec.x()/100e-6)**2 - (vec.y()/100e-6)**2) 
af = SrcAmplitudeFactor(Kx=model.Kx, Ky=model.Ky) 
meep.set_AMPL_Callback(af.__disown__())
f.add_volume_source(meep.Ex, src_time_type, srcvolume, meep.AMPL)

## Define visualisation output
## Define monitors planes and visualisation output
#monitor_options = {'size_x':model.size_x, 'size_y':model.size_y, 'Kx':model.Kx, 'Ky':0}
monitor1_Ex = meep_utils.AmplitudeMonitorPlane(f,comp=meep.Ex, size_x = model.size_x, size_y = model.size_y,resolution=20e-8,z_position=model.monitor_z1, Kx=model.Kx, Ky = model.Ky)
monitor1_Hy = meep_utils.AmplitudeMonitorPlane(f,comp=meep.Hy,  size_x = model.size_x, size_y = model.size_y,resolution=20e-8,z_position=model.monitor_z1, Kx=model.Kx, Ky = model.Ky)
monitor2_Ex = meep_utils.AmplitudeMonitorPlane(f,comp=meep.Ex,  size_x = model.size_x, size_y = model.size_y,resolution=20e-8,z_position=model.monitor_z2, Kx=model.Kx, Ky = model.Ky)
monitor2_Hy = meep_utils.AmplitudeMonitorPlane(f,comp=meep.Hy,  size_x = model.size_x, size_y = model.size_y,resolution=20e-8,z_position=model.monitor_z2, Kx=model.Kx, Ky = model.Ky)

slices =  [meep_utils.Slice(model=model, field=f, components=(meep.Dielectric), at_t=0, outputhdf=True,name='EPS')]
slices += [meep_utils.Slice(model=model, field=f, components=meep.Ex, at_y=0, at_x=[-0.35e-6,0.35e-6], min_timestep=0.5e-15, outputhdf=True,outputgif=True, name='ParallelCut')]
slices += [meep_utils.Slice(model=model, field=f, components=meep.Ex, at_z=model.monitor_z2, min_timestep=.05e-14, outputhdf=True, outputgif=True, name='PerpendicularCut')]
slices += [meep_utils.Slice(model=model, field=f, components=meep.Ex, at_z=model.monitor_z2, at_y=0, outputhdf=True, name='Line')]


f.step(); timer = meep_utils.Timer(simtime=model.simtime); meep.quiet(True) # use custom progress messages
controlsample = 0
while (f.time()/c < model.simtime):     # timestepping cycle
    f.step()
    timer.print_progress(f.time()/c)
    for monitor in (monitor1_Ex, monitor1_Hy, monitor2_Ex, monitor2_Hy): monitor.record(field=f)
    filesample = open('./'+model.simulation_name + '/Exsampleafter.dat','a')
    filesample.write(str(f.time()/c)+' ')
    filesample.write(str(f.get_field(meep.Ex, meep.vec(0, 0,model.monitor_z2)))+'\n')
    filesample.close()
    filesample = open('./'+model.simulation_name + '/Exsamplebefore.dat','a')
    filesample.write(str(f.time()/c)+' ')
    filesample.write(str(f.get_field(meep.Ex, meep.vec(0, 0,model.monitor_z1)))+'\n')
    filesample.close()

    for slice_ in slices: slice_.poll(f.time()/c)
for slice_ in slices: slice_.finalize()
meep_utils.notify(model.simulation_name, run_time=timer.get_time())

## Get the reflection and transmission of the structure
if meep.my_rank() == 0:
    freq, s11, s12, headerstring = meep_utils.get_s_parameters(monitor1_Ex, monitor1_Hy, monitor2_Ex, monitor2_Hy,             
            intf=getattr(model, 'interesting_frequencies', [0, model.src_freq+model.src_width]),
            pad_zeros=1.0, Kx=model_param.get('Kx', 0), Ky=model_param.get('Ky', 0),eps1=1.0,eps2=3.24)

    print "Saving the fields as a reference"
    np.savetxt(fname='./'+model.simulation_name+'/sparam.dat', fmt="%.6e",
                X=zip(freq, np.abs(s11), np.angle(s11), np.abs(s12), np.angle(s12)))
    with open("./last_simulation_name.dat", "w") as outfile: outfile.write(model.simulation_name) 

meep.all_wait()         # Wait until all file operations are finished
