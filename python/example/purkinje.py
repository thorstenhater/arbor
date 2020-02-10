import arbor                                                                                                                                                            
from arbor import mechanism as mech
from arbor import location as loc 
import matplotlib.pyplot as plt   

opt_vinit = -65
opt_temp = 35
tsim = 200
plot_to_file=False

# load morphology from swc file
tree = arbor.load_swc('../../test/unit/swc/purkinje_mouse_original.swc')

# append a custom axon
p = tree.append(0, arbor.msample(0, 0, 0,   0.97/2, 2))
p = tree.append(p, arbor.msample(0, 0, 17,  0.97/2, 2))
p = tree.append(p, arbor.msample(0, 0, 21,  0.97/2, 2))
p = tree.append(p, arbor.msample(0, 0, 121, 0.73/2, 2))
p = tree.append(p, arbor.msample(0, 0, 125, 0.73/2, 2))
p = tree.append(p, arbor.msample(0, 0, 225, 0.73/2, 2)) 
p = tree.append(p, arbor.msample(0, 0, 229, 0.73/2, 2))
p = tree.append(p, arbor.msample(0, 0, 329, 0.73/2, 2))
p = tree.append(p, arbor.msample(0, 0, 333, 0.73/2, 2))
p = tree.append(p, arbor.msample(0, 0, 433, 0.73/2, 2))

morph = arbor.morphology(tree, spherical_root=True)

# define named regions and locations
labels = { 'all': '(all)',
           'stim_site': '(location 0 0.5)',
           'soma':       '(tag 1)',
           'axon':       '(tag 2)',
           'dend':       '(tag 3)',
           'dendtrunk':  '(intersect (radius_ge (region "dend")  0.7)   (radius_lt (region "dend") 1.025))',
           'dendsodium': '(intersect (radius_ge (region "dend")  1.025) (radius_ge (region "dend") 3))',
           'AIS':        '(distal_interval (proximal (region "axon"))  17)',
           'AISK':       '(distal_interval (distal (region "AIS"))      4)',
           'myelin0':    '(distal_interval (distal (region "AISK"))   100)',
           'rn0':        '(distal_interval (distal (region "myelin0"))  4)',
           'myelin1':    '(distal_interval (distal (region "rn0"))    100)',
           'rn1':        '(distal_interval (distal (region "myelin1"))  4)',
           'myelin2':    '(distal_interval (distal (region "rn1"))    100)',
           'rn2':        '(distal_interval (distal (region "myelin2"))  4)',
           'myelin3':    '(distal_interval (distal (region "rn2"))    100)',
           'myelin':     '(join (region "myelin0") (region "myelin1") (region "myelin2") (region "myelin3"))',
           'rn':         '(join (region "rn0") (region "rn1") (region "rn2"))'}

ldict = arbor.label_dict(labels)

# construct cell
cell = arbor.cable_cell(morph, ldict)

# set cell properties
cell.set_properties(Vm=opt_vinit, tempK=opt_temp+273.15)
cell.set_ion( 'k', int_con=54.4, ext_con=2.5, method=mech('nernst/k'))
cell.set_ion('na', int_con=10,   ext_con=140, method=mech('nernst/na'))
cell.set_ion('ca', int_con=5e-5, ext_con=2)

cell.paint('soma', mech('Leak',   {"e": -61, "gmax": 0.001}))
cell.paint('soma', mech('Nav1_6', {"gbar": 0.18596749324385001}))
cell.paint('soma', mech('Kv1_1',  {"gbar": 0.0029172006269699998}))
cell.paint('soma', mech('Kv3_4',  {"gkbar": 0.069972751903779995}))
cell.paint('soma', mech('Kir2_3', {"gkbar": 2.322613156e-05}))
cell.paint('soma', mech('Kca1_1', {"gbar": 0.01197387128516}))
cell.paint('soma', mech('Kca2_2', {"gkbar": 0.0013377920303699999}))
cell.paint('soma', mech('Kca3_1', {"gkbar": 0.01388910824701}))
cell.paint('soma', mech('Cav2_1', {"pcabar": 0.00020306777733000001}))
cell.paint('soma', mech('Cav3_1', {"pcabar": 5.1352684600000001e-06}))
cell.paint('soma', mech('Cav3_2', {"gcabar": 0.00070742370991999995}))
cell.paint('soma', mech('Cav3_3', {"pcabar": 0.00034648446559000002}))
cell.paint('soma', mech('HCN1',   {"gbar": 0.0016391306742300001}))
cell.paint('soma', mech('cdp5',   {"TotalPump": 2e-08}))
cell.paint('soma', rL=122, cm=0.0077000000000000002)

cell.paint('dend', mech('Leak',   {"e": -61, "gmax": 0.00029999999999999997}))
cell.paint('dend', mech('Kv1_1',  {"gbar": 0.0012215542202700001}))
cell.paint('dend', mech('Kv1_5',  {"gKur": 0.00011449636712999999}))
cell.paint('dend', mech('Kv3_3',  {"gbar": 0.01054618632087}))
cell.paint('dend', mech('Kv4_3',  {"gkbar": 0.00147529033238}))
cell.paint('dend', mech('Kca1_1', {"gbar": 0.036614181526090001}))
cell.paint('dend', mech('Cav2_1', {"pcabar": 0.0030012117233499998}))                                                                                                   
cell.paint('dend', mech('Cav3_3', {"pcabar": 0.0001470343953}))
cell.paint('dend', mech('HCN1',   {"gbar": 4.9150980800000004e-06}))
cell.paint('dend', mech('cdp5',   {"TotalPump": 4.9999999999999998e-08}))
cell.paint("dend", rL=122, cm=0.1)

cell.paint('dendtrunk', mech('Kir2_3', {"gkbar": 1.8891651439999999e-05}))
cell.paint('dendtrunk', mech('Cav3_1', {"pcabar": 5.9360810899999996e-06}))
cell.paint('dendtrunk', mech('Cav3_2', {"gcabar": 0.0013660737419300001}))
cell.paint('dendtrunk', mech('Kca2_2', {"gkbar": 0.00075069914383999999}))
cell.paint('dendtrunk', mech('Kca3_1', {"gkbar": 0.0033585524343299998}))

cell.paint('dendsodium', mech('Nav1_6', {"gbar": 0.01539837081695}))

cell.paint('AIS', mech('Leak',   {"e": -61, "gmax": 0.00029999999999999997}))
cell.paint('AIS', mech('Nav1_6', {"gbar": 0.51928565670617}))
cell.paint('AIS', mech('Kv3_4',  {"gkbar": 0.015944699248819999}))
cell.paint('AIS', mech('Cav2_1', {"pcabar": 0.00030802692871}))
cell.paint('AIS', mech('Cav3_1', {"pcabar": 8.8181145799999994e-06}))
cell.paint('AIS', mech('cdp5',   {"TotalPump": 2e-08}))
cell.paint('AIS', rL=122, cm=0.0077000000000000002)

cell.paint('AISK', mech('Leak',  {"e": -61, "gmax": 0.00029999999999999997}))
cell.paint('AISK', mech('Kv1_1', {"gbar": 0.01058797648119}))
cell.paint('AISK', rL=122, cm=0.0077000000000000002)

cell.paint('myelin', mech('pas', {"e": -63, "g": 4.9999999999999998e-08}))
cell.paint('myelin', rL=122, cm=1.8700000000000001e-13)

cell.paint('rn', mech('Leak',   {"e": -61, "gmax": 0.00029999999999999997}))
cell.paint('rn', mech('Nav1_6', {"gbar": 0.027565014930900002}))
cell.paint('rn', mech('Kv3_4',  {"gkbar": 0.029949599085}))
cell.paint('rn', mech('Cav2_1', {"pcabar": 0.00026695621961}))
cell.paint('rn', mech('Cav3_1', {"pcabar": 1.487097008e-05}))
cell.paint('rn', mech('cdp5',   {"TotalPump": 4.9999999999999998e-07}))
cell.paint('rn', rL=122, cm=0.0077000000000000002)

# attach the things
cell.place('stim_site', arbor.iclamp(tstart=0, duration=tsim, current=0.2))

cell.compartments_on_samples()

# make the model
model = arbor.single_cell_model(cell)

model.add_ion( 'h', valence=1, int_con=1.0, ext_con=1.0, rev_pot=-34.4);
model.add_ion('no', valence=1, int_con=1.0, ext_con=1.0, rev_pot=0);

model.probe('voltage', where=loc(0,0), frequency=50000)
model.probe('voltage', where=loc(758,0), frequency=50000)
model.probe('voltage', where=loc(758,0.5), frequency=50000)
model.probe('voltage', where=loc(758,1), frequency=50000)

model.run(tsim)


# Plot the recorded voltages over time.
fig, ax = plt.subplots()
for t in model.traces:
    ax.plot(t.time, t.value)

legend_labels = ['{}: {}'.format(s.variable, s.location) for s in model.traces]
ax.legend(legend_labels)
ax.set(xlabel='time (ms)', ylabel='voltage (mV)', title='Purkinje cell demo')
plt.xlim(0, tsim)
#plt.ylim(-80,50)
ax.grid()

if plot_to_file:
    fig.savefig("voltages.png", dpi=300)
else:
    plt.show()             
