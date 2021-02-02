import arbor
import pandas
import seaborn
import sys
from arbor import mechanism as mech

#(1) Read the morphology from an SWC file.

# Read the SWC filename from input
# Example from docs: morph.swc

if len(sys.argv) < 2:
    print("No SWC file passed to the program")
    sys.exit(0)

filename = sys.argv[1]
morph = arbor.load_swc_arbor(filename)

#(2) Create and populate the label dictionary.

labels = arbor.label_dict()

# Regions:

labels['soma'] = '(tag 1)'
labels['axon'] = '(tag 2)'
labels['dend'] = '(tag 3)'
labels['last'] = '(tag 4)'

labels['all'] = '(all)'

labels['gt_1.5'] = '(radius-ge (region "all") 1.5)'
labels['custom'] = '(join (region "last") (region "gt_1.5"))'

# Locsets:

labels['root']     = '(root)'
labels['terminal'] = '(terminal)'
labels['custom_terminal'] = '(restrict (locset "terminal") (region "custom"))'
labels['axon_terminal'] = '(restrict (locset "terminal") (region "axon"))'

# (3) Create and populate the decor.

decor = arbor.decor()

# Set the default properties of the cell (this overrides the model defaults).

decor.set_property(Vm =-55)

# Override the cell defaults.

decor.paint('"custom"', tempK=270)
decor.paint('"soma"',   Vm=-50)

# Paint density mechanisms.

decor.paint('"all"', 'pas')
decor.paint('"custom"', 'hh')
decor.paint('"dend"',  mech('Ih', {'gbar': 0.001}))

# Place stimuli and spike detectors.

decor.place('"root"', arbor.iclamp(10, 1, current=2))
decor.place('"root"', arbor.iclamp(30, 1, current=2))
decor.place('"root"', arbor.iclamp(50, 1, current=2))
decor.place('"axon_terminal"', arbor.spike_detector(-10))

# Set cv_policy

soma_policy = arbor.cv_policy_single('"soma"')
dflt_policy = arbor.cv_policy_max_extent(1.0)
policy = dflt_policy | soma_policy
decor.discretization(policy)

# (4) Create the cell.

cell = arbor.cable_cell(morph, labels, decor)

# (5) Construct the model

model = arbor.single_cell_model(cell)

# (6) Set the model default properties

model.properties.set_property(Vm =-65, tempK=300, rL=35.4, cm=0.01)
model.properties.set_ion('na', int_con=10,   ext_con=140, rev_pot=50, method='nernst/na')
model.properties.set_ion('k',  int_con=54.4, ext_con=2.5, rev_pot=-77)

# Extend the default catalogue with the allen catalogue.

model.catalogue.extend(arbor.allen_catalogue(), "")

# (7) Add probes.

model.probe('voltage', where='"custom_terminal"',  frequency=50000)

# (8) Run the simulation.

model.run(tfinal=100, dt=0.025)

# (9) Print the spikes.

print(len(model.spikes), 'spikes recorded:')

# Print the spike times.

for s in model.spikes:
    print(s)

# (10) Plot the voltages

df = pandas.DataFrame()
for t in model.traces:
    df=df.append(pandas.DataFrame({'t/ms': t.time, 'U/mV': t.value, 'Location': str(t.location), 'Variable': t.variable}))

seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV",hue="Location",col="Variable",ci=None).savefig('single_cell_detailed_result.svg')
