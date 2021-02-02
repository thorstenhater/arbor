.. _pysimulation:

Simulations
===========

From recipe to simulation
-------------------------

To build a simulation the following concepts are needed:

    * an :class:`arbor.recipe` that describes the cells and connections in the model;
    * an :class:`arbor.context` used to execute the simulation.

The workflow to build a simulation is to first generate an
:class:`arbor.domain_decomposition` based on the :class:`arbor.recipe` and :class:`arbor.context` describing the distribution of the model
over the local and distributed hardware resources (see :ref:`pydomdec`). Then, the simulation is build using this :class:`arbor.domain_decomposition`.

.. container:: example-code

    .. code-block:: python

        import arbor

        # Get a communication context (with 4 threads, no GPU)
        context = arbor.context(threads=4, gpu_id=None)

        # Initialise a recipe of user defined type my_recipe with 100 cells.
        n_cells = 100
        recipe = my_recipe(n_cells)

        # Get a description of the partition of the model over the cores.
        decomp = arbor.partition_load_balance(recipe, context)

        # Instatitate the simulation.
        sim = arbor.simulation(recipe, decomp, context)

        # Run the simulation for 2000 ms with time stepping of 0.025 ms
        tSim = 2000
        dt = 0.025
        sim.run(tSim, dt)

.. currentmodule:: arbor

.. class:: simulation

    The executable form of a model.
    A simulation is constructed from a recipe, and then used to update and monitor the model state.

    Simulations take the following inputs:

    The **constructor** takes

        * an :class:`arbor.recipe` that describes the model;
        * an :class:`arbor.domain_decomposition` that describes how the cells in the model are assigned to hardware resources;
        * an :class:`arbor.context` which is used to execute the simulation.

    Simulations provide an interface for executing and interacting with the model:

        * Specify what data (spikes, probe results) to record.
        * **Advance the model state** by running the simulation up to some time point.
        * Retrieve recorded data.
        * Reset simulator state back to initial conditions.

    **Constructor:**

    .. function:: simulation(recipe, domain_decomposition, context)

        Initialize the model described by an :class:`arbor.recipe`, with cells and network distributed according to :class:`arbor.domain_decomposition`, and computational resources described by :class:`arbor.context`.

    **Updating Model State:**

    .. function:: reset()

        Reset the state of the simulation to its initial state.
        Clears recorded spikes and sample data.

    .. function:: run(tfinal, dt)

        Run the simulation from current simulation time to ``tfinal``,
        with maximum time step size ``dt``.

        :param tfinal: The final simulation time [ms].

        :param dt: The time step size [ms].

    .. function:: set_binning_policy(policy, bin_interval)

        Set the binning ``policy`` for event delivery, and the binning time interval ``bin_interval`` if applicable [ms].

        :param policy: The binning policy of type :class:`binning`.

        :param bin_interval: The binning time interval [ms].

    **Recording spike data:**

    .. function:: record(policy)

        Disable or enable recorder of rank-local or global spikes, as determined by the ``policy``.

        :param policy: Recording policy of type :class:`spike_recording`.

    .. function:: spikes()

        Return a NumPy structured array of spikes recorded during the course of a simulation.
        Each spike is represented as a NumPy structured datatype with signature
        ``('source', [('gid', '<u4'), ('index', '<u4')]), ('time', '<f8')``.

    **Sampling probes:**

    .. function:: sample(probe_id, schedule, policy)

        Set up a sampling schedule for the probes associated with the supplied probe_id of type :class:`cell_member`.
        The schedule is any schedule object, as might be used with an event generator — see :ref:`pyrecipe` for details.
        The policy is of type :class:`sampling_policy`. It can be omitted, in which case the sampling will accord with the
        ``sampling_policy.lax`` policy.

        The method returns a handle which can be used in turn to retrieve the sampled data from the simulator or to
        remove the corresponding sampling process.

    .. function:: probe_metadata(probe_id)

        Retrieve probe metadata for the probes associated with the given probe_id of type :class:`cell_member`.
        The result will be a list, with one entry per probe; the specifics of each metadata entry will depend upon
        the kind of probe in question.

    .. function:: remove_sampler(handle)

        Disable the sampling process referenced by the argument ``handle`` and remove any associated recorded data.

    .. function:: remove_all_samplers()

        Disable all sampling processes and remove any associated recorded data.

    .. function:: samples(handle)

        Retrieve a list of sample data associated with the given ``handle``.
        There will be one entry in the list per probe associated with the probe id used when the sampling was set up.
        Each entry is a pair ``(samples, meta)`` where ``meta`` is the probe metadata as would be returned by
        ``probe_metadata(probe_id)``, and ``samples`` contains the recorded values.

        The format of the recorded values will depend upon the specifics of the probe, though generally it will
        be a NumPy array, with the first column corresponding to sample time and subsequent columns holding
        the value or values that were sampled from that probe at that time.

    **Types:**

    .. class:: binning

        Enumeration for event time binning policy.

        .. attribute:: none

            No binning policy.

        .. attribute:: regular

            Round time down to multiple of binning interval.

        .. attribute:: following

            Round times down to previous event if within binning interval.

    .. class:: spike_recording

        Enumeration for spike recording policy.

        .. attribute:: off

            Disable spike recording.

        .. attribute:: local

            Record all generated spikes from cells on this MPI rank.

        .. attribute:: all

            Record all generated spikes from cells on all MPI ranks.

    .. class:: sampling_policy

        Enumeration for deteriming sampling policy.

        .. attribute:: lax

            Sampling times may not be exactly as requested in the sampling schedule, but
            the process of sampling is guaranteed not to disturb the simulation progress or results.

        .. attribute:: exact

            Interrupt the progress of the simulation as required to retrieve probe samples at exactly
            those times requested by the sampling schedule.

Recording spikes
----------------

By default, spikes are not recorded. Recording is enabled with the
:py:func:`simulation.record` method, which takes a single argument instructing
the simulation object to record no spikes, all locally generated spikes, or all
spikes generated by any MPI rank.

Spikes recorded during a simulation are returned as a NumPy structured datatype with two fields,
``source`` and ``time``. The ``source`` field itself is a structured datatype with two fields,
``gid`` and ``index``, identifying the spike detector that generated the spike.


.. container:: example-code

    .. code-block:: python

        import arbor

        # Instatitate the simulation.
        sim = arbor.simulation(recipe, decomp, context)

        # Direct the simulation to record all spikes.
        sim.record(arbor.spike_recording.all)

        # Run the simulation for 2000 ms with time stepping of 0.025 ms
        tSim = 2000
        dt = 0.025
        sim.run(tSim, dt)

        # Print the spikes and according spike time
        for s in sim.spikes():
            print(s)

>>> ((0,0), 2.15168)
>>> ((1,0), 14.5235)
>>> ((2,0), 26.9051)
>>> ((3,0), 39.4083)
>>> ((4,0), 51.9081)
>>> ((5,0), 64.2902)
>>> ((6,0), 76.7706)
>>> ((7,0), 89.1529)
>>> ((8,0), 101.641)
>>> ((9,0), 114.125)

Recording samples
-----------------

Definitions
***********

probe
    A measurement that can be perfomed on a cell. Each cell kind will have its own sorts of probe;
    Cable cells (:attr:`arbor.cable_probe`) allow the monitoring of membrane voltage, total membrane
    current, mechanism state, and a number of other quantities, measured either over the whole cell,
    or at specific sites (see :ref:`pycablecell-probes`).

    Probes are described by probe addresses, and the collection of probe addresses for a given cell is
    provided by the :class:`recipe` object. One address may correspond to more than one probe:
    as an example, a request for membrane voltage on a cable cell at sites specified by a location
    expression will generate one probe for each site in that location expression.

probe id
    A designator for one or more probes as specified by a recipe. The *probe id* is a
    :class:`cell_member` refering to a specific cell by gid, and the index into the list of
    probe addresses returned by the recipe for that gid.

metadata
    Each probe has associated metadata describing, for example, the location on a cell where the
    measurement is being taken, or other such identifying information. Metadata for the probes
    associated with a *probe id* can be retrieved from the simulation object, and is also provided
    along with any recorded samples.

sample
    A record of data corresponding to the value at a specific *probe* at a specific time.

schedule
    An object representing a series of monotonically increasing points in time, used for determining
    sample times (see :ref:`pyrecipe`).

Procedure
*********

    There are three parts to the process of recording cell data over a simulation.

    1. Describing what to measure.

       The recipe object must provide a method :py:func:`recipe.get_probes` that returns a list of
       probe addresses for the cell with a given ``gid``. The kth element of the list corresponds
       to the *probe id* ``(gid, k)``.

       Each probe address is an opaue object describing what to measure and where, and each cell kind
       will have its own set of functions for generating valid address specifications. Possible cable
       cell probes are described in the cable cell documentation: :ref:`pycablecell-probes`.

    2. Instructing the simulator to record data.

       Recording is set up with the method :py:func:`simulation.sample`
       as described above. It returns a handle that is used to retrieve the recorded data after
       simulation.

    3. Retrieve recorded data.

       The method :py:func:`simulation.samples` takes a handle and returns the recorded data as a list,
       with one entry for each probe associated with the *probe id* that was used in step 2 above. Each
       entry will be a tuple ``(data, meta)`` where ``meta`` is the metadata associated with the
       probe, and ``data`` contains all the data sampled on that probe over the course of the
       simulation.

       The contents of ``data`` will depend upon the specifics of the probe, but note:

       i. The object type and structure of ``data`` is fully determined by the metadata.

       ii. All currently implemented probes return data that is a NumPy array, with one
           row per sample, first column being sample time, and the remaining columns containing
           the corresponding data.

.. container:: example-code

    .. code-block:: python

        import arbor

        # [... define recipe, decomposition, context ... ]
        # Initialize simulation:

        sim = arbor.simulation(recipe, decomp, context)

        # Sample probe id (0, 0) (first probe id on cell 0) every 0.1 ms with exact sample timing:

        handle = sim.sample((0, 0), arbor.regular_schedule(0.1), arbor.sampling_policy.exact)

        # Run simulation and retrieve sample data from the first probe associated with the handle.

        sim.run(tfinal=3, dt=0.1)
        data, meta = sim.samples(handle)[0]
        print(data)

>>> [[  0.         -50.        ]
>>>  [  0.1        -55.14412111]
>>>  [  0.2        -59.17057625]
>>>  [  0.3        -62.58417912]
>>>  [  0.4        -65.47040168]
>>>  [  0.5        -67.80222861]
>>>  [  0.6        -15.18191623]
>>>  [  0.7         27.21110919]
>>>  [  0.8         48.74665099]
>>>  [  0.9         48.3515727 ]
>>>  [  1.          41.08435987]
>>>  [  1.1         33.53571111]
>>>  [  1.2         26.55165892]
>>>  [  1.3         20.16421752]
>>>  [  1.4         14.37227532]
>>>  [  1.5          9.16209063]
>>>  [  1.6          4.50159342]
>>>  [  1.7          0.34809083]
>>>  [  1.8         -3.3436289 ]
>>>  [  1.9         -6.61665687]
>>>  [  2.          -9.51020525]
>>>  [  2.1        -12.05947812]
>>>  [  2.2        -14.29623969]
>>>  [  2.3        -16.24953688]
>>>  [  2.4        -17.94631322]
>>>  [  2.5        -19.41182385]
>>>  [  2.6        -52.19519009]
>>>  [  2.7        -62.53349949]
>>>  [  2.8        -69.22068995]
>>>  [  2.9        -73.41691825]]

