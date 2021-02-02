.. _morph:

Cable cell morphology
=====================

In morphologically detailed simulations of neurons,
`the cell is modelled <https://en.wikipedia.org/wiki/Cable_theory>`_
as a set of round electrical cables, which have, among others properties,
capacitances and resistances. These can in turn be expressed as a system of
partial differential equations which can be solved to compute a current or
voltage. The geometry of the biological cell is approximated as a tree of
cable segments, which is a :term:`morphology` in Arbor nomenclature.

Let's first define the building blocks out of which we can construct a
proper definition for a morphology.

.. note::
  In certain cases, a second term appears in the
  definitions, where the second term corresponds to the associated type in Python and C++.
  The ``m`` prefix, short for *morphology*, avoids overly generic type names.

.. glossary::

  point
  mpoint
    A point in 3D space has three coordinates. In Arbor, we add a fourth coordinate:
    radius. The mpoint thus represents the centre of a cable and the radius represents
    the cross-sectional radius of the cable.

.. csv-table:: The properties of a :term:`point <mpoint>`, in the context of cable cell morphologies.
   :widths: 10, 10, 30
   :align: center

   **Field**,   **Type**, **Description**
   ``x``,       real, x coordinate of centre of cable (μm).
   ``y``,       real, y coordinate of centre of cable (μm).
   ``z``,       real, z coordinate of centre of cable (μm).
   ``radius``,  real, cross sectional radius of cable (μm).

.. glossary::

  segment
  msegment
    A segment is a frustum (cylinder or truncated cone), with the centre and radius at each
    end defined by a pair of :term:`points <mpoint>`. In other words, in Arbor the radius between two points is interpolated
    linearly, resulting in either a cylinder (equal radii) or truncated cone (differing radii),
    centred at the line through the pair of points.

.. csv-table:: The properties of a :term:`segment`.
   :widths: 10, 10, 30
   :align: center

   **Field**,      **Type**,                           **Description**
   ``prox``,       :term:`mpoint`,   the centre and radius of the proximal end.
   ``dist``,       :term:`mpoint`,   the centre and radius of the distal end.
   ``tag``,        integer,              ":term:`tag` meta-data, can be used to classify segments of the same kind (ex: soma, dendrite, but also arbitrary use-defined groups"

.. figure:: ../gen-images/term_segments.svg
  :width: 300
  :align: center

  Four segments arranged head to toe.

.. glossary::

  branch
    A branch is the longest possible unbranched sequence of :term:`segments <segment>`.

.. figure:: ../gen-images/term_branch.svg
  :width: 300
  :align: center

  A branch corresponding to the previous segments.

.. glossary::

  location
  mlocation
    A location is not a point in 3D space, but a point in the cable cell morphology's
    coordinate system. It is defined by a specific branch and a position along the length of the branch.

.. csv-table:: The properties of :term:`mlocation`.
   :widths: 10, 10, 30
   :align: center

   **Field**,      **Type**,        **Description**
   ``branch``,     integer,         id of the branch
   ``pos``,        real (0≤pos≤1),  position along the length of the branch

.. glossary::

  cable
  mcable
    A cable is a subset of a :term:`branch`, and is thus defined as between two :term:`locations <mlocation>` on a particular branch.

.. figure:: ../gen-images/term_cable.svg
  :width: 300
  :align: center

  A cable shown in black is defined as a section :term:`located <mlocation>` between 0.2 and 0.8 of the length of the previous branch.

.. glossary::

  tag
    A tag is an integer label on every segment, which can be used to define disjoint regions on cells.
    The meaning of tag values are not fixed in Arbor, however we typically use tag values that correspond
    to SWC `structure identifiers <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_.

With these definitions, we can create proper definitions for :term:`morphology` and :term:`segment tree`.

.. note::

  NEURON uses different nomenclature for segments and branches. The segments (alternatively referred to as compartments) that
  NEURON uses for control over discretisation (by assigning ``nseg`` segments per section)
  most closely correspond to :term:`control volumes <control volume>` in Arbor. Arbor
  uses truncated cones to represent branches and segments, Neuron uses a weighted
  average radius to create cylinders (see "trapezoidal integration" in the
  `Neuron documentation <https://www.neuron.yale.edu/neuron/static/py_doc/modelspec/programmatic/topology/geometry.html>`_.).

.. _morph-segment_tree:

Segment trees
--------------

.. glossary::

  segment tree
  segment_tree
    A segment tree describes a morphology as a set of :term:`segments <segment>` and their connections,
    designed to support both the diverse descriptions of cell morphologies (e.g. SWC, NeuroLicida, NeuroML),
    and tools that iteratively construct cell morphologies (e.g. L-system generators, interactive cell-builders).

Segment trees comprise a sequence of segments starting from
at lease one :term:`root` segment, together with a parent-child adjacency relationship
where a child segment is distal to its parent. Branches in the tree occur where a segment
has more than one child. Furthermore, a segment can not have more than one parent.
In this manner, neuron morphologies are modelled as a tree, where cables that
represent dendrites and axons can branch, but branches can not rejoin.

.. _morph-segment-definitions:

The following definitions are used to refer to segments in a segment tree:

.. glossary::

  root
    Segments at the root or start of the tree. A non-empty tree must have at least one root segment,
    and the first segment will always be a root.

  parent
    Each segment has one parent, except for root segments which have :data:`mnpos <arbor.mnpos>` as their parent.

    * The id of a segment is always greater than the id of its parent.
    * The ids of segments on the same unbranched sequence of segments do not need to be contiguous.

  child
    A segment's children are the segments that have the segment as their parent.

  terminal
    A segment with no children. Terminals lie at the end of dendritic trees or axons.

  fork
    A segment with more than one child. The distal end of a fork segment are *fork points*,
    where a cable splits into two or more branches.

    * Arbor allows more than two branches at a fork point.

The following segment tree models a soma as a cylinder, a branching dendritic tree and
an axon with an axonal hillock. The segments are coloured according to their tag, which
in this case are SWC structure identifiers: tag 1 coloured pink for soma;
tag 2 coloured grey for axon; tag 3 coloured light blue for basal dendrites.

.. _morph-label-seg-fig:

.. figure:: ../gen-images/label_seg.svg
  :width: 600
  :align: center

  A ten segment cable cell, with soma (pink), axon (grey) and dendrite (light blue).
  Python code to generate this cable cell is in the :class:`segment_tree<arbor.segment_tree>`
  documentation :ref:`here <morph-label-seg-code>`.

* The tree is composed of 11 segments (1 soma, 2 axon, 8 dendrite).
* The proximal ends of segments 0 and 9 (the soma and axon hillock respectively) are attached to the root of the tree.
* Segment 2 is a fork, with segments 3 and 5 as children.
* Segment 5 is a fork, with segments 6 and 7 as children.
* There is also a fork at the root, with segments 0 and 9 as children.
* Segments 4, 6, 8 and 10 are terminal segments.

In the example above there are no gaps between segments, however
it is possible for segments to be detached, where the proximal end of a segment is not coincident
with the distal end of its parent. The following morphology has gaps between the start of the
axon and dendritic tree and the soma segment to which they attach.

.. _morph-detached-seg-fig:

.. figure:: ../gen-images/detached_seg.svg
  :width: 600
  :align: center

.. note::
    In Arbor, segments are always treated as though they are connected directly
    to their parents, regardless of whether ends where they attached are collocated.

    Gaps are frequently the result of simplifying the soma,
    whereby the complex geometry of a soma is represented using a cylinder or sphere
    (spheres are represented by a cylinder with length and diameter equal to that of
    the sphere in simulation tools like Arbor and NEURON).

    A gap between a cylindrical soma and segments attached to it does not mean
    that the segmentation is invalid.
    To illustrate why this can occur, consider a potato-shaped soma modelled with a
    cylinder of the same surface area.
    If the cell description places the first segment of a dendritic tree where it attaches to
    the "potato soma", it is unlikely to be collocated with an end of the simplified soma.
    The cell model will correctly represent the location and dimension of the dendritic tree,
    while preserving the soma surface area with a simplified cylindrical model.

Because Arbor supports tapered segments (where radius varies linearly along a segment) it is possible to
represent more complex soma shapes using multiple segments, for example the segmentation below
uses 4 segments to model the soma.

.. _morph-stacked-seg-fig:

.. figure:: ../gen-images/stacked_seg.svg
  :width: 600
  :align: center

.. _morph-morphology:

Morphology
----------

.. glossary::

  morphology
    Morphologies in Arbor are modelled as a set of one dimensional :term:`cables <cable>` of variable radius,
    joined together to form a tree. Only :ref:`cable cells <modelcablecell>` support custom
    morphologies in Arbor. Morphologies can be created by :ref:`loading a file<morph-formats>` with a cell description,
    or by manually constructing one from a :term:`segment tree`.

A segment tree and a morphology can both describe the exact same cable cell geometry, and if you create a morphology *from* a segment tree, they do! The two descriptions differ in two ways:

#. in their 'morphological coordinate system': a :term:`segment tree` is defined in terms
   of connections between :term:`points <mpoint>` in 3D space, while a morphology is defined
   in terms of connections between :term:`branches <branch>`. A segment tree makes it easy to
   recreate a cell from image data, because that is usually done by setting points in space
   and assigning a cable radius. A :term:`morphology` makes accessing locations in terms of
   the cable cells shape easy: "We've placed the clamp on the midway point of the 53rd branch
   of cell B." :ref:`More on placement later <labels>`.
#. the fact that angles between branches are not defined in morphologies but are in segment trees, or,
   equivalently, in morphologies the branching points have no specific position in space, other than in
   relation to a specific distance from other branching points through any branch that links them.

Every segment tree can be used to generate a unique :term:`morphology`, which derives and enumerates
:term:`branches <branch>` from the segments. The branches of a morphology are unbranched cables,
composed of one or more segments, where:

  * the first (proximal) segment of the branch is either a root or the child of fork segment;
  * the last (distal) segment of the branch is either a fork or terminal segment;
  * branches are enumerated in order, following the order of the ids of their proximal segments in the segment tree.

When constructed in this manner, the following statements are true for the branches and
their enumeration:

  * Because a branch must have root, fork or terminal ends, a branch can not be sub-divided
    into two or more branches, and hence there is only one possible set of branches that
    can be derived from a segment tree.
  * Because branches are enumerated according to the id of their proximal segments,
    there is only one branch enumeration representation for a segment tree.
  * However, it is possible for two topologically equivalent morphologies to be
    derived from different segment trees (e.g. two trees with the same segments, however
    different valid segment enumerations), and potentially have different branch numbers.
  * Every valid segment tree can be used to construct a valid morphology.

.. Note::

    Because two topologically-equivalent morphologies may have different segment and
    branch numbering, it is important that model descriptions should avoid referring to
    branches or segments by id.
    This should only be relaxed when the configuration of branches in a particular morphology is known exactly and unambiguously.

To illustrate branch generation, consider the first segment tree example on this page,
which is illustrated along with its branches below.

.. _morph-label-morph-fig:

.. figure:: ../gen-images/label_morph.svg
  :width: 800
  :align: center

  Left, the same 10 segment cable cell seen before. On the right, the associated morphology and branches.
  Note that the :term:`root` point of the soma is always the start (and possibly end) of a branch.
  The code used to generate this morphology is in the :class:`segment_tree<arbor.segment_tree>`
  :ref:`python documentation <morph-label-seg-code>`.

The first branch contains the soma and the first two segments of the dendritic tree.
There are four more branches in the dendritic tree, and one representing the two
segments of the axon.

Note, that though it is possible to create an unbranched sequence of segments composed
of the axon, soma and first two segments in the dendritic tree, this sequence is decomposed
as two branches because segments 0 (soma) and 9 (first segment in axon) are at the
root of the tree.

Similarly to segments, the branches in a morphology have a parent child relationship.
Every branch has one parent, with branches at the root of the tree having the placeholder
parent index :data:`mnpos <arbor.mnpos>`. Segments can have any non-negative number of children,
however by nature of their construction, no branch can have only one child: a branch has
either no children, or two or more children.
The parent-child information and segments for the morphology are summarized:

.. csv-table::
   :widths: 10, 10, 10, 10

   **Branch**, **Parent**, **Children**, **Segments**
   0,          ``mnpos``,  "[1, 2]",       "[0, 1, 2]"
   1,          0,          "[]",           "[3, 4]"
   2,          0,          "[3, 4]",       "[5]"
   3,          2,          "[]",           "[6]"
   4,          2,          "[]",           "[7, 8]"
   5,          ``mnpos``,  "[]",           "[9, 10]"

Gaps between segments do not influence branch creation, hence branches
can contain gaps between segments. Take the example of a morphology with
a gap between the soma and the axon and dendritic trees:

.. figure:: ../gen-images/detached_morph.svg
  :width: 800
  :align: center

The soma is part of branch 0, despite the gap:

.. csv-table::
   :widths: 10, 10, 10, 10

   **Branch**, **Parent**, **Children**, **Segments**
   0,          ``mnpos``,  "[1, 2]",       "[0, 1, 2]"
   1,          0,          "[]",           "[3, 4]"
   2,          0,          "[3, 4]",       "[5]"
   3,          2,          "[]",           "[6]"
   4,          2,          "[]",           "[7, 8]"
   5,          ``mnpos``,  "[]",           "[9]"

Tag information is not used when creating branches, so that a branch can
contain segments with different tags, which in our examples gives branches
that contain both soma and dendrite segments. For example, when building the
soma from multiple segments:

.. figure:: ../gen-images/stacked_morph.svg
  :width: 800
  :align: center

The morphology has the same number of branches as the other examples, with
multiple soma and dendrite segments in branch 0.

.. csv-table::
   :widths: 10, 10, 10, 10

   **Branch**, **Parent**, **Children**, **Segments**
   0,          ``mnpos``,  "[1, 2]",       "[0, 1, 2, 3, 4, 5]"
   1,          0,          "[]",           "[6, 7]"
   2,          0,          "[3, 4]",       "[8]"
   3,          2,          "[]",           "[9]"
   4,          2,          "[]",           "[10, 11]"
   5,          ``mnpos``,  "[]",           "[12, 13]"

.. Note::
    Arbor provides a consistent representation of morphologies with no
    special cases for concepts like magical soma branches, in order to
    build reproducible and consistent model descriptions.

    Users of NEURON who are used to creating a separate soma section
    that is always the first section in a morphology should not
    worry that the soma is not treated as a special branch
    in the examples above.

    The soma in the examples above can be referred to in later model
    building phases, for example when describing the distribution of
    ion channels, by using referring to all parts of the cell with
    :ref:`tag 1 <labels-expressions>`.

.. Note::
   This representation of the cell morphology in terms of *branches* is what
   Arbor uses to create a :ref:`cable cell <cablecell>`, and it is how Arbor
   view's the cell's geometry and refers to it internally.
   :term:`Regions <region>` and :term:`locsets <locset>` formed
   on the cell, are eventually represented either as
   :ref:`subsets of branches <labels-cables>` of the morphology, or exact
   :ref:`locations on branches <labels-locations>` of the morphology.

   Once the morphology is formed from a segment tree, the specific segments
   are no longer of much use for the user and it is better to think of the
   cell structure as Arbor does: in terms of branches.

Examples
~~~~~~~~~~~~~~~

Here we present a series of morphology examples of increasing complexity.
The examples use the Python API are two-dimensional, with the z-dimension set to zero.

.. _morph-tree1:

Example 1: Spherical cell
""""""""""""""""""""""""""""""

A simple model of a cell as a sphere can be modelled using a cylinder with length
and diameter equal to the diameter of the sphere, which will have the same
surface area (disregarding the area of the cylinder's circular ends).

Here a cylinder of length and diameter 5 μm is used to represent a *spherical cell*
with a radius of 2 μm, centred at the origin.

.. code:: Python

    tree = arbor.segment_tree()
    tree.append(mnpos, mpoint(-2, 0, 0, 2), mpoint(2, 0, 0, 2), tag=1)
    morph = arbor.morphology(tree)

.. figure:: ../gen-images/sphere_morph.svg
  :width: 400
  :align: center

  The morphology is a single cylinder segment (left) that forms branch 0 (right).

.. _morph-tree2:

Example 2: Unbranched cable
""""""""""""""""""""""""""""""

Consider a cable of length 10 μm, with a radius that tapers from 1 μm to 0.5 μm
at the proximal and distal ends respectively.
This can be described using a single segment.

.. code:: Python

    tree = arbor.segment_tree()
    tree.append(mnpos, mpoint(0, 0, 0, 1), mpoint(10, 0, 0, 0.5), tag=3)
    morph = arbor.morphology(tree)

.. figure:: ../gen-images/branch_morph1.svg
  :width: 600
  :align: center

  A tapered cable with one cable segment (left), generates a morphology with one branch (right).

The radius of a cable segment varies linearly between its end points. To define an unbranched cable
with irregular radius and "squiggly" shape, use multiple segments to build a piecewise linear reconstruction
of the cable geometry.
This example starts and ends at the same locations as the previous, however it is constructed from 4
distinct cable segments:

.. code:: Python

    tree = arbor.segment_tree()
    tree.append(mnpos, mpoint( 0.0,  0.0,  0.0, 1.0), mpoint( 3.0,  0.2,  0.0, 0.8), tag=1)
    tree.append(0,     mpoint( 3.0,  0.2,  0.0, 0.8), mpoint( 5.0, -0.1,  0.0, 0.7), tag=2)
    tree.append(1,     mpoint( 5.0, -0.1,  0.0, 0.7), mpoint( 8.0,  0.0,  0.0, 0.6), tag=2)
    tree.append(2,     mpoint( 8.0,  0.0,  0.0, 0.6), mpoint(10.0,  0.0,  0.0, 0.5), tag=3)
    morph = arbor.morphology(tree)

.. figure:: ../gen-images/branch_morph2.svg
  :width: 600
  :align: center

  The morphology is an unbranched cable comprised of 4 cable segments,
  coloured according to their tags: tag 1 pink; tag 2 grey; tag 3 light blue (left).
  The four segments form one branch (right).

Gaps are possible between two segments. The example below inserts a 1 μm gap between the second
and third segments of the previous morphology. Note that Arbor will ignore the gap, effectively
joining the segments together, such that the morphology with the gap is the same as that without.

.. code:: Python

    tree = arbor.segment_tree()
    tree.append(mnpos, mpoint( 0.0,  0.0,  0.0, 1.0), mpoint(3.0,  0.2,  0.0, 0.8), tag=1)
    tree.append(0,     mpoint( 3.0,  0.2,  0.0, 0.8), mpoint(5.0, -0.1,  0.0, 0.7), tag=2)
    tree.append(1,     mpoint( 7.0, -0.1,  0.0, 0.7), mpoint(10.0, 0.0,  0.0, 0.6), tag=2)
    tree.append(2,     mpoint(10.0,  0.0,  0.0, 0.6), mpoint(12.0, 0.0,  0.0, 0.5), tag=3)
    morph = arbor.morphology(tree)

.. figure:: ../gen-images/branch_morph3.svg
  :width: 600
  :align: center

  There is a gap between segment 1 and segment 2 (left), and there is a single branch (right).

The radius of a cable is piecewise linear, with discontinuities permitted at the
interface between segments.
The next example adds a discontinuity to the previous example between segments
3 and 4, where the radius changes from 0.5 μm to 0.3 μm:

.. code:: Python

    tree = arbor.segment_tree()
    tree.append(mnpos, mpoint( 0.0,  0.0,  0.0, 1.0), mpoint( 3.0,  0.2,  0.0, 0.8), tag=1)
    tree.append(0,     mpoint( 3.0,  0.2,  0.0, 0.8), mpoint( 5.0, -0.1,  0.0, 0.7), tag=2)
    tree.append(1,     mpoint( 5.0, -0.1,  0.0, 0.7), mpoint( 8.0,  0.0,  0.0, 0.5), tag=2)
    tree.append(2,     mpoint( 8.0,  0.0,  0.0, 0.3), mpoint(10.0,  0.0,  0.0, 0.5), tag=3)
    morph = arbor.morphology(tree)

.. figure:: ../gen-images/branch_morph4.svg
  :width: 600
  :align: center

  The resulting morphology has a step discontinuity in radius.

.. _morph-example4:

Example 3: Y-shaped cell
""""""""""""""""""""""""""""""

The simplest branching morphology is a cable that bifurcates into two branches,
which we will call a *y-shaped cell*.
In the example below, the first branch of the tree is a cable of length 10 μm with a
a radius that tapers from 1 μm to 0.5 μm.
The two child branches are attached to the end of the first branch, and taper from from 0.5 μ m
to 0.2 μm.

Note that only the distal point is required to describe the child segments,
because the proximal end of each child segment has the same location and
radius as the distal end of the parent.

.. code:: Python

    tree = arbor.segment_tree()
    tree.append(mnpos, mpoint( 0.0, 0.0, 0.0, 1.0), mpoint(10.0, 0.0, 0.0, 0.5), tag= 3)
    tree.append(0,     mpoint(15.0, 3.0, 0.0, 0.2), tag= 3)
    tree.append(0,     mpoint(15.0,-3.0, 0.0, 0.2), tag= 3)
    morph = arbor.morphology(tree)

.. figure:: ../gen-images/yshaped_morph.svg
  :width: 800
  :align: center

Example 4: Soma with branches
""""""""""""""""""""""""""""""

Now let's look at cell with a simple dendritic tree attached to a spherical soma.
The spherical soma of radius 3 μm is modelled with a cylinder with length and
diameter equal to 6 μm, which has the same surface area as the sphere.

.. code:: Python

    tree = arbor.segment_tree()
    tree.append(mnpos, mpoint(-3.0, 0.0, 0.0, 3.0), mpoint( 3.0, 0.0, 0.0, 3.0), tag=1)
    tree.append(0, mpoint( 4.0, -1.0,  0.0, 0.6), mpoint(10.0,  -2.0,  0.0, 0.5), tag=3)
    tree.append(1, mpoint(15.0, -1.0,  0.0, 0.5), tag=3)
    tree.append(2, mpoint(18.0, -5.0,  0.0, 0.3), tag=3)
    tree.append(2, mpoint(20.0,  2.0,  0.0, 0.3), tag=3)
    morph = arbor.morphology(tree)


.. figure:: ../gen-images/ysoma_morph1.svg
  :width: 900
  :align: center

  Note that branch 0 (right) is composed of segments 0, 1, and 2 (left).

The soma is the first segment, labelled with tag 1. The dendritic tree is a simple
y-shaped tree composed of 4 segments, each labelled with tag 3.
The first branch is composed of 3 segments: the soma segment and the first two segments
in the dendritic tree because the segments have parent child ordering and no fork points.

.. note::
    The first branch is derived directly from the topological relationship between the segments,
    and no special treatment is given to the soma.
    There is no need to treat segments with different tags (e.g. tags that we might associate
    with soma, axon, basal dendrite and apical dendrite) when defining geometric primitives like
    segments and branches, because they can later be referenced later using
    :ref:`region expressions <labels-expressions>`.

Now we can attach another dendrite and an axon to the soma, to make a total of three cables
attached to the soma (two dendrites and an axon).
The dendrites are attached to the distal end of the soma (segment 0), so they have the
0 as their parent.
The axon is attached to the proximal end of the soma, which is at the root of the tree,
so it has :data:`mnpos` as its parent.
There are 7 branches generated from 10 segments, and soma segment is its own branch,
because it has two children: the dendrites attached to its distal end.

.. figure:: ../gen-images/ysoma_morph2.svg
  :width: 900
  :align: center


.. note::
    The discretisation process, which converts :term:`segments <segment>` and
    :term:`branches <branch>` into :term:`control volumes <control volume>`,
    will ignore gaps between segments in the input. The cell below, in which the dendrites
    and axon have been translated to remove any gaps, is equivalent to the previous example
    for the back end simulator.

    Note that the dendrites are children of the soma segment, so they are coincident with
    the distal end of the soma, and the axon is translated to the proximal end of the
    soma segment because both it and the soma have :py:data:`mnpos <arbor.mnpos>` as a parent.
    More generally, segments at the root of the tree are connected electrically at their
    proximal ends.

    .. figure:: ../gen-images/ysoma_morph3.svg
      :width: 900
      :align: center

.. _morph-formats:

Supported file formats
----------------------

Arbor supports morphologies described using the SWC file format and the NeuroML file format.

SWC
~~~

Arbor supports reading morphologies described using the
`SWC <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_ file format.

SWC files may contain comments, which are stored as metadata. And a blank line anywhere in the file is
interpreted as end of data.

The description of the morphology is encoded as a list of samples with an id,
an `x,y,z` location in space, a radius, a tag and a parent id. Arbor parses these samples, performs some checks,
then generates a morphology according to one of three possible interpretations.

The SWC file format specifications are not very detailed, which has lead different simulators to interpret
SWC files in different ways, especially when it comes to the soma. Arbor has its own an interpretation that
is powerful and simple to understand at the same time. However, we have also developed functions that will
interpret SWC files similarly to how the NEURON simulator would, and how the Allen Institute would.

Despite the differences between the interpretations, there is a common set of checks that are always performed
to validate an SWC file:

* Check that there are no duplicate ids.
* Check that the parent id of a sample is less than the id of the sample.
* Check that the parent id of a sample refers to an existing sample.

In addition, all interpretations agree that a *segment* is (in the common case) constructed between a sample and
its parent and inherits the tag of the sample; and if more than 1 sample have the same parent, the parent sample
is interpreted as a fork point in the morphology, and acts as the proximal point to a new branch for each of its
"child" samples. There a couple of exceptions to these rules which are listed below.

Arbor interpretation:
"""""""""""""""""""""
In addition to the previously listed checks, the arbor interpretation explicitly disallows SWC files where the soma is
described by a single sample. It constructs the soma from 2 or more samples, forming 1 or more segments. A *segment* is
always constructed between a sample and its parent. This means that there are no gaps in the resulting morphology.

Arbor has no magic rules or transformations for the soma. It can be a single branch or multiple branches; segments
of a different tag can connect to its distal end, proximal end or anywhere in the middle. For example, to create a
morphology with a single segment soma; a single segment axon connected to one end of the soma; and a single segment
dendrite connected to the other end of the soma, the following swc file can be used:


.. literalinclude :: example.swc
   :language: python
   :linenos:

Samples 1 and 2 will form the soma; samples 1 and 3 will form the axon, connected to the soma at the proximal end;
samples 2 and 4 will form the dendrite, connected to the soma at the distal end. The morphology will look something
like this:

.. figure:: ../gen-images/swc_morph.svg
   :width: 400
   :align: center


Allen interpretation:
"""""""""""""""""""""
In addition to the previously mentioned checks, the Allen interpretation expects a single-sample soma to be the first
sample of the file and to be interpreted as a spherical soma. Arbor represents the spherical soma as a cylinder with
length and diameter equal to the diameter of the sample representing the sphere.

This interpretation also expects that samples have the same tag as their parent samples, with the exception of samples
that have the soma sample as a parent. In this case, when a sample's parent is the soma, no *segment* is created
between the 2 samples; instead there is a gap in the morphology (represented electrically as a zero-resistance wire).
Samples with the soma as a parent start new segments, that connect to the distal end of the soma if they are dendrites,
or to the proximal end of the soma if they are axons or apical dendrites. Only axons, dendrites and apical dendrites
(tags 2, 3 and 4 respectively) are allowed in this interpretation, in addition to the spherical soma.

Finally the Allen institute interpretation of SWC files centres the morphology around the soma at the origin (0, 0, 0)
and all samples are translated in space towards the origin.

NEURON interpretation:
""""""""""""""""""""""
The NEURON interpretation was obtained by experimenting with the ``Import3d_SWC_read`` function. We came up with the
following set of rules that govern NEURON's SWC behavior and enforced them in arbor's NEURON-complaint SWC
interpreter:

* SWC files must contain a soma sample and it must to be the first sample.
* A soma is represented by a series of n≥1 unbranched, serially listed samples.
* A soma is constructed as a single cylinder with diameter equal to the piecewise average diameter of all the
  segments forming the soma.
* A single-sample soma at is constructed as a cylinder with length=diameter.
* If a non-soma sample is to have a soma sample as its parent, it must have the most distal sample of the soma
  as the parent.
* Every non-soma sample that has a soma sample as its parent, attaches to the created soma cylinder at its midpoint.
* If a non-soma sample has a soma sample as its parent, no segment is created between the sample and its parent,
  instead that sample is the proximal point of a new segment, and there is a gap in the morphology (represented
  electrically as a zero-resistance wire)
* To create a segment with a certain tag, that is to be attached to the soma, we need at least 2 samples with that
  tag.

NeuroML
~~~~~~~

Arbor offers limited support for models described in `NeuroML version 2 <https://neuroml.org/neuromlv2>`_.
This is not built by default (see :ref:`NeuroML support <install-neuroml>` for instructions on how
to build arbor with NeuroML).

Once support is enabled, Arbor is able to parse and check the validity of morphologies described in NeuroML files,
and present the encoded data to the user.  This is more than a simple a `segment tree`.

NeuroML can encode in the same file multiple top-level morphologies, as well as cells:

.. code:: XML

   <neuroml xmlns="http://www.neuroml.org/schema/neuroml2">
   <morphology id="m1">
       <segment id="seg-0">
           <proximal x="1" y="1" z="1" diameter="1"/>
           <distal x="2" y="2" z="2" diameter="2"/>
       </segment>
       <segmentGroup id="group-0">
           <member segment="1"/>
       </segmentGroup>
   </morphology>
   <morphology id="m2"/>
   <cell id="c1" morphology="m1"/>
   <cell id="c2">
       <morphology id="m3"/>
   </cell>
   </neuroml>

The above NeuroML description defines 2 top-level morphologies ``m1`` and ``m2`` (empty); a cell ``c1`` that uses
morphology ``m1``; and a cell ``c2`` that uses an internally defined (empty) morphology ``m3``.

Arbor can query the cells and morphologies using their ids and return all the associated morphological data for each.
The morphological data includes the actual morphology as well as the named segments and groups of the morphology.
For example, the above ``m1`` morphology has one named segment ``seg-0`` and one named group ``group-0`` that are
both represented using Arbor's :ref:`region expressions <labels-expressions>`.

API
---

* :ref:`Python <pymorph>`
* :ref:`C++ <cppcablecell-morphology-construction>`
