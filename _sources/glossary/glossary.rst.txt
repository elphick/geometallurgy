Glossary
========

.. glossary::

    mass
        The mass of the samples.  Dry mass is mandatory, either by supplying it directly, or by back-calculation
        from wet mass and H2O.  Without dry mass composition cannot be managed, since composition is on a dry basis.

    composition
        The composition of the DRY mass.  Typically chemical composition is of interest (elements and oxides),
        however mineral composition is/will be supported.

    MassComposition
        A class that holds the mass and composition of a sample.

    Stream
        A class that inherits MassComposition, and has a source and destination property. It is used to represent
        a MassComposition object flowing between two nodes in a network. Synonymous with a stream in a processing
        flowsheet.