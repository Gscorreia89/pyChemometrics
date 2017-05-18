Using the pyChemometrics Objects
--------------------------------

Principal Component Analysis
============================

Principal Component Analysis is ...
atasets in the nPYc toolbox are represented by instances of sub-classes of the :py:class:`~nPYc.objects.Dataset` class. Each instance of a :py:class:`~nPYc.objects.Dataset` represents the measured abundances of features, plus sample and feature metadata, in a metabolic profiling experiment.

Initializing a pyChemometricsPCA object with the default options and Unit-Variance scaling::

	pca_model = pyChemometrics.ChemometricsPCA(...)

The pyChemometrics objects follow a similar lofic Similarly to scikit-learn::

	dataset.noSamples

The main::

	dataset.sampleMetadata
	dataset.featureMetadata
	dataset.intensityData

In addition to all the scikit-learn like methods, Routine methods for K-Fold cross validation, permutation tests and 

`ISA-TAB <http://isa-tools.org>`_ format study design documents provide the simplest method for mapping experimental design parameters into the object::

	datasetObject.addSampleInfo(descriptionFormat='ISATAB', filePath='~/path to study file')

Or if analytical data is also represented in ISA-TAB, Dataset objects may be instantiated from the ISA-TAB documents [#]_::

	dataset = nPYc.Dataset('~/path to ISATAB study directory/, fileType='ISATAB', assay='assay name')

Where analytical file names have been generated according to a standard that allows study design parameters to parsed out, this can be accomplished be means of a regular expression that captures paramaters in named groups::

	datasetObject.addSampleInfo(descriptionFormat='Filenames', filenameSpec='regular expression string')

Mapping metadata into an object is an accumulative operation, so multiple calls can be used to map metadata from several sources\:

.. code-block:: python
	:linenos:

    	# Load analytical data to sample ID mappings
    	datasetObject.addSampleInfo(descriptionFormat='NPCLIMS', filePath='~/path to LIMS file')

    	# Use the mappings to map in sample metadata
    	datasetObject.addSampleInfo(descriptionFormat='NPC Subject Info', filePath='~/path to Subject Info file')

    	# Get samples info from filenames
    	datasetObject.addSampleInfo(descriptionFormat='filenames')

    See the documentation for :py:meth:`~nPYc.objects.Dataset.addSampleInfo` for possible options.

Partial Least Squares
=====================

The nPYc toolbox incorporates the concept of analytical quality directly into the subclasses of :py:class:`~nPYc.objects.Dataset`. Depending on the analytical platform and protocol, quality metrics may be judged on a sample-by-sample, or feature-by-feature basis, or both.

To generate reports of analytical quality, call the :py:func:`~nPYc.reports.generateReport` function, with the dataset object as an argument::

	nPYc.reports.generateReport(datasetObject, 'feature summary')


Quality-control of UPLC-MS profiling datasets
*********************************************

By default the nPYc toolbox assumes an :py:class:`~nPYc.objects.MSDataset` instance contains untargeted peak-picked UPLC-MS data, and defines two primary quality control criteria for the features detected, as outlined in Lewis *et al.* [#]_.

* Precision of measurement
	A Relative Standard Deviation (RSD) threshold ensures that only features measured with a precision above this level are propagated on to further data analysis. This can be defined both in absolute terms, as measured on reference samples, but also by removing features where analytical variance is not sufficiently lower than biological variation.
	In order to characterise RSDs, the dataset must include a sufficient number of precision reference samples, ideally a study reference pool to allow calculation of RSDs for all detected features.
* Linearity of response
	By filtering features based on the linearity of their measurement *vs* concentration in the matrix, we ensure that only features that can be meaningfully related to the study design are propagated into the analysis.
	To asses linearity, features must be assayed across a range of concentrations, again in untargeted assays, using the pooled study reference will ensure all relevant features are represented.

Beyond feature QC, the toolbox also allows for the detection and reduction of analytical run-order and batch effects.


Quality-control of NMR profiling datasets
*****************************************

:py:class:`~nPYc.objects.NMRDataset` objects containing spectral data, may have their per-sample analytical quality assessed on the criteria laid out in Dona *et al.* [#]_, being judged on:

* Line-width
	By default, line-widths below 1.4\ Hz, are considered acceptable
* Even baseline
	The noise in the baseline regions flanking the spectrum are expected to have equal means (centred on zero), and variances
* Adequate water-suppression
	The residual water signal should not affect the spectrum outside of the 4.9 to 4.5 ppm region

Before finalising the dataset, typically the wings of the spectrum will be trimmed, and the residual water signal and references peaks removed.


Filtering of samples *&* variables
**********************************

Filtering of features by the generic procedures defined for each type of dataset, using the thresholds load from the SOP and defined in :py:attr:`~nPYc.objects.Dataset.Attributes` is accomplished with the :py:meth:`~nPYc.objects.Dataset.updateMasks` method. When called, the elements in the  :py:attr:`~nPYc.objects.Dataset.featureMask` are set to ``False`` where the feature does not meet quality criteria, and nd elements in :py:attr:`~nPYc.objects.Dataset.sampleMask` are set to ``False`` for samples that do not pass quality criteria, or sample types and roles not specified.

The defaults arguments to py:meth:`~nPYc.objects.Dataset.updateMasks` will filter the dataset to contain only study and study reference samples and only those features meeting quality criteria::

	dataset.updateMasks(filterSamples=True, filterFeatures=True, sampleTypes=[<SampleType.StudySample>, <SampleType.StudyPool>], assayRoles=[<AssayRole.Assay>, <AssayRole.PrecisionReference>])

Specific samples or features may be excluded based on their ID or other associated metadata with the :py:meth:`~nPYc.objects.Dataset.excludeFeatures` and :py:meth:`~nPYc.objects.Dataset.excludeSamples` methods.

These methods operate by setting the relevant entries in the :py:attr:`~nPYc.objects.Dataset.featureMask` and :py:attr:`~nPYc.objects.Dataset.sampleMask` vectors to ``False``, which has the effect of hiding the sample or feature from further analysis. Elements masked from the dataset may then be permanently removed by calling the :py:meth:`~nPYc.objects.Dataset.applyMasks` method.

Normalisation
=============

Dilution effects on global sample intensity can be normalised using the functions in the :py:mod:`~nPYc.utilities.normalise` sub-module.


.. [#] Not yet implemented.

.. [#] Development and Application of Ultra-Performance Liquid Chromatography-TOF MS for Precision Large Scale Urinary Metabolic Phenotyping, Lewis MR, *et al.*, **Anal. Chem.**, 2016, 88, pp 9004-9013

.. [#] Precision High-Throughput Proton NMR Spectroscopy of Human Urine, Serum, and Plasma for Large-Scale Metabolic Phenotyping, Anthony C. Dona *et al.* **Anal. Chem.**, 2014, 86 (19), pp 9887–9894
