import os
from ..objects import Dataset, MSDataset, NMRDataset, TargetedDataset
from ..reports._generateReportMS import _generateReportMS # , _generateReportNMR
from ..reports._generateSampleReport import _generateSampleReport
from ..reports._generateReportNMR import _generateReportNMR
from ..reports._generateReportTargeted import _generateReportTargeted

def PrincipalComponentAnalysisReport(data, reportType, destinationPath=None, **kwargs):
	"""
	Generates one of a range of reports visualising different qualities of the dataset. Reports can be plotted interactively, or saved to disk.

	reportType **'sample summary'**

	Summarises samples in the dataset. Lists samples acquired, plus if possible, those missing as based on the expected sample manifest.

	Table 1: Summary of Samples Acquired. Lists the number of samples acquired broken down by sample type (if information available). Also lists numbers of that total marked for exclusion (samples set to False in sample mask, and those marked as _x in MS data), missing from LIMS (if available) and with missing sample information (if available). Finally, if any samples have already been excluded, these numbers are listed in an additional column at the end (note these numbers are not included in the total).

	Table 2: Summary of Samples Missing from Acquisition/Import (i.e., present in LIMS but not acquired/imported). If LIMS available, this table lists the number of samples which were listed in the LIMS file but missing from acquisition. Samples marked as *missing* were not provided, samples marked as *sample* were expected, this table also documents those which have already been excluded.

	Remaining tables: If any samples are listed in Table 1 as *Marked for Exclusion*, *Missing Subject Information*, or in Table 2 as *Marked as Sample* (but not *Already Excluded*), the details of these samples are listed in subsequent tables.

	reportType options specific for MSDataset:

	* **'feature summary'** Generates feature summary report, plots figures including those for feature abundance, sample TIC and acquisition structure, correlation to dilution, RSD and an ion map
	* **'correlation to dilution'** Generates a more detailed report on correlation to dilution, broken down by batch subset with TIC, detector voltage, a summary, and heat-map indicating potential saturation or other issues
	* **'batch correction assessment'** Generates a report before batch correction showing TIC overall and intensity and batch correction fit for a subset of features, to aid specification of batch start and end points
	* **'batch correction summary'** Generates a report post batch correction with pertinant figures (TIC, RSD etc.) before and after
	* **'feature selection'** Generates a summary of the number of features passing feature selection (with current settings as definite in the SOP), and a heat-map showing how this number would be affected by changes to RSD and correlation to dilution thresholds
	* **'final report'** Generates a summary of the final dataset, lists sample numbers present, a selection of figures summarising dataset quality, and a final list of samples missing from acquisition
	* **'BI-LISA'** Plots BI-LISA datasets, visualising internal correlation of parameters
	* **'BI Quant-UR'** Plot BI Quant-UR datasets, visualising feature distributions
	* **'merge loq assessment'** Generates a report before :py:meth:`~TargetedData.mergeLimitsOfQuantification`, highlighting the impact of updating limits of quantification across batch. List and plot limits of quantification that are altered, number of samples impacted.

	Generating reports requires the presence of at least two Study Samples and two Study-Reference samples in the dataset in order to generate aggregate statistics.

	:param Dataset data: Dataset object to report on
	:param str reportType: Type of report to generate. If MSDataset: one of **'sample summary'**, **'feature summary'**, **'correlation to dilution'**, **'batch correction'**, **'feature selection'**, or **'final report`'**. If NMRDataset: one of **'sample summary'**, **'feature summary'**, or **'final summary'**.
	:param destinationPath: If ``None`` plot interactively, otherwise save the figure to the path specified
	:type destinationPath: None or str
	:param bool withExclusions: If ``True``, only report on features and samples not masked by the sample and feature masks
	:param MSDataset msDataCorrected: Only if **'batch correction'**, if msDataCorrected included will generate report post correction
	:param PCAmodel pcaModel: Only if **'final report'**, if PCAmodel object is available PCA scores plots coloured by sample type will be added to report
	:param bool returnOutput: Only if **'sample summary'**, if ``True``, returns a dictionary of all tables generated during run
	"""

	# Check inputs
	if not isinstance(data, Dataset):
		raise TypeError('data must be an instance of nPYc.Dataset')

	if isinstance(data, MSDataset):
		acceptAllOptions = {'sample summary', 'feature summary', 'correlation to dilution', 'batch correction assessment', 'batch correction summary', 'feature selection', 'final report', 'final report abridged', 'final report peakpanther'}
	elif isinstance(data, NMRDataset):
		acceptAllOptions = {'sample summary', 'feature summary', 'final report'}
	elif isinstance(data, TargetedDataset):
		acceptAllOptions = {'sample summary', 'feature summary', 'merge loq assessment', 'final report'}
	if not isinstance(reportType, str) & (reportType.lower() in acceptAllOptions):
		raise ValueError('reportType must be == ' + str(acceptAllOptions))

	if destinationPath is not None:
		if not isinstance(destinationPath, str):
			raise TypeError('destinationPath must be a string')

	# Create directory to save destinationPath
	if destinationPath:
		if not os.path.exists(destinationPath):
			os.makedirs(destinationPath)
		if not os.path.exists(os.path.join(destinationPath, 'graphics')):
			os.makedirs(os.path.join(destinationPath, 'graphics'))


	# Generate sample summary report
	if reportType.lower() == 'sample summary':
		_generateSampleReport(data, destinationPath=destinationPath, **kwargs)

	# Generate method specific summary report
	else:
		if isinstance(data, MSDataset):
			_generateReportMS(data, reportType.lower(), destinationPath=destinationPath, **kwargs)
		if isinstance(data, NMRDataset):
			_generateReportNMR(data, reportType.lower(), destinationPath=destinationPath, **kwargs)
		if isinstance(data, TargetedDataset):
			_generateReportTargeted(data, reportType.lower(), destinationPath=destinationPath, **kwargs)