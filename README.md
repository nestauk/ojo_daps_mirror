# Welcome to the Open Jobs Observatory

## Contents
  * [About the Open Jobs Observatory](#about-the-open-jobs-observatory)
  * [**Download the Data**: Labour Market Statistics](#download-the-data-labour-market-statistics)
  * [For developers](#for-developers)

## About the Open Jobs Observatory

The [Open Jobs Observatory](https://www.nesta.org.uk/data-visualisation-and-interactive/open-jobs-observatory) is a pilot project that provides free and up-to-date insights on the skills requested in UK job adverts. The Open Jobs Observatory was created by [Nesta](http://www.nesta.org.uk/), in partnership with the [Department for Education](https://www.gov.uk/government/organisations/department-for-education). This page contains all the data series released from the Observatory, as well as our code for extracting key variables from adverts (which include skills<!--, locations and occupation--> and locations).

### Why did we build the Observatory?

Job adverts are a rich source of labour market information; they allow us to quickly detect changes in the skills demanded by employers, and explore how skill demands vary by geography<!--, occupation--> and industry. Information on skill demands is not available from official vacancy statistics, and the need for timely and high-quality intelligence about skills has only been heightened by the COVID-19 pandemic. 

The Observatory is a pilot project and we welcome any feedback and questions. Please email dataanalytics\<at\>nesta.org.uk.


## **Download the Data**: Labour Market Statistics

These data series are highly experimental and are subject to revision. All series are updated on a monthly basis, although there may be occasional short delays.

The weekly time series are based on the estimated stock of live job adverts on the Monday of each week. The monthly snapshot series are based on the estimated stock of live adverts on the 15th day of the latest month. The estimated stock on any given day is the total number of new job adverts (excluding duplicates) that were collected in the 6 weeks leading up to that particular day. The key assumption, underpinning this approach, is that adverts remain live for 6 weeks.

The skills have been clustered into a skills taxonomy and it contains three levels (Level 0 is the broadest and Level 2 is the most granular).

None of the series are seasonally adjusted and the salary series are not adjusted for inflation.

Unfortunately, we cannot share the underlying dataset of job adverts that have been collected. More information on the strengths and weaknesses of job adverts, as a source of data, can be found on the [home page for the Observatory](https://www.nesta.org.uk/data-visualisation-and-interactive/open-jobs-observatory/).

| ID of data series 	| Category of data series 	| Name of data series 	| Download data (CSV) 	| Data dictionary 	| Description 	| Weekly time series or monthly snapshot 	| Normalised to April 2021? (see note below) 	|
|:---:	|:---:	|---	|:---:	|:---:	|---	|:---:	|:---:	|
| 1 	| General 	| Volume of online job adverts  	| [here](https://open-jobs-indicators.s3.eu-west-1.amazonaws.com/dev/latest/weekly_stock.csv) 	| [here](https://open-jobs-indicators.s3.eu-west-1.amazonaws.com/dev/latest/weekly_stock_data_dict.txt)	| The estimated stock of online job adverts on each Monday 	| Weekly time series 	| Y 	|
| 2 	| Locations 	| Volume of online job adverts by region 	| [here](https://open-jobs-indicators.s3.eu-west-1.amazonaws.com/dev/latest/weekly_loc_vacancies.csv) 	| [here](https://open-jobs-indicators.s3.eu-west-1.amazonaws.com/dev/latest/weekly_loc_vacancies_data_dict.txt)	| Volume of online job adverts by NUTS 2 region 	| Weekly time series 	| Y 	|
| 3 	| Locations 	| Mix of skills mentioned in online job adverts for each region 	| [here](https://open-jobs-indicators.s3.eu-west-1.amazonaws.com/dev/latest/skills_cats_by_loc_snapshot.csv) 	| [here](https://open-jobs-indicators.s3.eu-west-1.amazonaws.com/dev/latest/skills_cats_by_loc_snapshot_data_dict.txt) | Mix of skills mentioned in online job adverts that have been assigned to each NUTS 2 region 	| Monthly snapshot 	|  	|
| 4 	| Skills 	| Mix of skills mentioned in online job adverts 	| [here](https://open-jobs-indicators.s3.eu-west-1.amazonaws.com/dev/latest/skills_demand_snapshot.csv) 	| [here](https://open-jobs-indicators.s3.eu-west-1.amazonaws.com/dev/latest/skills_demand_snapshot_data_dict.txt) 	| Mix of skills mentioned, in online job adverts, at the most granular level of the skills taxonomy 	| Monthly snapshot 	|  	|
| 5 	| Advertised salaries 	| Annualised salaries mentioned in online job adverts 	| [here](https://open-jobs-indicators.s3.eu-west-1.amazonaws.com/dev/latest/weekly_salary_spread.csv) 	| [here](https://open-jobs-indicators.s3.eu-west-1.amazonaws.com/dev/latest/weekly_salary_spread_data_dict.txt)	| Quartiles for the annualised salary ranges mentioned in online job adverts (£000s pa) 	| Weekly time series 	|  	|
| 6 	| Advertised salaries 	| Annualised salaries mentioned in online job adverts that also contain select skills 	| [here](https://open-jobs-indicators.s3.eu-west-1.amazonaws.com/dev/latest/skills_salary_spread_snapshot.csv) 	| [here](https://open-jobs-indicators.s3.eu-west-1.amazonaws.com/dev/latest/skills_salary_spread_snapshot_data_dict.txt) 	| Quartiles for the annualised salary ranges mentioned in online job adverts (£000s pa) that also mention select skills 	| Monthly snapshot 	|

<!-- | 5 	| Occupations 	| Volume of online job adverts by occupation group 	| [here](https://open-jobs-indicators.s3.eu-west-1.amazonaws.com/dev/latest/skill_cats_by_soc_snapshot.csv) 	| [here](path/to/list/datadict)	| Volume of job adverts by first digit SOC code  	| Weekly time series 	| Y 	|  	|
| 6 	| Occupations 	| Mix of skills mentioned in online job adverts for select occupation groups 	| [here](https://open-jobs-indicators.s3.eu-west-1.amazonaws.com/dev/latest/skill_cats_by_soc_snapshot.csv) 	| [here](path/to/list/datadict)  	| Mix of skills mentioned in online job adverts that have been assigned to select occupation groups (4-digit SOC codes) 	| Monthly snapshot 	|  	| The occupation groups chosen were those which had the largest number of adverts assigned to them.  	| -->

### Notes:

* The data series are updated on a monthly basis, and all job adverts and features reflect the latest version of our codebase. Codebase changes after September 2021 are described in our [CHANGELOG](CHANGELOG). If you wish to pin your data to a fixed version then you should swap the term `latest` in the URL of the CSV to a specific version number: you can find [a list of versions here](https://open-jobs-indicators.s3.eu-west-1.amazonaws.com/dev/versions.txt). Note that pinning your data to a fixed version means that you will be using a snapshot of the data, frozen at that point in time (i.e. no new data collected after that version is available).
* NUTS refers to the [Nomenclature of Territorial Units for Statistics](https://ec.europa.eu/eurostat/web/nuts/background)
* The NUTS 2 areas in London have been merged, as job adverts tend not to be geographically specific within London.
* For series 3: the stock of adverts in each region has been indexed to its own average stock level in April 2021.
* Many job adverts (approximately 37%) do not contain a salary. When salaries are given, they are typically advertised as a range (£`MIN`-£`MAX`). This creates two salary series: the lower values from these ranges (`MIN`), and the upper values (`MAX`). In instances where a single salary is given we assign this value to both `MIN` and `MAX`.

<!-- * In the case of locations (series 3) and occupations (series 5), each region or occupation is normalised to its own average stock in April 2021 -->


## For developers

This is a public mirror of the [production system](https://github.com/nestauk/daps_utils) for the Open Jobs Observatory. The intention of the private version of this codebase is to continuously collect online job advert data and then to extract, enrich and aggregate labour market information from the dataset. At present, code in this mirror is not intended to be run out-of-the-box: rather it is primarily intended for reference to our methodology.

There are three outstanding tasks/issues which prevent code in this mirror from being run out-of-the-box:

* We plan to factor out the data science components (which appear under `flows/enrich/labs`) which are currently coupled to our infrastructure. At that point, models which are able to predict, for example, skills, occupational and industrial codes and job titles from a raw job description will be made available as a separate and distinct public package.
* Unfortunately, we are unable to release the code that we use to collect adverts.
* In order to run this codebase you will need access to cloud infrastructure that maps onto that of the project. We have partially implemented `terraform` for some part of our infrastructure, but not for others. Until this has been fully implemented it will not be straightforward for users to run pipelines in this codebase. At that stage users would also need to take responsibility for the budget of the infrastructure, which we are not liable for.


## License

- Codebase:  [MIT](LICENSE)
- Open Data: [ODbL](.mirror/DATA-LICENSE)
