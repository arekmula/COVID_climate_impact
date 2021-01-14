# COVID19 - climate impact
Project for Data Science course @ Poznan University of Technology, Institute of Robotics and Machine Intelligence.

### The goal of the project was to verify hypotheses about COVID-19:
- Temperature has a big impact on COVID-19 reproduction. 
  * This hypothesis should be checked using ANOVA test (Mean temperature per month and country vs Normalized reproduction coefficient per country and month). 
- There's a big difference in death ratio between different countries from Europe. This analysis should be done in 2 ways.
   *  chi2 test (number of deaths vs number of confirmed cases)
   *  ANOVA test (country vs death ratio per month)

### Input data
- The data about COVID-19 comes from [JHU CSSE COVID-19 Dataset](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series)
- The data about temperature in each country comes from [TerraClimate Dataset](http://www.climatologylab.org/terraclimate.html)

### Project requirements:
- only one .py script 
- input files given by a relative path, directly from working directory
- Python 3.8
- see `requirements.txt`