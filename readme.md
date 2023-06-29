## This is a difficult regression task, where the aim is to predict the burned area of forest fires, in the northeast region of Portugal, by using meteorological and other data (see details at: http://www.dsi.uminho.pt/~pcortez/forestfires).
For more information, read [Cortez and Morais, 2007].
   1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9
   2. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9
   3. month - month of the year: 'jan' to 'dec' 
   4. day - day of the week: 'mon' to 'sun'
   5. FFMC - Fine Fuel Moisture Code index from the FWI system: 18.7 to 96.20
   6. DMC - Duff Moisture Code index from the FWI system: 1.1 to 291.3 
   7. DC - Drought Code index from the FWI system: 7.9 to 860.6   (moisture in soil)
   8. ISI - Initial Spread Index index from the FWI system: 0.0 to 56.10
   9. temp - temperature in Celsius degrees: 2.2 to 33.30
   10. RH - relative humidity in %: 15.0 to 100
   11. wind - wind speed in km/h: 0.40 to 9.40 
   12. rain - outside rain in mm/m2 : 0.0 to 6.4 
   13. area - the burned area of the forest (in ha): 0.00 to 1090.84 
   (this output variable is very skewed towards 0.0, thus it may make
    sense to model with the logarithm transform).

Importance Features:

FFMC: Fine Fuel Moisture Code (FFMC) is a measure of the moisture content of fine fuels, such as grass and leaves.
DMC: Duff Moisture Code (DMC) is a measure of the moisture content of duff, which is the layer of dead organic matter that lies on the forest floor.
DC: Drought Code (DC) is a measure of the moisture content of the soil.
ISI: Initial Spread Index (ISI) is a measure of how easily a fire can spread.
Temp: Temperature is a measure of the air temperature.
RH: Relative Humidity (RH) is a measure of the amount of water vapor in the air.
Wind: Wind speed is a measure of the speed of the wind.
Rain: Rain is a measure of the amount of rainfall.


Histograms: Histograms are a good way to visualize the distribution of continuous variables, such as temperature, relative humidity, and wind speed. or temp and fire
Scatter plots: Scatter plots are a good way to visualize the relationship between two continuous variables, such as temperature and area of the fire.
Bar charts: Bar charts are a good way to visualize the distribution of categorical variables, such as month and day of the fire.
Heatmaps: Heatmaps are a good way to visualize the correlation between multiple variables.

we could use a histogram to visualize the distribution of the temperature of the fires. This would help you to see if there are any particular temperatures that are more likely to lead to fires.

we could use a scatter plot to visualize the relationship between the temperature and area of the fires. This would help you to see if there is any correlation between these two variables.

we could use a bar chart to visualize the distribution of the month of the fires. This would help you to see if there are any particular months that are more likely to have fires.

we could use a heatmap to visualize the correlation between all of the variables in your dataset. This would help you to see which variables are most strongly correlated with each other.