# Pandas and numpy - pair-up

With a partner of your choosing complete the following in a jupyter notebook.

1. Read the following data into a pandas data frame
 ` ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_mm_mlo.txt`
 
2. Select columns 0,1,3 
`[[0, 1, 3]]`

3. Use a for loop to find all rows where 
Co2 (column 3) enteries with the value -99.99 (these are missing values) and replace them with NaN values (try using np.nan - do you know what it is? )

4. Change names of columns to year, month, and CO2 (use colnames)

5. Add a column 'Day' and specifiy the day 15 for all enteries

6. Add a date column according to the 'year', 'month' and 'day' columns (options: use apply with lambda or for loop together with datetime.date (make sure to import it)) 

7. Drop the 'Day' column
