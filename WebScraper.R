library(XML)
library(stringr)
Time = matrix()
# build the URL
#thepage = readLines("http://www.espn.com/nfl/playbyplay?gameId=401030775")
grep('T.Taylor',thepage)

mypattern = '\t\t\t\t\t\t\t([:digit:]* - 1st)'
datalines = grep(mypattern,thepage,value=TRUE)


