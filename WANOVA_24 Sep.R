# WANOVA example

#Vector of length 1024
x = seq(from=0,to=2*pi,2*pi/1023)

#Create Bias to test wanova
bias = rep(0,1024)

#Bias interval 1/4 of data
bias[256:512] = .75

#System with no error or bias 
y_sys = 2*sin(x) + 2*cos(x)^2

#Model with error and interval bias
y_model <- y_sys + rnorm(1024,0,sd=sqrt(2))+bias
plot(x,y,'l')
lines(x,y_model,'l')

#Calculate WANOVA of Differences
#lines(x,idwt(dwt(y_sys-y_model)),max.level=4, hard = TRUE)),col=2,lwd=2)
lines(x,idwt(universal.thresh(dwt((y_sys-y_model)), max.level=4, hard = TRUE)),col=2,lwd=2)
W.dwt = universal.thresh(dwt(y_sys-y_model),max.level=4, hard = TRUE)

#Calculate the WANOVA Stat 
kappa = sum(idwt(universal.thresh(dwt((y_sys-y_model)), max.level=4, hard = TRUE))^2)/.1

#Calculate the WANOVA Test Stat (kappa-eta) #Need to figur out the test stat
#kE = dchisq(1, df = 1:4)

#Bisect the data set and perform WANOVA
kappaB1 = sum(idwt(universal.thresh(dwt((y_sys-y_model)[0:512]), max.level=4, hard = TRUE))^2)/.1
kappaB2 = sum(idwt(universal.thresh(dwt((y_sys-y_model)[513:1024]), max.level=4, hard = TRUE))^2)/.1
kappaB11 = sum(idwt(universal.thresh(dwt((y_sys-y_model)[0:256]), max.level=4, hard = TRUE))^2)/.1
kappaB12 = sum(idwt(universal.thresh(dwt((y_sys-y_model)[257:512]), max.level=4, hard = TRUE))^2)/.1



#Shows the detail in course levels 
#par(mfrow=c(3,2), pty="s")
#plot(W.dwt$d1,type='l')
#plot(W.dwt$d2,type='l')
#plot(W.dwt$d3,type='l')
#plot(W.dwt$d4,type='l')
#plot(W.dwt$s4,type='l')


