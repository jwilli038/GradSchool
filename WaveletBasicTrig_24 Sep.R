# Basic wavelet transform with trig function 
# See R guide https://cran.r-project.org/web/packages/waveslim/waveslim.pdf
# Create the x values, and true y values must be diatic (divisible by 2^J) 
x = seq(from=0,to=1,1/511)
y = 2*sin(x) + 2*cos(x)^2
plot(x,y,col=5)

# Create the signal with specified noise
y_r = y + rnorm(512,0,sd=.01)
#lines(x,y_r,col=1)
#Perform Discrete Wavelet Transform on noise (DWT)
#This is obtained to do transformation on the 4 different levels of the transform
# d1-d4 and s4
dwt(y_r) 

#Transformation Step (performs wavelet shrinkage and level specification, hard vs soft)
#universal.thresh(wc, max.level = 4, hard = TRUE)
lines(x,idwt(universal.thresh(dwt(y_r), max.level=4, hard = TRUE)),col=2,lwd=2)
lines(x,idwt(universal.thresh(dwt(y_r), max.level=4, hard = FALSE)),col=3,lwd=2)
#Original signal
#lines(x,y_r)
