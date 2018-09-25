# Daubechies Image (DWT)
data(dau)
par(mfrow=c(1,1), pty="s")
image(dau, col=rainbow(128))
sum(dau^2)
dau.dwt <- dwt.2d(dau, "d4", 3)
plot.dwt.2d(dau.dwt)
sum(plot.dwt.2d(dau.dwt, plot=FALSE)^2)

## Xbox image (Reconstruction using wavelets)
data(xbox)
xbox.dwt <- dwt.2d(xbox, "haar", 3)
par(mfrow=c(1,1), pty="s")
plot.dwt.2d(xbox.dwt)
par(mfrow=c(2,2), pty="s")
image(1:dim(xbox)[1], 1:dim(xbox)[2], xbox, xlab="", ylab="",
      main="Original Image")
image(1:dim(xbox)[1], 1:dim(xbox)[2], idwt.2d(xbox.dwt), xlab="", ylab="",
      main="Wavelet Reconstruction")
image(1:dim(xbox)[1], 1:dim(xbox)[2], xbox - idwt.2d(xbox.dwt),
      xlab="", ylab="", main="Difference")

## Boat import 

#Boat from the internet
library(imager)
im <- load.image("I:/My Documents/Data Files/satelite_boat.jpg")
thmb <-resize(im, 512,512)

#Boat Reconstruction
boat.dwt <- dwt.2d(thmb[,,,2], "haar", 3)
par(mfrow=c(1,1), pty="s")
plot.dwt.2d(boat.dwt)
par(mfrow=c(2,2), pty="s")
image(1:dim(thmb[,,,2])[1], 1:dim(thmb[,,,2])[2], thmb[,,,2], xlab="", ylab="",
      main="Original Image")
image(1:dim(thmb[,,,2])[1], 1:dim(thmb[,,,2])[2], idwt.2d(boat.dwt), xlab="", ylab="",
      main="Wavelet Reconstruction")
image(1:dim(thmb[,,,2])[1], 1:dim(thmb[,,,2])[2], thmb[,,,2] - idwt.2d(boat.dwt),
      xlab="", ylab="", main="Difference")


# Boat DWT
image(thmb[,,,2],col=rainbow(128))
boat.dwt <- dwt.2d(thmb, "haar", 3)
plot.dwt.2d(boat.dwt)
