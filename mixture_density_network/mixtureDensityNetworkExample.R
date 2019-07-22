#http://mikedusenberry.com/mixture-density-networks
#Bishop: Pattern Recognition and Machine Learning
#https://cran.r-project.org/web/packages/condmixt/condmixt.pdf

rm(list = ls()) #Clear environment variables
dev.off() #Close plots
cat("\014") #Clear console

library(condmixt)

N <- 2000 #number of simulated examples

x <- matrix(runif(N), nrow=1) #generate random vector

t <- x + 0.3*sin(2*pi*x) + runif(N, min = -0.1, max = 0.1) #compute function with noise

h <- 10 #number of nodes in hidden layer
m <- 30 #number of mixture components

#find the parameters of the neural network
theta <- condgaussmixt.train(h= h, m = m,x = t,y = as.vector(x), nstart = 1)

tsim <- matrix(runif(N), nrow=1) #generate random vector for test

#from the trained neural network, compute the feedforward for mixture parameters for each test point
params <- condgaussmixt.fwd(theta=theta, h=h, m=m, x=tsim)

xsim <- vector(mode="numeric", length=N)
musim <- vector(mode="numeric", length=N)
#simulate and plot points from the computed network
for (i in 1:N) {
  j <- sum(cumsum(params[,1,i]) < runif(1)) + 1 #sample a component from the weighting distribution
  jmu <- which.max(params[,1,i]) #most likely Gaussian component

  pie <- params[j, 1, i] #component weighting
  mu <- params[j, 2, i] #component mean
  sig <- params[j, 3, i] #component std deviation
  
  musim[i] <- params[jmu, 2, i]
  xsim[i] <- rnorm(1, mean=mu, sd=sig)
}

#plot actual data
plot(t, x)
#plot simulated data from learned mixture density network
points(tsim, xsim, col = "red")
#plot mean of most likely Gaussian component from learned mixture density network
points(tsim, musim, col = "blue")