rm(list=ls()) #clear environment variables
cat('\014') #clear console

# This script implements a permutation test and confidence interval for a location
# parameter

library(pracma)

perm_test2 <- function(X, m0){
  #returns the p-value of a two sided permutation test with null hypothesis
  #location parameter of m0
  
  #sample size
  n <- length(X)
  
  #sample deviations from null
  Xd <- X - m0
  
  #test statistic
  d <- sum(Xd)
  
  #binary representation {-1, +1} for signs
  S <- matrix(0, nrow=2^n, ncol=n)
  for (i in 1:(2^n)){
    j <- 1
    tmp <- i - 1
    while ((j <= n) && (tmp > 0)){
      S[i, j] <- rem(tmp, 2)
      tmp <- floor(tmp/2)
      j <- j + 1
    }
  }
  S <- fliplr(S)
  S[S == 0] <- -1
  
  #permutation distribution
  T <- matrix(nrow = 2^n)
  for (i in 1:(2^n)){
    T[i] <- sum(S[i,]*Xd)
  }
  
  #p-value
  p2 <- (sum(T >= abs(d)) + sum(T <= -abs(d)))/(2^n)
  return(p2)
  
}


conf_int <- function(X, alpha){
  #returns a 100*(1 - alpha) confidence interval for the location parameter
  #by inverting permutation tests

  m0s <- linspace(-2, 2, n=100)
  p2s = matrix(nrow=length(m0s))
  for (k in 1:length(m0s)){
    m0 <- m0s[k]
    
    p2s[k] <- perm_test2(X, m0)

  }
  
  #lower and upper bounds of confidence interval
  lb <- min(m0s[p2s >= alpha])
  ub <- max(m0s[p2s >= alpha])
  
  return(c(lb, ub))
}

#set random seed
set.seed(42)

#sample size
n <- 10
#level of significance
alpha <- 0.05
#actual location paraneter
theta <- 0

#generate random normal sample
X <- rnorm(n, mean=theta)

perm_test2(X, theta)
conf_int(X, alpha)