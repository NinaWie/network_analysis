library(sna)
library(network)
library('here')
setwd(here())

run_qap <- function (net, attr, distances, userID, timeBin, base_path) {
  print(sprintf("Run QAP user %s time step %s", userID, as.character(timeBin)))
  # Multiple QAP regression:
  # H1: Transitions are more likely between locations that are close to another
  # H2: Locations that are close to home are more popular
  # H3: Transitions between locations with the same purpose are more likely
  
  # prepare the data: 
  # H1 is simply the matrix "distances"
  # H2 is dist_from_home_receiver:
  dist_from_home <- attr$dist_from_home
  num_rows = length(dist_from_home)
  dist_from_home_receiver <- matrix(dist_from_home,num_rows,num_rows,byrow=TRUE) # H2
  # H3 is purpose
  purpose <- attr$purpose
  samePurpose <- outer(purpose,purpose,"==")*1
  
  model_qap <- netlm(net, list(distances, dist_from_home_receiver, samePurpose), rep=5000, nullhyp="qapspp")
  model_qap$names <- c("inter", "distances", "from_home_in", "samePurpose")
  print(model_qap)
  
  # save outputs:
  # save_outp = c(model_qap[1], model_qap[10], model_qap[11], model_qap[12])
  # write.csv(save_outp, sprintf("%s/%s/qap_fitted_%s.csv", base_path, userID, timeBin))
  res <- summary(model_qap)
  expRes <- cbind(res$coefficients, exp(res$coefficients), res$pleeq, res$pgreq, res$pgreqabs)
  colnames(expRes) <- c("ESt.", "exp(Est.)", "p_lower", "p_higher", "p-value")
  rownames(expRes) <- res$names
  write.csv(expRes, sprintf("%s/%s/qap_fitted_%s.csv", base_path, userID, timeBin))
}

plot_pvalues <- function(model_qap, permutations) {
  z.values <- rbind(model_qap$dist,model_qap$tstat)
  p.values <- function(x,permutations){
    sum(abs(x[1:permutations]) > abs(x[permutations+1]))/permutations}
  empirical.p.values <- apply(z.values,2,p.values,permutations)
  empirical.p.values
  
  par(mfrow=c(2,3))
  for (i in 1:4)
  {
    jpeg(sprintf('report/p_value_permutation_%s.jpg', i))
    hist(model_qap$dist[,i],breaks=30,xlim=c(min(c(model_qap$tstat[i],model_qap$dist[,i]))-1,
                                             max(c(model_qap$tstat[i],model_qap$dist[,i]))+1),
         main=model_qap$names[i],xlab="z-values")
    abline(v=model_qap$tstat[i],col="red",lwd=3,lty=2)
    dev.off()
  }
}

set.seed(42)


#### Iterate over all users, do QAP for all graphs
base_path <- "data/foursquare_120"
possible_users = list.files(base_path)

for (i in 1:length(possible_users)) {
  
  # Load stable attributes
  user_id <- possible_users[i]
  attr <- read.csv(sprintf("%s/%s/project_attr.csv", base_path, user_id), header = T)
  distances <- read.csv(sprintf("%s/%s/distances.csv", base_path, user_id), header=T)
  
  ########## QAP ###########
  # Do QAP for all three time graphs
  for (j in 0:2) {
    # Load net for this time step
    net_weighted <- as.matrix(read.csv(sprintf("%s/%s/project_weighted_adj_%s.csv", base_path, user_id, j), header = T))
    # run QAP regression
    run_qap(net_weighted, attr, distances, user_id, as.character(j), base_path)
  }
  
}


### Run QAP for full graphs
# possible_users = list.files("graph_tist_all")
# for (i in 50:50) { # 1:length(possible_users)) {
#   net_weighted <- as.matrix(read.csv(sprintf("graph_tist_all/%s/project_weighted_adj_full.csv", possible_users[i]), header = T))
#   attr <- read.csv(sprintf("graph_tist_all/%s/project_attr_full.csv", possible_users[i]), header = T)
#   distances <- read.csv(sprintf("graph_tist_all/%s/distances_full.csv", possible_users[i]), header=T)
#   run_qap(net_weighted, attr, distances, possible_users[i], "full", "graph_tist_all")
# }
