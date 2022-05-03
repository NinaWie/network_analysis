

library(sna)
library(network)
library(RSiena)
library(here)
setwd(here())
source("printSiena.R")

jaccard <- function(first_net, sec_net) {
  diff <- first_net-sec_net
  N11 <- sum(diff == 0 & first_net == 1)
  N10 <- sum(diff == 1)
  N01 <- sum(diff == -1)
  N00 <- sum(diff == 0 & sec_net == 0)
  
  # sanity check
  stopifnot(dim(first_net)^2 == (N11 + N10 + N01 + N00))
  
  return(N11 / (N11+ N01 + N10)) 
}

purpose_list = c("work", "shop", "leisure", "home")
get_best_match <- function(x){
  return(match(x, purpose_list))
}

# Geodesic distance
GeodesicDistribution <- function(i, data, sims, period, groupName,
                                 varName, levls = c(1:5, Inf), cumulative = TRUE) {
  x <- networkExtraction(i, data, sims, period, groupName, varName)
  require(sna)
  a <- sna::geodist(symmetrize(x))$gdist
  if (cumulative) {
    gdi <- sapply(levls, function(i) {
      sum(a <= i)
    })
  }
  else {
    gdi <- sapply(levls, function(i) {
      sum(a == i)
    })
  }
  names(gdi) <- as.character(levls)
  gdi
}

saom_get_gof <- function(saom_model, out_path) {
  ############### GOF - Indegree
  print("Start GOF")
  gof1.id <- sienaGOF(saom_model,
                      verbose = TRUE, varName = "mobility",
                      IndegreeDistribution
  )
  print(sprintf('%s/gof_id.pdf', out_path))
  pdf(sprintf('%s/gof_id.pdf', out_path))
  print(plot(gof1.id, fontsize=18))
  dev.off()
  
  ############### GOF - outdegree
  gof1.od <- sienaGOF(saom_model,
                      verbose = TRUE, varName = "mobility",
                      OutdegreeDistribution
  )
  
  pdf(sprintf('%s/gof_od.pdf', out_path))
  print(plot(gof1.od, fontsize=18))
  dev.off()
  
  ############### GOF - Geodesic
  gof1.gd <- sienaGOF(saom_model,
                      verbose = FALSE, varName = "mobility",
                      GeodesicDistribution
  )
  pdf(sprintf('%s/gof_gd.pdf', out_path))
  print(plot(gof1.gd, fontsize=18))
  dev.off()
  
  print("Saved GoF plots")
  # -------------- collect and save p vals:
  pvals <- c(gof1.id$Joint$p, gof1.od$Joint$p, gof1.gd$Joint$p)
  write.csv(pvals, sprintf('%s/saom_gof_pvals.csv', out_path))
}

run_saom <- function (net1, net2, net3, attr, dyad_distances, userID, out_path, run_gof=TRUE) {
  print("Run saom for user")
  print(userID)
  # Compiute jaccard indices
  jac1 <- jaccard(net1, net2)
  jac2 <- jaccard(net2, net3)
  
  # create siena variables
  all_arr = array(c(net1, net2, net3), dim = c(dim(net1)[1], dim(net1)[1], 3))
  mobility <- sienaDependent(all_arr)
  
  #  -------- constant covariates
  # preprocess purpose
  purpose_transformed <- apply(as.matrix(attr$purpose), 1, FUN=get_best_match)
  purpose <- coCovar(purpose_transformed)
  # distances
  dist_home <- coCovar(attr$dist_from_home)
  distance <- coDyadCovar(dyad_distances)
  
  myData <- sienaDataCreate(mobility, distance, purpose, dist_home)
  myeff <- getEffects(myData) 
  myeff <- includeEffects(myeff, X, interaction1="distance") # distances
  myeff <- includeEffects(myeff, transTrip) # transitivity
  myeff <- includeEffects(myeff, outAct) # out-degree related activity effect
  myeff <- includeEffects(myeff, egoX, altX, sameX, interaction1 = "purpose") # purpose effects
  myeff <- includeEffects(myeff, egoX, altX, interaction1 = "dist_home") # distance from home
  myeff

  # create model
  saom_alg <- sienaAlgorithmCreate(
    projname = "mobility",
    nsub = 4, n3 = 3000, seed = 1908
  )
  
  # Estimate the model
  saom_model <- siena07(saom_alg,
                        data = myData, effects = myeff, returnDeps = TRUE,
                        useCluster = TRUE, nbrNodes = 3, batch = FALSE, silent=T
  )
  print("Convergence")
  print(saom_model$tconv.max)
  output <- printSiena(saom_model)
  print(output)
  
  # add the things we want to save
  output$jac1[1] <- jac1
  output$jac2[1] <- jac2
  output$tconv_max[1] <- saom_model$tconv.max
  
  # Write csv with outputs
  write.csv(output, sprintf("%s/%s/saom_fitted_1.csv", out_path, userID)) # TODO: switch back before code submission
  
  if (run_gof) {
    saom_get_gof(saom_model, sprintf("%s/%s", out_path, userID))
  }
}


#### Iterate over all users, do saom for time graphs
base_path <- "../data/foursquare_120"
possible_users = list.files(base_path)

for (i in 1:length(possible_users)) {
  user_id <- possible_users[i]
  
  # check if already done
  if (file.exists(sprintf("%s/%s/saom_gof_pvals.csv", base_path, user_id))){
    print("Already exists")
    print(user_id)
    next
  }
  
  # Load stable attributes
  attr <- read.csv(sprintf("%s/%s/project_attr.csv", base_path, user_id), header = T)
  distances <- read.csv(sprintf("%s/%s/distances.csv", base_path, user_id), header=T)
  
  # read networks over time
  net1 <- as.matrix(read.csv(sprintf("%s/%s/project_adj_0.csv", base_path, user_id), header = T))
  net2 <- as.matrix(read.csv(sprintf("%s/%s/project_adj_1.csv", base_path, user_id), header = T))
  net3 <- as.matrix(read.csv(sprintf("%s/%s/project_adj_2.csv", base_path, user_id), header = T))
  
  # normalize (otherwise saom does not converge well)
  dist_matrix <- as.matrix(distances)
  # normed_dyad_distances <- dist_matrix / max(dist_matrix)
  
  # Run
  run_saom(net1, net2, net3, attr, dist_matrix, possible_users[i], base_path)
}
