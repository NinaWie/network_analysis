#' print a summary object from a Siena object
#'
#' @param siena.fit an object with a SAOM estimated model using 
#'   the package \code{RSiena}
#'
#' @return a \code{data.frame} with the estimated coefficients and 
#'   standard errors with a p-value using the z-statistic
#' @noRd
#'
#' @examples
printSiena <- function(siena.fit) {
  options(scipen = 99)

  temp.rates <- data.frame(
    dependent = "rate", effect = attributes(siena.fit$f)$condEffects$effectName,
    theta = round(siena.fit$rate, 3), s.e. = round(siena.fit$vrate, 3), p.value = 0,
    sig. = 0, t.conv = 0
  )

  # Computing p-values and assigning symbols
  siena.fit$pvalues <- 2 * (1 - pnorm(abs(siena.fit$theta / sqrt(diag(siena.fit$covtheta)))))
  stars <- ifelse(siena.fit$pvalues <= 0.10, ifelse(siena.fit$pvalues <= 0.05,
    ifelse(siena.fit$pvalues <= 0.01,
      ifelse(siena.fit$pvalues <= 0.001, "***", "**"),
      "*"
    ), "."
  ), "")
  effects <- siena.fit$requestedEffects

  # Create a data frame with the outcome
  temp.effects <- data.frame(
    dependent = effects[, c("name")], effect = effects[, c("effectName")],
    theta = round(siena.fit$theta, 3), s.e. = round(sqrt(diag(siena.fit$covtheta)), 3),
    p.value = round(siena.fit$pvalues, 3), sig. = stars,
    t.conv = round(siena.fit$tconv, 3)
  )

  siena.results.temp <- rbind(temp.rates, temp.effects)

  siena.results.temp[siena.results.temp$dependent == "rate", c("p.value", "sig.", "t.conv")] <- ""

  return(siena.results.temp)
}


#' print coefficients from a Siena object
#'
#' @param siena.fit an object with a SAOM estimated model using 
#'   the package \code{RSiena}
#' @param nperiods \code{int} number of periods to print
#'
#' @return
#' @noRd
#'
#' @examples
printSienaCoev <- function(siena.fit, nperiods) {
  options(scipen = 99)

  pvalues <- 2 * (1 - pnorm(abs(siena.fit$theta / sqrt(diag(siena.fit$covtheta)))))
  stars <- ifelse(pvalues <= 0.10, ifelse(pvalues <= 0.05,
    ifelse(pvalues <= 0.01,
      ifelse(pvalues <= 0.001, "***", "**"),
      "*"
    ), "."
  ), "")

  res <- data.frame(
    effect = c(siena.fit$f$basicEffects[[1]][, 2], siena.fit$f$basicEffects[[2]][, 2]),
    theta = round(siena.fit$theta, 3),
    s.e. = round(sqrt(diag(siena.fit$covtheta)), 3), p.value = round(pvalues, 3),
    sig. = stars, t.conv = round(siena.fit$tconv, 3)
  )

  res[grepl("rate", res$effect), c("p.value", "sig.", "t.conv")] <- ""

  return(res)
}
