#' Imputation by the Support Vector Machine (mice-SVM)
#'
#' This function performs proper multiple imputation using the Support Vector Machine (SVM) 
#' combined with bootstrapping, as proposed by Takahashi (2026). It is specifically 
#' designed for binary variables in high-dimensional data settings.
#'
#' @inheritParams mice.impute.pmm
#' @param type A vector of length \code{ncol(x)} identifying the predictors. 
#' Captured here to avoid conflicts with the SVM 'type' argument from mice's internal calls.
#' @param C Cost of constraints violation (default = 1).
#' @param scaled A logical vector indicating the variables to be scaled.
#' @param kernel The kernel function used in training and predicting (default = "vanilladot").
#' @param ... Other named arguments to be passed to \code{kernlab::ksvm()}.
#' @return A vector of length \code{sum(!ry)} with imputed values.
#' @references 
#' Takahashi, M. (2026). Multiple Imputation based on the Support Vector Machine for 
#' High-Dimensional Data with General Missing Patterns in Causal Inference.
#' @export
mice.impute.svm <- function(y, ry, x, wy = NULL, type = NULL, C = 1, scaled = TRUE, kernel = "vanilladot", ...) {
  
  if (!requireNamespace("kernlab", quietly = TRUE)) {
    stop("Package 'kernlab' is needed for this function. Please install it.")
  }
  
  if (is.null(wy)) wy <- !ry
  
  # 1. Bootstrap for estimation uncertainty
  xobs <- x[ry, , drop = FALSE]
  yobs <- y[ry]
  n1 <- sum(ry)
  s <- sample(n1, n1, replace = TRUE)
  
  dotx <- xobs[s, , drop = FALSE]
  doty <- as.factor(yobs[s])
  
  # 2. SVM Model Training
  # By forcing doty to be a factor above, ksvm correctly enters classification mode.
  fit <- kernlab::ksvm(
    x = as.matrix(dotx),
    y = doty,
    type = "C-svc", 
    kernel = kernel,
    C = C,
    scaled = scaled,
    prob.model = TRUE, 
    ...
  )
  
  # 3. Predict probabilities for fundamental uncertainty
  # Column 2 usually corresponds to the second level of the factor
  p <- kernlab::predict(fit, as.matrix(x[wy, , drop = FALSE]), type = "probabilities")[, 2]
  
  # 4. Stochastic drawing
  draw <- as.integer(runif(length(p)) <= p)
  
  # 5. Type adjustment to match the original 'y'
  if (is.factor(y)) {
    # If original y was a factor, restore its levels and labels
    res <- factor(draw, levels = c(0, 1), labels = levels(y))
  } else {
    # If original y was numeric 0/1, return as integer/numeric
    res <- draw
  }
  
  return(res)
}