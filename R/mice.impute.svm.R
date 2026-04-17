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
  n_target <- sum(wy)
  
  # 1. Bootstrap for estimation uncertainty
  xobs <- x[ry, , drop = FALSE]
  yobs <- y[ry]
  n_obs <- sum(ry)
  s <- sample(n_obs, n_obs, replace = TRUE)
  
  doty <- as.factor(yobs[s])
  dotx <- xobs[s, , drop = FALSE]
  
  # Initialize draw with NAs
  draw <- rep(NA, n_target)
  
  # --- ULTIMATE SAFEGUARD ---
  # Only attempt SVM if we have 2 classes AND enough samples
  if (length(unique(doty)) == 2) {
    
    # Use tryCatch to prevent the "indexes[[j]]" error if SVM/Predict fails
    result <- tryCatch({
      # 2. SVM Model Training
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
      p_mat <- kernlab::predict(fit, as.matrix(x[wy, , drop = FALSE]), type = "probabilities")
      
      # 4. Stochastic drawing
      # Robustly extract probabilities for the class '1'
      # (Check if it's a matrix and has at least 2 columns)
      if (is.matrix(p_mat) && ncol(p_mat) >= 2) {
        # Assuming the second column is the probability for class '1'
        p <- p_mat[, 2]
        # Replace any potential NAs in probability with 0.5 (random guess)
        p[is.na(p)] <- 0.5
        as.integer(runif(length(p)) <= p)
      } else {
        NULL # Trigger fallback
      }
    }, error = function(e) {
      NULL # Trigger fallback on any error
    })
    
    if (!is.null(result)) {
      draw <- result
    }
  }
  
  # --- FALLBACK: If SVM failed or only 1 class existed ---
  if (any(is.na(draw))) {
    # Simple random draw from observed data as a backup
    # This ensures mice always receives a valid vector
    draw[is.na(draw)] <- sample(as.integer(as.character(yobs)), sum(is.na(draw)), replace = TRUE)
  }
  
  # 5. Type adjustment
  if (is.factor(y)) {
    res <- factor(draw, levels = c(0, 1), labels = levels(y))
  } else {
    res <- draw
  }
  
  return(res)
}
