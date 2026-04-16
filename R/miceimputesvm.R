#' Imputation by the Support Vector Machine
#'
#' Imputes binary variables using the Support Vector Machine (SVM) with a 
#' bootstrap step to ensure proper multiple imputation.
#'
#' @inheritParams mice.impute.pmm
#' @param C Cost of constraints violation (default = 1).
#' @param scaled A logical vector indicating the variables to be scaled.
#' @param kernel The kernel function used in training and predicting (default = "vanilladot").
#' @param ... Other named arguments to be passed to \code{kernlab::ksvm()}.
#'
#' @return Vector with imputed values, same length as \code{sum(wy)}
#'
#' @details
#' The method combines the bootstrap and SVM to generate multiple imputation 
#' that is proper. It is particularly useful for high-dimensional data.
#'
#' @references
#' Takahashi, M. (2026). Multiple Imputation based on the Support Vector Machine 
#' for High-Dimensional Data with General Missing Patterns in Causal Inference. 
#' Journal of Statistical Computation and Simulation.
#'
#' @author Masayoshi Takahashi
#' @family univariate imputation functions
#' @keywords datagen
#' @export
#' @importFrom stats predict runif
mice.impute.svm <- function(y, ry, x, wy = NULL, C = 1, scaled = TRUE, kernel = "vanilladot", ...) {
  
  # 1. Check for dependencies
  if (!requireNamespace("kernlab", quietly = TRUE)) {
    stop("Package 'kernlab' is needed for this function to work. Please install it.", call. = FALSE)
  }

  if (is.null(wy)) {
    wy <- !ry
  }

  # 2. Bootstrap for proper multiple imputation
  # Resampling from observed data (yobs, xobs)
  n1 <- sum(ry)
  s <- sample(n1, n1, replace = TRUE)
  
  xobs <- x[ry, , drop = FALSE]
  yobs <- y[ry]
  
  dotx <- xobs[s, , drop = FALSE]
  doty <- yobs[s]

  # 3. Training SVM model (using matrix interface)
  # type = "C-svc" for classification. prob.model = TRUE allows us to compute probabilities.
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

  # 4. Compute predicted probabilities for missing values (wy)
  # We assume that the probabilities for category [1] (or second level) are stored in column 2.
  p <- predict(fit, as.matrix(x[wy, , drop = FALSE]), type = "probabilities")[, 2]

  # 5. Sampling 0 or 1 based on the probabilities
  # Standard stochastic imputation using runif
  draw <- as.integer(runif(length(p)) <= p)

  # 6. Post-processing according to the type (such as factor) of the original y
  if (is.factor(y)) {
    # Transform to the factor type keeping the level of the original y
    res <- factor(draw, levels = c(0, 1), labels = levels(y))
  } else {
    res <- draw
  }

  return(res)
}