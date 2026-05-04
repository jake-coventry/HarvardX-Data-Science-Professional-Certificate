
############################################################
# HarvardX - Data Science Capstone - PH125.9x
# MovieLens Report
# April 15, 2026 - December 16, 2026 Session
# Jake Coventry
############################################################

# This is boiler plate code provided by the HarvardX PH125.9x course to ensure consistency of data

##########################################################
# Create edx and final_holdout_test sets
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

options(timeout = 120)


dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)


colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# load additional libraries required for analysis
library(dplyr)
library(knitr)
library(ggplot2)


# -----------------------------
# Data source
# -----------------------------

# Check for missing values in each column of the edx training dataset.
colSums(is.na(edx))

# Create a compact summary of the data type for each variable in edx.
data_types <- data.frame(
  Datatype = sapply(edx, function(x) class(x)[1])
)

# -----------------------------
# Summary statistics
# -----------------------------

# Calculate the number of unique users and movies in the dataset.
edx %>% 
  summarize(
    Unique_users=n_distinct(userId),
    Unique_movies=n_distinct(movieId),
    Total_ratings=n()
  )


# -----------------------------
# Ratings distribution
# -----------------------------
summary(edx$rating) # Summarise the distribution of ratings.
sort(unique(edx$rating)) # Sort so they are in numerical order when presenting in markdown

# Plot the distribution of ratings.
edx %>%
  ggplot(aes(x = rating)) +
  geom_histogram( 
    binwidth = 0.5, # matches ratings
    color = "black"
  ) +
  labs(
    x = "Rating",
    y = "Count"
  ) +
  theme_classic() +
  scale_x_continuous(breaks = seq(0.5, 5, 0.5)) # aligning x axis to rating values

# -----------------------------
# Ratings distribution
# -----------------------------

# Calculate the number of ratings per movie.
ratings_per_movie <- edx %>%
  count(movieId, title, sort = TRUE)

# summarise the distribution of ratings per movie.
summary(ratings_per_movie$n)

# Plot the distribution of the number of ratings per movie (original scale).
ratings_per_movie %>%
  ggplot(aes(x = n)) +
  geom_histogram(
    bins = 50,
    fill = "peachpuff",
    color = "black"
  ) +
  labs(
    title = "Ratings per Movie",
    x = "Number of Ratings",
    y = "Number of Movies"
  ) +
  theme_classic() 


# Re-plot the distribution using a logarithmic scale on the x-axis.
ratings_per_movie %>%
  ggplot(aes(x = n)) +
  geom_histogram(
    bins = 50,
    fill = "peachpuff",
    color = "black"
  ) +
  scale_x_log10() +
  labs(
    title = "Ratings per Movie (Log Scale)",
    x = "Number of Ratings (log scale)",
    y = "Number of Movies"
  ) +
  theme_classic()

# -----------------------------
# Ratings per user
# -----------------------------

# Calculate the number of ratings per user.
ratings_per_user <- edx %>%
  count(userId, sort = TRUE)

# summarise the distribution of ratings per user.
summary(ratings_per_user$n)

# Plot the distribution of the number of ratings per user using a log scale.

ratings_per_user %>%
  ggplot(aes(x = n)) +
  geom_histogram(
    bins = 50,
    fill = "peachpuff",
    color = "black"
  ) +
  scale_x_log10() +
  labs(
    title = "Ratings per User (Log Scale)",
    x = "Number of Ratings (log scale)",
    y = "Number of Users"
  ) +
  theme_classic()

# -----------------------------
# Feature engineering - Genre
# -----------------------------

edx <- edx %>%
  mutate(main_genre = str_split_fixed(genres, "\\|", 2)[, 1]) # Using reg ex to split genres

# Calculate summary statistics for each genre.
genre_summary <- edx %>%
  group_by(main_genre) %>%
  summarize(
    avg_rating = mean(rating),
    sd_rating = sd(rating),
    n = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(avg_rating))

# -----------------------------
# Feature engineering - Release year
# -----------------------------

# Extract the release year from the movie title.
edx <- edx %>%
  mutate(
    year = str_extract(title, "\\([0-9]{4}\\)") %>% # using reg ex to extract release year
      str_remove_all("\\(|\\)") %>%
      as.numeric()
  )

# summarise the distribution of release years.
summary(edx$year)

# Create one row per movie, keeping the release year
movies_unique <- edx %>%
  distinct(movieId, title, year) %>%
  filter(!is.na(year))

# Count unique movies per release year
movies_per_year <- movies_unique %>%
  count(year, name = "number_of_movies")


# Plot the distribution of movies by release year.
ggplot(movies_per_year, aes(x = year, y = number_of_movies)) +
  geom_col(
    fill = "peachpuff",
    color = "black"
  ) +
  labs(
    title = "Distribution of Movies by Release Year",
    x = "Release Year",
    y = "Number of Movies"
  ) +
  theme_classic() +
  scale_x_continuous(breaks = seq(1920, 2010, 20))

# extracting ratings per year
ratings_by_year <- edx %>%
  filter(!is.na(year)) %>%
  group_by(year) %>%
  summarize(
    n = n(),
    avg_rating = mean(rating),
    .groups = "drop"
  )


# combining movies per year and ratings per year so that they are both in one dataframe
combined <- movies_per_year %>%
  left_join(ratings_by_year, by = "year")


# creating a scaling factor so that they can be plotted on the same graph
scale_factor <- max(combined$n, na.rm = TRUE) / 
  max(combined$number_of_movies, na.rm = TRUE)


ggplot(combined, aes(x = year)) +
  # ratings 
  geom_line(aes(y = n, color = "Ratings"), linewidth = 1) +
  
  # movies (scaled to match)
  geom_line(aes(y = number_of_movies * scale_factor, color = "Movies"), linewidth = 1) +
  
  # plot number of ratings (primary y-axis)  
  scale_y_continuous(
    name = "Number of Ratings",
    labels = scales::label_number(scale_cut = scales::cut_short_scale()),
    sec.axis = sec_axis(
      ~ . / scale_factor,
      name = "Number of Movies"
    )
  ) +
  
  # manually set line colours
  scale_color_manual(
    values = c(
      "Movies" = "peachpuff",
      "Ratings" = "black"
    )
  ) +
  
  # customising labels and title
  labs(
    title = "Movies Released vs Ratings Volume by Year",
    x = "Release Year",
    color = NULL
  ) +
  
  scale_x_continuous(breaks = seq(1920, 2010, 20)) +
  theme_classic()


# Plot the average rating by release year.
ratings_by_year %>%
  filter(n >= 1000) %>%
  ggplot(aes(x = year, y = avg_rating)) +
  geom_line(color="black") +
  geom_smooth(se = FALSE, color = "peachpuff") +
  labs(
    title = "Average Rating by Release Year",
    x = "Release Year",
    y = "Average Rating"
  ) +
  theme_classic()

# -----------------------------
# Modeling - Creating training and validation datasets

# -----------------------------

# Create a reproducible train/validation split from the edx dataset.
set.seed(2, sample.kind = "Rounding")

# Select 10% of edx as a validation set while preserving the rating distribution.
test_index <- createDataPartition(
  y = edx$rating,
  times = 1,
  p = 0.1,
  list = FALSE
)

# Use 90% of edx for training and 10% for validation.
train_set <- edx[-test_index, ]
validation_set <- edx[test_index, ]

# Ensure all users and movies in validation are in training
validation_set <- validation_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add any removed validation rows back into the training set 
# so that no valid observations are discarded from model development.
removed <- anti_join(edx[test_index, ], validation_set)
train_set <- rbind(train_set, removed)

# Remove temporary objects to keep the workspace clean.
rm(test_index, removed)

# -----------------------------
# Global average model
# -----------------------------

mu <- mean(train_set$rating)

# Predict the same value for every observation in validation set
baseline_preds <- rep(mu, nrow(validation_set))

# Compute RMSE
baseline_rmse <- RMSE(validation_set$rating, baseline_preds)


# -----------------------------
# Movie Effect Model
# -----------------------------

# estimate the movie-specific effect (b_i). This measures how much each movie's
# average rating deviates from the global mean (mu).

movie_effects <- train_set %>%
  group_by(movieId) %>%
  summarize(
    b_i = mean(rating - mu),
    .groups = "drop"
  )

# generate predictions on the validation set.
pred_movie <- validation_set %>%
  left_join(movie_effects, by = "movieId") %>%
  mutate(
    b_i = ifelse(is.na(b_i), 0, b_i), # join the movie effects and compute
    # predicted ratings as prediction = global mean + movie effect
    pred = mu + b_i
  )

# compute RMSE to evaluate model performance on unseen (validation) data.
rmse_movie <- RMSE(validation_set$rating, pred_movie$pred)


# Create a summary table comparing RMSE values for the baseline
model_comparison <- tibble(
  model = c("Baseline", "Movie Effect"),
  RMSE = c(baseline_rmse, rmse_movie)
)


# -----------------------------
# User Effect Model
# -----------------------------

# Estimate the user-specific effect (b_u).
user_effects <- train_set %>%
  left_join(movie_effects, by = "movieId") %>%
  group_by(userId) %>%
  summarize(
    b_u = mean(rating - mu - b_i),
    .groups = "drop"
  )

# Generate predictions on the validation set.
pred_user <- validation_set %>%
  left_join(movie_effects, by = "movieId") %>%
  left_join(user_effects, by = "userId") %>%
  mutate(
    b_i = ifelse(is.na(b_i), 0, b_i), # handle unseen movies
    b_u = ifelse(is.na(b_u), 0, b_u), # handle unseen users
    pred = mu + b_i + b_u
  )

# Compute RMSE to evaluate performance.
rmse_user <- RMSE(validation_set$rating, pred_user$pred)

# Create a table comparing RMSE across models.
model_comparison <- tibble(
  model = c("Baseline", "Movie Effect", "Movie + User Effect"),
  RMSE = c(baseline_rmse, rmse_movie, rmse_user)
)


# -----------------------------
# Regularisation tuning
# -----------------------------

# Define a sequence of lambda values to test.
lambdas <- seq(0, 10, 0.5)
lambdas

# Compute RMSE for each lambda value.
rmse_vals <- sapply(lambdas, function(l){
  
  # Compute global mean rating
  mu <- mean(train_set$rating)
  
  # summarise the movie effect
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(
      b_i = sum(rating - mu) / (n() + l),
      .groups = "drop"
    )
  
  # summarise the user effect
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(
      b_u = sum(rating - mu - b_i) / (n() + l),
      .groups = "drop"
    )
  
  # Predictions on validation
  preds <- validation_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(
      b_i = ifelse(is.na(b_i), 0, b_i),
      b_u = ifelse(is.na(b_u), 0, b_u),
      pred = mu + b_i + b_u
    ) %>%
    pull(pred)
  # Return RMSE for this lambda value. 
  RMSE(validation_set$rating, preds)
})

# Store results for plotting and selection of optimal lambda.
results <- tibble(
  lambda = lambdas,
  rmse = rmse_vals
)

# Plot RMSE as a function of lambda.
results %>%
  ggplot(aes(lambda, rmse)) +
  geom_line(color = "Blue") +
  geom_point() +
  labs(
    title = "Regularisation tuning",
    x = "Lambda",
    y = "RMSE"
  ) +
  theme_classic()

# -----------------------------
# Select Optimal Lambda
# -----------------------------

# Identify the lambda value that minimizes validation RMSE.
best_lambda <- results %>%
  slice_min(rmse) %>%
  pull(lambda)

# Extract the corresponding minimum RMSE.
best_rmse <- min(results$rmse)

# Display the optimal lambda and its RMSE.
best_lambda
best_rmse

# Create a table comparing RMSE across models, including the regularised model.
model_comparison <- tibble(
  model = c(
    "Baseline",
    "Movie Effect",
    "Movie + User Effect",
    "Regularised Model"
  ),
  RMSE = c(
    baseline_rmse,
    rmse_movie,
    rmse_user,
    best_rmse
  )
)

# -----------------------------
# Feature-Enhanced Model
# -----------------------------

# Add engineered features to both training and validation sets.
train_set <- train_set %>%
  mutate(
    genre = str_split_fixed(genres, "\\|", 2)[,1],
    year = str_extract(title, "\\([0-9]{4}\\)") %>%
      str_remove_all("\\(|\\)") %>%
      as.numeric()
  )

validation_set <- validation_set %>%
  mutate(
    genre = str_split_fixed(genres, "\\|", 2)[,1],
    year = str_extract(title, "\\([0-9]{4}\\)") %>%
      str_remove_all("\\(|\\)") %>%
      as.numeric()
  )


# Use the lambda value selected during validation tuning.
lambda <- best_lambda

# Recalculate the global mean using the training set.
mu <- mean(train_set$rating)

# Movie effect
b_i <- train_set %>%
  group_by(movieId) %>%
  summarize(
    b_i = sum(rating - mu)/(n() + lambda),
    .groups = "drop"
  )

# User effect
b_u <- train_set %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(
    b_u = sum(rating - mu - b_i)/(n() + lambda),
    .groups = "drop"
  )

# Genre effect
b_g <- train_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genre) %>%
  summarize(
    b_g = sum(rating - mu - b_i - b_u)/(n() + lambda),
    .groups = "drop"
  )

# Year effect
b_y <- train_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genre") %>%
  group_by(year) %>%
  summarize(
    b_y = sum(rating - mu - b_i - b_u - b_g)/(n() + lambda),
    .groups = "drop"
  )

# Generate validation predictions using all learned effects.
pred_final <- validation_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genre") %>%
  left_join(b_y, by = "year") %>%
  mutate(
    across(starts_with("b_"), ~ifelse(is.na(.x), 0, .x)),
    pred = mu + b_i + b_u + b_g + b_y
  )

# Evaluate the feature-enhanced model on the validation set.
rmse_final <- RMSE(validation_set$rating, pred_final$pred)


# Summarize RMSE across all models to show incremental performance improvements.
final_results <- tibble(
  model = c(
    "Baseline",
    "Movie Effect",
    "Movie + User Effect",
    "Regularised Model",
    "Final Model (with features)"
  ),
  RMSE = c(
    baseline_rmse,
    rmse_movie,
    rmse_user,
    best_rmse,
    rmse_final
  )
)

# -----------------------------
# Final model evaluation on final hold-out test set
# -----------------------------

# Use the lambda selected during validation tuning.
lambda <- best_lambda

# Add engineered features to edx.
edx <- edx %>%
  mutate(
    genre = str_split_fixed(genres, "\\|", 2)[,1],
    year = str_extract(title, "\\([0-9]{4}\\)") %>%
      str_remove_all("\\(|\\)") %>%
      as.numeric()
  )

# Add the same engineered features to the final hold-out test set.
final_holdout_test <- final_holdout_test %>%
  mutate(
    genre = str_split_fixed(genres, "\\|", 2)[,1],
    year = str_extract(title, "\\([0-9]{4}\\)") %>%
      str_remove_all("\\(|\\)") %>%
      as.numeric()
  )

# Calculate the global mean using the full edx dataset.
mu <-mean(edx$rating)

# Movie effect
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(
    b_i = sum(rating - mu)/(n() + lambda),
    .groups = "drop"
  )

# User effect
b_u <- edx %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(
    b_u = sum(rating - mu - b_i)/(n() + lambda),
    .groups = "drop"
  )

# Genre effect
b_g <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genre) %>%
  summarize(
    b_g = sum(rating - mu - b_i - b_u)/(n() + lambda),
    .groups = "drop"
  )

# Year effect
b_y <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genre") %>%
  group_by(year) %>%
  summarize(
    b_y = sum(rating - mu - b_i - b_u - b_g)/(n() + lambda),
    .groups = "drop"
  )

# Generate predictions for the final hold-out test set.
final_preds <- final_holdout_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genre") %>%
  left_join(b_y, by = "year") %>%
  mutate(
    # Missing effects are replaced with zero so unavailable effects 
    # do not adjust the prediction.
    across(starts_with("b_"), ~ifelse(is.na(.x), 0, .x)),
    pred = mu + b_i + b_u + b_g + b_y
  )

# Compute the final RMSE.
final_rmse <- RMSE(final_holdout_test$rating, final_preds$pred)


# Store final predicted ratings for the final hold-out test set
predicted_ratings <- final_preds %>%
  select(userId, movieId, rating, pred)

# Display first few predictions
head(predicted_ratings)

# Print final RMSE
print(paste("Final RMSE:", final_rmse))
