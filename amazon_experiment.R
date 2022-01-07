# Empirical Bayes experimentation with amazon gift cards dataset

# Step 0: Loading the data

raw_df = read.csv("data/Gift_Cards.csv", header = FALSE, col.names=c("item", "user", "rating", "timestamp"))
head(raw_df)

# Step 1: Data exploration

library(tidyverse)
df <- raw_df %>% 
  select(c("item", "rating"))

counts <- count(df, item) %>%
  filter(n > 20)
ggplot(counts, aes(x=n)) + geom_histogram() + xlim(0, 500)

# pivot table for prior calculation

# train test validation split - need to split at product level

set.seed(1729)

df = merge(counts, df, by="item")
in.train = sample(unique(df$item), size = 0.6*length(unique(df$item)))
df.train = filter(df, item %in% in.train)
df.rest = filter(df, !item %in% in.train)
in.val = sample(unique(df.rest$item), size=0.5*length(unique(df.rest$item)))
df.val = filter(df.rest, item %in% in.val)
df.test = filter(df.rest, !item %in% in.val)

counts <- count(df.train, item) %>%
  filter(n > 20)
ggplot(counts, aes(x=n)) + geom_histogram() + xlim(0, 500)

# set up counts for gradient descent

library(data.table)

prior_df <- dcast(setDT(df.train), item ~ rating, fun.aggregate = length)

colnames(prior_df) <- c("item", "n1", "n2", "n3", "n4", "n5")

prior_df$n = prior_df$n1 + prior_df$n2 + prior_df$n3 + prior_df$n4 + prior_df$n5

prior_df <- filter(prior_df, n > 20)

# remove all columns with a 0 count for one category
prior_df  <- transmute(prior_df, n1 = n1 / n, n2 = n2 / n, n3 = n3 / n, n4 = n4 / n, n5 = n5 / n) %>% 
  filter(n1 != 0 & n2 != 0 & n3 != 0 & n4 != 0 & n5 != 0)

log_p <- summarize(prior_df, summarize(prior_df, n1 = sum(log(n1)), n2 = sum(log(n2)), n3 = sum(log(n3)), n4=sum(log(n4)), n5=sum(log(n5))))
log_p <- as.numeric(log_p[1,])
N <- nrow(prior_df)

alphas = c(1,1,1,1,1)

# actual gradient descent functions

# this one is currently far too slow and yet simultaneously unstable.
step <- function(alphas, log_p, N, step_size, forward=TRUE) {
  x = step_size*N*(rep(digamma(sum(alphas)), 5) - digamma(alphas) + log_p/N)
  if (forward) {
    alphas = alphas + x
  } else {
    alphas = alphas - x
  }
  return(alphas)
}

library(distr)

fp_step <- function(alphas, log_p, N) {
  return(igamma(rep(digamma(sum(alphas)), 5) + log_p/N))
}

# this step must be repeated many times.
alphas <- fp_step(alphas, log_p, N)

df.test <- filter(raw_df, item %in% df.test$item) %>%
  group_by(item) %>% 
  arrange(timestamp, .by_group = TRUE)

quant_df <- group_by(df.test, item) %>%
  summarize(q70 = quantile(timestamp, .7))

df.test <- inner_join(df.test, quant_df, by="item")

df.test_learn = filter(df.test, timestamp <= q70)
df.test_eval = filter(df.test, timestamp > q70)

df.test_learn <- dcast(setDT(df.test_learn), item ~ rating, fun.aggregate = length) %>% arrange(item)
df.test_eval <- dcast(setDT(df.test_eval), item ~ rating, fun.aggregate = length) %>% arrange(item)

colnames(df.test_learn) <- c("item", "n1", "n2", "n3", "n4", "n5")
colnames(df.test_eval) <- c("item", "n1", "n2", "n3", "n4", "n5")

df.test_learn$n = df.test_learn$"n1" + df.test_learn$"n2" + df.test_learn$"n3" + df.test_learn$"n4" + df.test_learn$"n5"
df.test_eval$n = df.test_eval$"n1" + df.test_eval$"n2" + df.test_eval$"n3" + df.test_eval$"n4" + df.test_eval$"n5"

df.test_learn <- mutate(df.test_learn, b1 = n1 + alphas[1], b2 = n2 + alphas[2], b3 = n3 + alphas[3], 
                        b4 = n4 + alphas[4], b5 = n5 + alphas[5], b = n + sum(alphas))

df.test_learn_frequentist <- transmute(df.test_learn, n1 = n1 / n, n2 = n2 / n, n3 = n3 / n, n4 = n4 / n, n5 = n5 / n)
df.test_learn_eb <- transmute(df.test_learn, b1 = b1/b, b2 = b2/b, b3 = b3/b, b4 = b4/b, b5 = b5/b)
df.test_eval <- transmute(df.test_eval, n1 = n1 / n, n2 = n2 / n, n3 = n3 / n, n4 = n4 / n, n5 = n5 / n)

# better metric to look at? This basically says for prediction of future values, freq and eb are same
mse_freq = sum((df.test_learn_frequentist - df.test_eval)^2) / (nrow(df.test_eval)*ncol(df.test_eval))
mse_eb = sum((df.test_learn_eb - df.test_eval)^2) / (nrow(df.test_eval)*ncol(df.test_eval))

# let's plot a scatter plot of number of datapoints and mse

df.test_learn$mse_f = rowSums((df.test_learn_frequentist - df.test_eval)^2)/5
df.test_learn$mse_b = rowSums((df.test_learn_eb - df.test_eval)^2)/5

a <- ggplot(df.test_learn, aes(x=n, y=mse_f)) + geom_point()
b <- ggplot(df.test_learn, aes(x=n,y = mse_b)) + geom_point()

library(gridExtra)

grid.arrange(a, b, ncol=2)


# other questions - how to deal with missing all ratings of one category for certain ASINs?
