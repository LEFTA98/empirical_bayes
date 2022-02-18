library(data.table)
library(tidyverse)

# alphas from naive experiment
alphas = c(18.41, 4.78, 7.57, 21.55, 309.98)
b = sum(alphas)

# load in the data
raw_df = read.csv("data/Gift_Cards.csv", header = FALSE, col.names=c("item", "user", "rating", "timestamp"))
head(raw_df)

df <- raw_df %>% 
  select(c("item", "rating"))

set.seed(1729)

counts <- count(df, item)

df = merge(counts, df, by="item")
in.train = sample(unique(df$item), size = 0.6*length(unique(df$item)))
df.train = filter(df, item %in% in.train)
df.test = filter(df, !item %in% in.train)

temp = filter(raw_df, item %in% df.test$item)
df.test <- merge(counts, temp, by="item")

df.test <- filter(df.test, n >= 100) %>%
  group_by(item) %>% 
  arrange(timestamp, .by_group = TRUE) %>%
  slice_head(n=100)

#split test set in half
df.test_learn = group_by(df.test, item) %>% 
  arrange(timestamp, .by_group = TRUE) %>%
  slice_head(n=50)
df.test_eval = group_by(df.test, item) %>% 
  arrange(timestamp, .by_group = TRUE) %>%
  slice_tail(n=50)

# set up evaluation datset for evaluating
df.test_eval <- dcast(setDT(df.test_eval), item ~ rating, fun.aggregate = length) %>% arrange(item)
colnames(df.test_eval) <- c("item", "n1", "n2", "n3", "n4", "n5")
df.test_eval$n = df.test_eval$"n1" + df.test_eval$"n2" + df.test_eval$"n3" + df.test_eval$"n4" + df.test_eval$"n5"
df.test_eval <- transmute(df.test_eval, n1 = n1 / n, n2 = n2 / n, n3 = n3 / n, n4 = n4 / n, n5 = n5 / n)

# start training and checking performance at each step
items = unique(df.test_learn$item)

mses_f = tibble(items)
mses_b = tibble(items)
kls_f = tibble(items)
kls_b = tibble(items)

for (i in c(1:50)) {
  df.test_learn_subset = group_by(df.test_learn, item) %>%
    arrange(timestamp, .by_group = TRUE) %>%
    slice_head(n=i)

  df.test_learn_subset <- dcast(setDT(df.test_learn_subset), item ~ rating, fun.aggregate = length, drop=FALSE) %>% arrange(item)
  
  # adding missing columns
  for (j in setdiff(c("item", "1", "2", "3", "4", "5"), colnames(df.test_learn_subset))) {
    df.test_learn_subset[[j]] <- rep(0, 89)
  }
  df.test_learn_subset <- df.test_learn_subset[,c("item", "1", "2", "3", "4", "5")]
  
  colnames(df.test_learn_subset) <- c("item", "n1", "n2", "n3", "n4", "n5")
  df.test_learn_subset$n = df.test_learn_subset$"n1" + df.test_learn_subset$"n2" + df.test_learn_subset$"n3" + df.test_learn_subset$"n4" + df.test_learn_subset$"n5"

  df.test_learn_f <- transmute(df.test_learn_subset, n1 = n1 / n, n2 = n2 / n, n3 = n3 / n, n4 = n4 / n, n5 = n5 / n)
  df.test_learn_b <- mutate(df.test_learn_subset, b1 = n1 + alphas[1], b2 = n2 + alphas[2], b3 = n3 + alphas[3],
         b4 = n4 + alphas[4], b5 = n5 + alphas[5], b = n + sum(alphas))
  df.test_learn_b <- transmute(df.test_learn_b, b1 = b1 / b, b2 = b2 / b, b3 = b3 / b, b4 = b4 / b, b5 = b5 / b)

  f_mse = rowSums((df.test_learn_f - df.test_eval)^2)
  b_mse = rowSums((df.test_learn_b - df.test_eval)^2)
  f_kl = rowSums(df.test_eval*log(df.test_eval/df.test_learn_f), na.rm = TRUE)
  f_kl[!is.finite(f_kl)] = 1
  b_kl = rowSums(df.test_eval*log(df.test_eval/df.test_learn_b), na.rm = TRUE)
  b_kl[!is.finite(b_kl)] = 1
  
  mses_f[[i]] = f_mse
  mses_b[[i]] = b_mse
  kls_f[[i]] = f_kl
  kls_b[[i]] = b_kl

}

mses_f = data.frame(val = colSums(mses_f) / 89)
mses_b = data.frame(val = colSums(mses_b) / 89)
kls_f = data.frame(val = colSums(kls_f) / 89)
kls_b = data.frame(val = colSums(kls_b) / 89)

mses_data = tibble(mses_f, mses_b, .name_repair = "unique")
colnames(mses_data) <- c("f", "b")

kls_data = tibble(kls_f, kls_b, .name_repair = "unique")
colnames(kls_data) <- c("f", "b")

#plotting time - these plots aren't great so work on em
library(gridExtra)
plot_mse_f = ggplot(mses_f, aes(x = seq_along(val), y=val)) + geom_point() + ggtitle("mse_f")
plot_mse_b = ggplot(mses_b, aes(x = seq_along(val), y=val)) + geom_point() + ggtitle("mse_b")
plot_kl_f = ggplot(kls_f, aes(x = seq_along(val), y=val)) + geom_point() + ggtitle("kl_f")
plot_kl_b = ggplot(kls_b, aes(x = seq_along(val), y=val)) + geom_point() + ggtitle("kl_b")

grid.arrange(plot_mse_f, plot_mse_b, ncol=2)
grid.arrange(plot_kl_f, plot_kl_b, ncol=2)

ggplot(mses_data) + 
  geom_point(aes(x=seq_along(b),y=f)) + 
  geom_point(aes(x=seq_along(b), y=b)) +
  ggtitle("mse loss")

ggplot(kls_data) + 
  geom_point(aes(x=seq_along(f),y=f)) + 
  geom_point(aes(x=seq_along(f), y=b)) +
  ggtitle("kl divergence from test")
