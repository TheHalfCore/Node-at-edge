library(ggplot2)
library(dplyr)

df_rf <- read.csv("optuna_RF_results.csv")

df_pareto_rf <- df_rf %>%
  arrange(model_size_mb, desc(f1_score)) %>%
  mutate(best_so_far = cummax(f1_score)) %>%
  filter(f1_score == best_so_far) %>%   # ✅ FIX HERE
  distinct(model_size_mb, f1_score, .keep_all = TRUE)

ggplot(df_rf, aes(x = model_size_mb, y = f1_score)) +
  geom_point(color = "grey60", size = 3) +
  geom_line(data = df_pareto_rf, color = "black", linewidth = 1) +
  geom_point(data = df_pareto_rf, color = "black", size = 3) +
  theme_minimal() +
  labs(
    title = "Pareto Frontier (RF)",
    x = "Model Memory (KB)",
    y = "F1 Score"
  )