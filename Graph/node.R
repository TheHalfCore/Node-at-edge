library(ggplot2)
library(dplyr)

df_node <- read.csv("optuna_NODE_results.csv")

df_pareto_node <- df_node %>%
  arrange(alloc_memory, desc(f1_score)) %>%
  mutate(best_so_far = cummax(f1_score)) %>%
  filter(f1_score == best_so_far) %>%   # ✅ FIX HERE
  distinct(alloc_memory, f1_score, .keep_all = TRUE)

ggplot(df_node, aes(x = alloc_memory, y = f1_score)) +
  geom_point(color = "grey60", size = 3) +
  geom_line(data = df_pareto_node, color = "black", linewidth = 1) +
  geom_point(data = df_pareto_node, color = "black", size = 3) +
  theme_minimal() +
  labs(
    title = "Pareto Frontier (NODE)",
    x = "Model Memory (MB)",
    y = "F1 Score"
  )