library(ggplot2)
library(dplyr)

df_node <- read.csv("node.csv")

df_pareto_node <- df_node %>%
  arrange(model_memory_kb, desc(f1_score)) %>%
  mutate(best_so_far = cummax(f1_score)) %>%
  filter(f1_score == best_so_far) %>%   # ✅ FIX HERE
  distinct(model_memory_kb, f1_score, .keep_all = TRUE)

ggplot(df_node, aes(x = model_memory_kb, y = f1_score)) +
  geom_point(color = "grey60", size = 3) +
  geom_line(data = df_pareto_node, color = "black", linewidth = 1) +
  geom_point(data = df_pareto_node, color = "black", size = 3) +
  theme_minimal() +
  labs(
    title = "Pareto Frontier (NODE)",
    x = "Model Memory (KB)",
    y = "F1 Score"
  )