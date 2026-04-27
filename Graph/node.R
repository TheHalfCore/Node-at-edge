library(ggplot2)
library(dplyr)

df_node <- read.csv("optuna_Test_NODE_withDataAgg_results.csv")

df_pareto_node <- df_node %>%
  #mutate(alloc_memory = alloc_memory * 1024) %>%
  arrange(alloc_memory, desc(f1_score)) %>%
  mutate(best_so_far = cummax(f1_score)) %>%
  filter(f1_score == best_so_far) %>%   # ✅ FIX HERE
  distinct(alloc_memory, f1_score, .keep_all = TRUE)



ggplot(df_node, aes(x = alloc_memory, y = f1_score)) +
  geom_point(color = "grey60", size = 3) +
  geom_line(data = df_pareto_node, color = "black", linewidth = 1) +
  geom_point(data = df_pareto_node, color = "black", size = 3) +
  theme_minimal() +
  scale_x_continuous(
    breaks = seq(floor(min(df_node$alloc_memory)),
                 ceiling(max(df_node$alloc_memory)),
                 by = 1.5)   
  ) +
  scale_y_continuous(
    breaks = seq(floor(min(df_node$f1_score)*100)/100,
                 ceiling(max(df_node$f1_score)*100)/100,
                 by = 0.02)  
  ) +
  labs(
    title = "Pareto Frontier (NODE)",
    x = "Model Memory (MB)",
    y = "F1 Score"
  )
