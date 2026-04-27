library(ggplot2)
library(dplyr)

df_rf <- read.csv("optuna_Test_RF_withDataAgg_results.csv")

df_pareto_rf <- df_rf %>%
  arrange(model_size_mb, desc(f1_score)) %>%
  mutate(best_so_far = cummax(f1_score)) %>%
  filter(f1_score == best_so_far) %>%   # ✅ FIX HERE
  distinct(model_size_mb, f1_score, .keep_all = TRUE)

ggplot(df_rf, aes(x = model_size_mb, y = f1_score)) +
  geom_point(color = "grey60", size = 3) +
  geom_line(data = df_pareto_rf, color = "black", linewidth = 1) +
  geom_point(data = df_pareto_rf, color = "black", size = 3) +
  scale_x_continuous(
    breaks = seq(floor(min(df_rf$model_size_mb)),
                 ceiling(max(df_rf$model_size_mb)),
                 by = 3)   
  ) +
  scale_y_continuous(
    breaks = seq(floor(min(df_rf$f1_score)*100)/100,
                 ceiling(max(df_rf$f1_score)*100)/100,
                 by = 0.02)  
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)  # 👈 angled labels
  ) +
  labs(
    title = "Pareto Frontier (RF)",
    x = "Model Memory (MB)",
    y = "F1 Score"
  )