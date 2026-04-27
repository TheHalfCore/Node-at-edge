library(ggplot2)
library(dplyr)

# --- Load data ---
df_node <- read.csv("optuna_Test_NODE_withDataAgg_results.csv") %>%
  rename(memory = alloc_memory) %>%
  mutate(model = "NODE")

df_rf <- read.csv("optuna_Test_RF_withDataAgg_results.csv") %>%
  rename(memory = model_size_mb) %>%
  mutate(model = "RF")

# --- Combine datasets ---
df_all <- bind_rows(df_node, df_rf)

# --- Pareto function ---
get_pareto <- function(df) {
  df %>%
    arrange(memory, desc(f1_score)) %>%
    mutate(best_so_far = cummax(f1_score)) %>%
    filter(f1_score == best_so_far) %>%
    distinct(memory, f1_score, .keep_all = TRUE)
}

# --- Compute Pareto per model ---
df_pareto <- df_all %>%
  group_by(model) %>%
  group_modify(~ get_pareto(.x)) %>%
  ungroup()

# --- Plot ---
ggplot(df_all, aes(x = memory, y = f1_score, color = model)) +
  
  # All points
  geom_point(alpha = 0.5, size = 2.5) +
  
  # Pareto lines
  geom_line(data = df_pareto %>% arrange(model, memory),
            linewidth = 1) +
  
  # Pareto points
  geom_point(data = df_pareto,
             size = 3) +
  
  scale_x_continuous(
    breaks = seq(
      floor(min(df_all$memory)),
      ceiling(max(df_all$memory)),
      by = 5
    )
  ) +
  scale_y_continuous(
    breaks = seq(
      floor(min(df_all$f1_score) * 100) / 100,
      ceiling(max(df_all$f1_score) * 100) / 100,
      by = 0.02
    )
  )+
  
  #coord_cartesian(xlim = c(0, 450), ylim = c(0.5, 0.9)) +
  
  theme_minimal() +
  labs(
    title = "Pareto Frontier: NODE vs RF",
    x = "Model Memory (MB)",
    y = "F1 Score",
    color = "Model"
  )