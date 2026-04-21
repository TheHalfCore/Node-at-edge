library(ggplot2)
library(dplyr)

# --- Load data ---
df_node <- read.csv("node.csv") %>% mutate(model = "NODE")
df_rf   <- read.csv("rf.csv")   %>% mutate(model = "RF")

# --- Combine datasets ---
df_all <- bind_rows(df_node, df_rf)

# --- Pareto function (keep your cummax version if you want) ---
get_pareto <- function(df) {
  df %>%
    arrange(model_memory_kb, desc(f1_score)) %>%
    mutate(best_so_far = cummax(f1_score)) %>%
    filter(f1_score == best_so_far) %>%
    distinct(model_memory_kb, f1_score, .keep_all = TRUE)
}

# --- Compute Pareto per model ---
df_pareto <- df_all %>%
  group_by(model) %>%
  group_modify(~ get_pareto(.x)) %>%
  ungroup()

# --- Plot ---
ggplot(df_all, aes(x = model_memory_kb, y = f1_score, color = model)) +
  
  # All points
  geom_point(alpha = 0.5, size = 2.5) +
  
  # Pareto lines
  geom_line(data = df_pareto %>% arrange(model, model_memory_kb),
            linewidth = 1) +
  
  # Pareto points
  geom_point(data = df_pareto,
             size = 3) +
  
  coord_cartesian(xlim = c(0, 450), ylim = c(0.5, 0.9)) +
  
  theme_minimal() +
  labs(
    title = "Pareto Frontier: NODE vs RF",
    x = "Model Memory (KB)",
    y = "F1 Score",
    color = "Model"
  )