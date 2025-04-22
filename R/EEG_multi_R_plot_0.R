rm(list = ls())

setwd("E:/MSPH/EEG methodology/Advanced EEG Code/Code for GitHub/SMGP_BCI_EEG/R")

library(ggplot2)
library(dplyr)
library(tidyr)

persons <- c("K178") 

hard_threshold <- 0.5

for (person in persons){

  EEG_multi_path <- paste0("E:/MSPH/EEG methodology/Advanced EEG Code/Code for GitHub/SMGP_BCI_EEG/EEG_multi/", person, "/R_plots")
  EEG_multi_ref_path <- paste0("E:/MSPH/EEG methodology/Advanced EEG Code/Code for GitHub/SMGP_BCI_EEG/EEG_multi_ref/", person, "/R_plots")
  
  EEG_train <- read.csv(paste0(EEG_multi_path, "/train_acc.csv"),header = FALSE)
  EEG_test <- read.csv(paste0(EEG_multi_path, "/test_acc.csv"),header = FALSE)
  EEG_ref_train <- read.csv(paste0(EEG_multi_ref_path, "/train_acc.csv"),header = FALSE)
  EEG_ref_test <- read.csv(paste0(EEG_multi_ref_path, "/test_acc.csv"),header = FALSE)
  EEG_swLDA_train <- read.csv(paste0(EEG_multi_path, "/swLDA_train_accuracy.csv"),header = FALSE)
  EEG_swLDA_test <- read.csv(paste0(EEG_multi_path, "/swLDA_test_accuracy.csv"),header = FALSE)
  # EEG_LDA_train <- read.csv(paste0(EEG_multi_path, "/xDAWN_LDA_train_accuracy.csv"),header = FALSE)
  # EEG_LDA_test <- read.csv(paste0(EEG_multi_path, "/xDAWN_LDA_test_accuracy.csv"),header = FALSE)
  
  train_sequence <- nrow(EEG_train)
  test_sequence <- nrow(EEG_test)
  
  EEG_train <- EEG_train %>% mutate(Legend = "SMGP")
  EEG_test <- EEG_test %>% mutate(Legend = "SMGP")
  EEG_ref_train <- EEG_ref_train %>% mutate(Legend = "BLDA")
  EEG_ref_test <- EEG_ref_test %>% mutate(Legend = "BLDA")
  EEG_swLDA_train <- EEG_swLDA_train %>% mutate(Legend = "swLDA")
  EEG_swLDA_test <- EEG_swLDA_test %>% mutate(Legend = "swLDA")
  # EEG_LDA_train <- EEG_LDA_train %>% mutate(Legend = "LDA")
  # EEG_LDA_test <- EEG_LDA_test %>% mutate(Legend = "LDA")
  
  EEG_train <- EEG_train[1:train_sequence,]
  EEG_test <- EEG_test[1:test_sequence,]
  EEG_ref_train <- EEG_ref_train[1:train_sequence,]
  EEG_ref_test <- EEG_ref_test[1:test_sequence,]
  EEG_swLDA_train <- EEG_swLDA_train[1:train_sequence,]
  EEG_swLDA_test <- EEG_swLDA_test[1:test_sequence,]
  # EEG_LDA_train <- EEG_LDA_train[1:train_sequence,]
  # EEG_LDA_test <- EEG_LDA_test[1:test_sequence,]
  
  smgp_final <- EEG_test[test_sequence, "V1"]
  swlda_final <- EEG_swLDA_test[test_sequence, "V1"]
  blda_final <- EEG_ref_test[test_sequence, "V1"]

  train <- rbind(EEG_train, EEG_ref_train, EEG_swLDA_train)
  test <- rbind(EEG_test, EEG_ref_test, EEG_swLDA_test)
  
  train$Legend <- factor(train$Legend, levels = c("SMGP", "BLDA", "swLDA"))
  test$Legend <- factor(test$Legend, levels = c("SMGP", "BLDA", "swLDA"))
  
  train <- train %>% group_by(Legend) %>% mutate(Index = row_number())
  test <- test %>% group_by(Legend) %>% mutate(Index = row_number())
  
  p <- ggplot(train, aes(x = Index, y = V1, color = Legend)) +
    geom_line(linewidth = 1.5) +  # 绘制折线
    geom_point(shape = 21, size = 3, fill = "white", stroke = 1.5) + # 圆圈标记
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) + # 设置Y轴
    scale_x_continuous(limits = c(1, train_sequence), breaks = seq(1, train_sequence, by = 1)) + # 设置X轴
    geom_hline(yintercept = 0.7, linetype = "dashed", color = "black") + # Y=0.7 虚线
    labs(x = "Number of Sequence",
         y = "Accuracy") +
    theme_bw() +
    theme( panel.grid.major = element_blank(),
           panel.grid.minor = element_blank(),
           
           axis.text = element_text(size = 25),
           axis.title = element_text(size = 25),
           legend.position = c(1, 0),  # 调整图例位置到右下角
           legend.justification = c(1, 0),  # 调整图例位置到右下角
           legend.background = element_rect(fill = NA),
           plot.background = element_rect(fill = "white", color = NA),
           legend.text = element_text(size = 22),  # 放大图例文本
           legend.title = element_blank()) +
    guides(color = guide_legend(keywidth = unit(1.5, "cm"), keyheight = unit(1.5, "cm")))  # 修改图例形状和大小
  ggsave(paste0(EEG_multi_path, "/train_acc.png"),plot = p, width = 6, height = 10, dpi = 600, bg = "white")
  
  b <- ggplot(test, aes(x = Index, y = V1, color = Legend)) +
    geom_line(linewidth = 1.5) +  # 绘制折线
    geom_point(shape = 21, size = 3, fill = "white", stroke = 1.5) + # 圆圈标记
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) + # 设置Y轴
    scale_x_continuous(limits = c(1, test_sequence), breaks = seq(1, test_sequence, by = 1)) + # 设置X轴
    geom_hline(yintercept = 0.7, linetype = "dashed", color = "black") + # Y=0.7 虚线
    labs(x = "Number of Sequence",
         y = "Accuracy") +
    theme_bw() +
    theme( panel.grid.major = element_blank(),
           panel.grid.minor = element_blank(),
           
           axis.text = element_text(size = 25),
           axis.title = element_text(size = 25),
           legend.position = c(1, 0),  # 调整图例位置到右下角
           legend.justification = c(1, 0),  # 调整图例位置到右下角
           legend.background = element_rect(fill = NA),
           plot.background = element_rect(fill = "white", color = NA),
           legend.text = element_text(size = 22),  # 放大图例文本
           legend.title = element_blank()) +
    guides(color = guide_legend(keywidth = unit(1.5, "cm"), keyheight = unit(1.5, "cm")))  # 修改图例形状和大小
  ggsave(paste0(EEG_multi_path, "/test_acc.png"),plot = b, width = 6, height = 10, dpi = 600, bg = "white")
  
  EEG_zeta_1 <- read.csv(paste0(EEG_multi_path, "/zeta_1.csv"),header = T)
  EEG_zeta_2 <- read.csv(paste0(EEG_multi_path, "/zeta_2.csv"),header = T)
  
  EEG_beta_1 <- read.csv(paste0(EEG_multi_path, "/beta_1.csv"),header = T) %>%
    select(-Channel)
  EEG_beta_1_long <- EEG_beta_1 %>%
    tidyr::pivot_longer(
      cols = -Time,
      names_to = c("Beta", ".value"),
      names_pattern = "Beta_(\\d+)_(min|mean|max)"
    ) %>%
    mutate(
      Beta = factor(Beta,
                    levels = c("1", "0"),
                    labels = c("target", "non-target")
      )
    )
  
  EEG_beta_2 <- read.csv(paste0(EEG_multi_path, "/beta_2.csv"),header = T) %>%
    select(-Channel)
  EEG_beta_2_long <- EEG_beta_2 %>%
    tidyr::pivot_longer(
      cols = -Time,
      names_to = c("Beta", ".value"),
      names_pattern = "Beta_(\\d+)_(min|mean|max)"
    ) %>%
    mutate(
      Beta = factor(Beta,
                    levels = c("1", "0"),
                    labels = c("target", "non-target")
      )
    )
  
  EEG_ref_beta_1 <- read.csv(paste0(EEG_multi_ref_path, "/beta_1.csv"),header = T) %>%
    select(-Channel)
  EEG_ref_beta_1_long <- EEG_ref_beta_1 %>%
    tidyr::pivot_longer(
      cols = -Time,
      names_to = c("Beta", ".value"),
      names_pattern = "Beta_(\\d+)_(min|mean|max)"
    ) %>%
    mutate(
      Beta = factor(Beta,
                    levels = c("1", "0"),
                    labels = c("target", "non-target")
      )
    )
  
  EEG_ref_beta_2 <- read.csv(paste0(EEG_multi_ref_path, "/beta_2.csv"),header = T) %>%
    select(-Channel)
  EEG_ref_beta_2_long <- EEG_ref_beta_2 %>%
    tidyr::pivot_longer(
      cols = -Time,
      names_to = c("Beta", ".value"),
      names_pattern = "Beta_(\\d+)_(min|mean|max)"
    ) %>%
    mutate(
      Beta = factor(Beta,
                    levels = c("1", "0"),
                    labels = c("target", "non-target")
      )
    )
  
  p <- ggplot(EEG_zeta_1, aes(x = Time)) + 
    # geom_ribbon(aes(ymin = pmax(Zeta_min, 0), ymax = pmin(Zeta_max, 1), fill = "Zeta 95% Credible Interval"), alpha = 0.2) +
    geom_line(aes(y = Zeta_mean, color = "Zeta Mean"),linewidth = 1.5) +
    geom_point(aes(y = Zeta_mean), shape = 21, size = 3, fill = "white", stroke = 1.5) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
    geom_hline(yintercept = hard_threshold, linetype = "dashed", color = "black") +
    scale_color_manual(values = c("Zeta Mean" = "blue")) +
    # scale_fill_manual(values = c("Zeta 95% Credible Interval" = "blue")) +  # 更改为灰色以确保可见性
    labs(title = "Zeta Estimation for Component 1",
         x = "ms",
         y = "",
         color = "",
         fill = "") +
    theme_bw() +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          
          axis.text = element_text(size = 30),
          axis.title = element_text(size = 30),
          legend.position = "none",
          legend.background = element_rect(fill = "white"),
          plot.background = element_rect(fill = "white", color = NA),
          plot.title = element_text(size = 40, face = "bold", hjust = 0.5),
          legend.text = element_text(size = 22))
  
  ggsave(paste0(EEG_multi_path, "/zeta_1.png"),plot = p, width = 13, height = 10, dpi = 600)
  
  p <- ggplot(EEG_zeta_2, aes(x = Time)) + 
    # geom_ribbon(aes(ymin = pmax(Zeta_min, 0), ymax = pmin(Zeta_max, 1), fill = "Zeta 95% Credible Interval"), alpha = 0.2) +
    geom_line(aes(y = Zeta_mean, color = "Zeta Mean"),linewidth = 1.5) +
    geom_point(aes(y = Zeta_mean), shape = 21, size = 3, fill = "white", stroke = 1.5) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
    geom_hline(yintercept = hard_threshold, linetype = "dashed", color = "black") +
    scale_color_manual(values = c("Zeta Mean" = "blue")) +
    # scale_fill_manual(values = c("Zeta 95% Credible Interval" = "blue")) +  # 更改为灰色以确保可见性
    labs(title = "Zeta Estimation for Component 2",
         x = "ms",
         y = "",
         color = "",
         fill = "") +
    theme_bw() +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          
          axis.text = element_text(size = 30),
          axis.title = element_text(size = 30),
          legend.position = "none",
          legend.background = element_rect(fill = "white"),
          plot.background = element_rect(fill = "white", color = NA),
          plot.title = element_text(size = 40, face = "bold", hjust = 0.5),
          legend.text = element_text(size = 22))
  
  ggsave(paste0(EEG_multi_path, "/zeta_2.png"),plot = p, width = 13, height = 10, dpi = 600)
  
  y_max <- max(c(EEG_ref_beta_1$Beta_0_max, EEG_ref_beta_2$Beta_0_max, EEG_ref_beta_1$Beta_1_max, EEG_ref_beta_2$Beta_1_max,
                 EEG_beta_1$Beta_0_max, EEG_beta_2$Beta_0_max, EEG_beta_1$Beta_1_max, EEG_beta_2$Beta_1_max))
  y_min <- min(c(EEG_ref_beta_1$Beta_0_min, EEG_ref_beta_2$Beta_0_min, EEG_ref_beta_1$Beta_1_min, EEG_ref_beta_2$Beta_1_min,
                 EEG_beta_1$Beta_0_min, EEG_beta_2$Beta_0_min, EEG_beta_1$Beta_1_min, EEG_beta_2$Beta_1_min))
  
  
  # create plot of beta 1
  p <- ggplot(EEG_beta_1_long, aes(x = Time)) +
    geom_ribbon(
      aes(ymin = min, ymax = max, fill = Beta),
      alpha = 0.35,
      color = NA
    ) +
    geom_line(
      aes(y = mean, color = Beta),
      linewidth = 1.2
    ) +
    # data points
    geom_point(
      aes(y = mean, color = Beta),
      shape = 21,          
      size = 2,           
      fill = "white",      
      stroke = 1.5,        
      alpha = 0.8          
    ) +
    # point color
    scale_color_manual(
      name = "Condition",
      values = c("target" = "red", "non-target" = "blue"),
      guide = guide_legend(override.aes = list(
        linetype = c(1, 1),  
        shape = c(21, 21),   
        fill = c("white", "white")  
      ))
    ) +
    # line color
    scale_color_manual(
      values = c("target" = "red", "non-target" = "blue")
    ) +
    scale_fill_manual(
      values = c("target" = "red", "non-target" = "blue")
    ) +
    # axis labels
    scale_x_continuous(
      name = "Time (ms)",
      breaks = seq(0, 1000, 200),
      expand = expansion(mult = c(0.03, 0.03))  # 右侧保留5%空间
    ) +
    scale_y_continuous(
      name = "Component 1",
      limits = c(-3, 2),
      breaks = seq(-3, 2, 1)
    ) +
    # theme settings
    theme(
      # axixs
      axis.title.x = element_text(size = 25, margin = margin(t = 5)),
      axis.title.y = element_text(size = 25, margin = margin(r = 5)),
      axis.text = element_text(size = 25, color = "gray30"),
      
      # legend
      legend.position = c(1, 1),
      legend.box = "vertical",
      legend.direction = "vertical",
      legend.justification = c(1, 1),
      legend.background = element_blank(), 
      legend.key = element_blank(),    
      legend.spacing.x = unit(1, "cm"),  
      legend.text = element_text(size = 25),
      legend.title = element_blank(),
      
      # margins
      plot.margin = margin(t = 30, r = 30, l = 30, b = 30, unit = "pt"),  
      panel.grid.major = element_line(color = "white", linewidth = 0.3),
      plot.background = element_rect(fill = "white", color = NA)
    ) +
    # control the x-axis limits
    coord_cartesian(xlim = c(0, 1000))  
  
  # save plot
  ggsave(
    paste0(EEG_multi_path, "/beta_1.png"),
    plot = p,
    width = 13,  
    height = 10,
    dpi = 600,
    bg = "white"
  )
  
  # delete the plot
  rm(p)
  
  # create plot of beta 2
  p <- ggplot(EEG_beta_2_long, aes(x = Time)) +
    geom_ribbon(
      aes(ymin = min, ymax = max, fill = Beta),
      alpha = 0.35,
      color = NA
    ) +
    geom_line(
      aes(y = mean, color = Beta),
      linewidth = 1.2
    ) +
    # data points
    geom_point(
      aes(y = mean, color = Beta),
      shape = 21,          
      size = 2,           
      fill = "white",      
      stroke = 1.5,        
      alpha = 0.8          
    ) +
    # point color
    scale_color_manual(
      name = "Condition",
      values = c("target" = "red", "non-target" = "blue"),
      guide = guide_legend(override.aes = list(
        linetype = c(1, 1),  
        shape = c(21, 21),   
        fill = c("white", "white")  
      ))
    ) +
    # line color
    scale_color_manual(
      values = c("target" = "red", "non-target" = "blue")
    ) +
    scale_fill_manual(
      values = c("target" = "red", "non-target" = "blue")
    ) +
    # axis labels
    scale_x_continuous(
      name = "Time (ms)",
      breaks = seq(0, 1000, 200),
      expand = expansion(mult = c(0.03, 0.03))  # 右侧保留5%空间
    ) +
    scale_y_continuous(
      name = "Component 2",
      limits = c(-3, 2),
      breaks = seq(-3, 2, 1)
    ) +
    # theme settings
    theme(
      # axixs
      axis.title.x = element_text(size = 25, margin = margin(t = 5)),
      axis.title.y = element_text(size = 25, margin = margin(r = 5)),
      axis.text = element_text(size = 25, color = "gray30"),
      
      # legend
      legend.position = c(1, 1),
      legend.box = "vertical",
      legend.direction = "vertical",
      legend.justification = c(1, 1),
      legend.background = element_blank(), 
      legend.key = element_blank(),    
      legend.spacing.x = unit(1, "cm"),  
      legend.text = element_text(size = 25),
      legend.title = element_blank(),
      
      # margins
      plot.margin = margin(t = 30, r = 30, l = 30, b = 30, unit = "pt"),  
      panel.grid.major = element_line(color = "white", linewidth = 0.3),
      plot.background = element_rect(fill = "white", color = NA)
    ) +
    # control the x-axis limits
    coord_cartesian(xlim = c(0, 1000))  
  
  # save plot
  ggsave(
    paste0(EEG_multi_path, "/beta_2.png"),
    plot = p,
    width = 13,  
    height = 10,
    dpi = 600,
    bg = "white"
  )
  
  # delete the plot
  rm(p)
  
  # create plot of ref beta 1
  p <- ggplot(EEG_ref_beta_1_long, aes(x = Time)) +
    geom_ribbon(
      aes(ymin = min, ymax = max, fill = Beta),
      alpha = 0.35,
      color = NA
    ) +
    geom_line(
      aes(y = mean, color = Beta),
      linewidth = 1.2
    ) +
    # data points
    geom_point(
      aes(y = mean, color = Beta),
      shape = 21,          
      size = 2,           
      fill = "white",      
      stroke = 1.5,        
      alpha = 0.8          
    ) +
    # point color
    scale_color_manual(
      name = "Condition",
      values = c("target" = "red", "non-target" = "blue"),
      guide = guide_legend(override.aes = list(
        linetype = c(1, 1),  
        shape = c(21, 21),   
        fill = c("white", "white")  
      ))
    ) +
    # line color
    scale_color_manual(
      values = c("target" = "red", "non-target" = "blue")
    ) +
    scale_fill_manual(
      values = c("target" = "red", "non-target" = "blue")
    ) +
    # axis labels
    scale_x_continuous(
      name = "Time (ms)",
      breaks = seq(0, 1000, 200),
      expand = expansion(mult = c(0.03, 0.03))  # 右侧保留5%空间
    ) +
    scale_y_continuous(
      name = "Component 1",
      limits = c(-3, 2),
      breaks = seq(-3, 2, 1)
    ) +
    # theme settings
    theme(
      # axixs
      axis.title.x = element_text(size = 25, margin = margin(t = 5)),
      axis.title.y = element_text(size = 25, margin = margin(r = 5)),
      axis.text = element_text(size = 25, color = "gray30"),
      
      # legend
      legend.position = c(1, 1),
      legend.box = "vertical",
      legend.direction = "vertical",
      legend.justification = c(1, 1),
      legend.background = element_blank(), 
      legend.key = element_blank(),    
      legend.spacing.x = unit(1, "cm"),  
      legend.text = element_text(size = 25),
      legend.title = element_blank(),
      
      # margins
      plot.margin = margin(t = 30, r = 30, l = 30, b = 30, unit = "pt"),  
      panel.grid.major = element_line(color = "white", linewidth = 0.3),
      plot.background = element_rect(fill = "white", color = NA)
    ) +
    # control the x-axis limits
    coord_cartesian(xlim = c(0, 1000))  
  
  # save plot
  ggsave(
    paste0(EEG_multi_ref_path, "/beta_1.png"),
    plot = p,
    width = 13, 
    height = 10,
    dpi = 600,
    bg = "white"
  )
  
  # delete the plot
  rm(p)
  
  # create plot of ref beta 2
  p <- ggplot(EEG_ref_beta_2_long, aes(x = Time)) +
    geom_ribbon(
      aes(ymin = min, ymax = max, fill = Beta),
      alpha = 0.35,
      color = NA
    ) +
    geom_line(
      aes(y = mean, color = Beta),
      linewidth = 1.2
    ) +
    # data points
    geom_point(
      aes(y = mean, color = Beta),
      shape = 21,          
      size = 2,           
      fill = "white",      
      stroke = 1.5,        
      alpha = 0.8          
    ) +
    # point color
    scale_color_manual(
      name = "Condition",
      values = c("target" = "red", "non-target" = "blue"),
      guide = guide_legend(override.aes = list(
        linetype = c(1, 1),  
        shape = c(21, 21),   
        fill = c("white", "white")  
      ))
    ) +
    # line color
    scale_color_manual(
      values = c("target" = "red", "non-target" = "blue")
    ) +
    scale_fill_manual(
      values = c("target" = "red", "non-target" = "blue")
    ) +
    # axis labels
    scale_x_continuous(
      name = "Time (ms)",
      breaks = seq(0, 1000, 200),
      expand = expansion(mult = c(0.03, 0.03))  # 右侧保留5%空间
    ) +
    scale_y_continuous(
      name = "Component 2",
      limits = c(-3, 2),
      breaks = seq(-3, 2, 1)
    ) +
    # theme settings
    theme(
      # axixs
      axis.title.x = element_text(size = 25, margin = margin(t = 5)),
      axis.title.y = element_text(size = 25, margin = margin(r = 5)),
      axis.text = element_text(size = 25, color = "gray30"),
      
      # legend
      legend.position = c(1, 1),
      legend.box = "vertical",
      legend.direction = "vertical",
      legend.justification = c(1, 1),
      legend.background = element_blank(), 
      legend.key = element_blank(),    
      legend.spacing.x = unit(1, "cm"),  
      legend.text = element_text(size = 25),
      legend.title = element_blank(),
      
      # margins
      plot.margin = margin(t = 30, r = 30, l = 30, b = 30, unit = "pt"),  
      panel.grid.major = element_line(color = "white", linewidth = 0.3),
      plot.background = element_rect(fill = "white", color = NA)
    ) +
    # control the x-axis limits
    coord_cartesian(xlim = c(0, 1000))  
  
  # save plot
  ggsave(
    paste0(EEG_multi_ref_path, "/beta_2.png"),
    plot = p,
    width = 13,  
    height = 10,
    dpi = 600,
    bg = "white"
  )
  
  # delete the plot
  rm(p)
}
