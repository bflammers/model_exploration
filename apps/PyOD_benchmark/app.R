library(ggplot2)
library(reshape2)
library(dplyr)
library(shiny)

#input <- list(selected_var = 'None', metric = 'Time in seconds')

# Read ROC results --> not always optimal with unbalanced data
df_auc <- read.csv('data/roc_results.csv')
# Read Precision in top N observations --> better for anomaly detection 
df_pn <- read.csv('data/topn_results.csv')
# Read ROC results --> not always optimal with unbalanced data
df_time <- read.csv('data/time_results.csv')

as.character(df_time$Data)

models <- c("ABOD", "CBLOF", "FB", "HBOS", "IForest", "KNN", "LOF", "MCD",
            "OCSVM", "PCA")
exclude_models <- c("KNN", "LOF")

datasets <- c("arrhythmia", "cardio", "glass", "ionosphere","letter", "lympho",
              "mnist", "musk", "optdigits", "pendigits", "pima", "satellite", 
              "satimage-2", "shuttle", "vertebral", "vowels", "wbc")
exclude_datasets <- c("ionosphere", "pima", "satellite")

vars <- c('nSamples', 'nDimensions', 'OutlierPerc')

# Define the UI
ui <- bootstrapPage(
  headerPanel('PyOD benchmark results'),
  sidebarPanel(
    radioButtons('metric', 'Select metric:', 
                 choices = c('TopN Precision', 'AUC', 'Time in seconds')),
    radioButtons('selected_var', 'Select variable:', 
                 choices = c('None', vars)),
    checkboxInput("log_transform", label = "Log transform:", value = FALSE),
    checkboxGroupInput("include_models", label = "Include models:", 
                       choices = models, 
                       selected = setdiff(models, exclude_models)),
    checkboxGroupInput("include_datasets", label = "Include datasets:", 
                       choices = datasets, 
                       selected = setdiff(datasets, exclude_datasets))
  ),
  mainPanel(
    tabsetPanel(
      tabPanel("Boxplot", plotOutput("boxplot")),
      tabPanel("Scatter plot", plotOutput("scatterplot")),
      tabPanel("Facet plot", plotOutput("facetplot"))
    )
  )
)


# Define the server code
server <- function(input, output) {
  
  rdf <- reactive({
    if (input$metric == 'TopN Precision') {
      df <- df_pn
    } else if (input$metric == 'AUC') {
      df <- df_auc
    } else if (input$metric == 'Time in seconds') {
      df <- df_time
    } else {
      stop('Unknown metric')
    }
    
    drop_cols <- setdiff(vars, input$selected_var)
    id_vars <- setdiff(c('Data', input$selected_var), 'None')
    
    df_plot <- df %>%
      select(-drop_cols) %>%
      melt(id.vars = id_vars) %>% 
      filter(variable %in% input$include_models) %>% 
      filter(Data %in% input$include_datasets)
    
    if (!state()$no_var && input$log_transform) {
      df_plot[,input$selected_var] <- log(df_plot[,input$selected_var], base = 12)
    }
    
    return(df_plot)
    
  })
  
  state <- reactive({
    plot_title <- paste('Metric:', ifelse(input$metric == 'roc', 'ROC', 
                                          'Precision topN'))
    no_var <- input$selected_var == 'None'
    return(list(plot_title = plot_title, no_var = no_var))
  })
  
  output$boxplot <- renderPlot({
    
    gg <- ggplot(data = rdf(), aes_string(x = 'variable', y = 'value')) +
      geom_boxplot() +
      ggtitle(state()$plot_title) +
      ylab(input$metric)
    
    if (!state()$no_var) {
      gg <- gg + 
        geom_point(aes_string(color = input$selected_var)) +
        scale_color_viridis_c()
    }
    return( gg )
  })
  
  output$scatterplot <- renderPlot({
    gg <- ggplot(data = rdf(), aes_string(x = 'variable', y = 'value')) +
      ggtitle(state()$plot_title) +
      ylab(input$metric)
    
    if (state()$no_var) {
      gg <- gg + geom_point()
    } else {
      gg <- gg + geom_point(aes_string(color = input$selected_var)) +
      scale_color_viridis_c()
    }
    return( gg )
  })
  
  output$facetplot <- renderPlot({
    
    x_var <- ifelse(state()$no_var, 'Data', input$selected_var)
    
    gg <- ggplot(data = rdf(), aes_string(x = x_var, y = 'value')) +
      geom_point() +
      geom_smooth(method = "lm", se=FALSE, color="darkgrey") +
      facet_wrap(~variable) +
      ggtitle(state()$plot_title) +
      ylab(input$metric)
    
    if (state()$no_var) {
      gg <- gg + theme(axis.text.x = element_text(angle = 90, hjust = 1))
    }
    
    return ( gg )
  })
  
  # output$time_boxplot <- renderPlot({
  #   
  #   gg <- ggplot(data = rdf_time(), aes_string(x = 'variable', y = 'value')) +
  #     geom_boxplot() +
  #     ggtitle(state()$plot_title) +
  #     ylab("Time in seconds")
  #   
  #   if (!state()$no_var) {
  #     gg <- gg + 
  #       geom_point(aes_string(color = input$selected_var)) +
  #       scale_color_viridis_c()
  #   }
  #   return( gg )
  # })
  # 
  # output$time_facetplot <- renderPlot({
  #   
  #   x_var <- ifelse(state()$no_var, 'Data', input$selected_var)
  #   
  #   gg <- ggplot(data = rdf_time(), aes_string(x = x_var, y = 'value')) +
  #     geom_point() +
  #     geom_smooth(method = "lm", se=FALSE, color="darkgrey") +
  #     facet_wrap(~variable) +
  #     ggtitle(state()$plot_title) +
  #     ylab("Time in seconds")
  #   
  #   if (state()$no_var) {
  #     gg <- gg + theme(axis.text.x = element_text(angle = 90, hjust = 1))
  #   }
  #   
  #   return ( gg )
  # })
  
}

# Return a Shiny app object
shinyApp(ui = ui, server = server)

