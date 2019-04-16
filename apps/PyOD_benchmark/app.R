library(ggplot2)
library(reshape2)
library(dplyr)
library(shiny)

# Read ROC results --> not always optimal with unbalanced data
df_auc <- read.csv('data/roc_results.csv')
# Read Precision in top N observations --> better for anomaly detection 
df_pn <- read.csv('data/topn_results.csv')
# Read ROC results --> not always optimal with unbalanced data
df_time <- read.csv('data/time_results.csv')

vars <- c('nSamples', 'nDimensions', 'OutlierPerc')

# Define the UI
ui <- bootstrapPage(
  headerPanel('PyOD benchmark results'),
  sidebarPanel(
    radioButtons('metric', 'Select metric:', 
                 choices = c('TopN Precision', 'AUC')),
    radioButtons('selected_var', 'Select variable:', 
                 choices = c('None', vars))
  ),
  mainPanel(
    tabsetPanel(
      tabPanel("Boxplot", plotOutput("boxplot")),
      tabPanel("Scatter plot", plotOutput("scatterplot")),
      tabPanel("Facet plot", plotOutput("facetplot")),
      tabPanel("Time Boxplot", plotOutput("time_boxplot")),
      tabPanel("Time Facetplot", plotOutput("time_facetplot"))
    )
  )
)


# Define the server code
server <- function(input, output) {
  
  df <- reactive({
    if (input$metric == 'TopN Precision') {
      df <- df_pn
    } else if (input$metric == 'AUC') {
      df <- df_auc
    } else {
      stop('Unknown metric')
    }
    
    drop_cols <- setdiff(vars, input$selected_var)
    id_vars <- setdiff(c('Data', input$selected_var), 'None')
    
    df_plot <- df %>%
      select(-drop_cols) %>%
      melt(id.vars = id_vars)
    
    if (!state()$no_var) {
      df_plot[,input$selected_var] <- log(df_plot[,input$selected_var], base = 12)
    }
    
    return(df_plot)
    
  })
  
  df_time <- reactive({
    
    drop_cols <- setdiff(vars, input$selected_var)
    id_vars <- setdiff(c('Data', input$selected_var), 'None')
    
    df_plot <- df_time %>%
      select(-drop_cols) %>%
      melt(id.vars = id_vars)
    
    if (!state()$no_var) {
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
    gg <- ggplot(data = df(), aes_string(x = 'variable', y = 'value')) +
      geom_boxplot() +
      ggtitle(state()$plot_title)
    
    if (!state()$no_var) {
      gg <- gg + 
        geom_point(aes_string(color = input$selected_var)) +
        scale_color_viridis_c()
    }
    return( gg )
  })
  
  output$scatterplot <- renderPlot({
    gg <- ggplot(data = df(), aes_string(x = 'variable', y = 'value')) +
      ggtitle(state()$plot_title)
    
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
    
    gg <- ggplot(data = df(), aes_string(x = x_var, y = 'value')) +
      geom_point() +
      geom_smooth(method = "lm", se=FALSE, color="darkgrey") +
      facet_wrap(~variable) +
      ggtitle(state()$plot_title)
    
    if (state()$no_var) {
      gg <- gg + theme(axis.text.x = element_text(angle = 90, hjust = 1))
    }
    
    return ( gg )
  })
  
  output$time_boxplot <- renderPlot({
    
    gg <- ggplot(data = df_time(), aes_string(x = 'variable', y = 'value')) +
      geom_boxplot() +
      ggtitle(state()$plot_title)
    
    if (!state()$no_var) {
      gg <- gg + 
        geom_point(aes_string(color = input$selected_var)) +
        scale_color_viridis_c()
    }
    return( gg )
  })
  
  output$time_facetplot <- renderPlot({
    
    x_var <- ifelse(state()$no_var, 'Data', input$selected_var)
    
    gg <- ggplot(data = df_time(), aes_string(x = x_var, y = 'value')) +
      geom_point() +
      geom_smooth(method = "lm", se=FALSE, color="darkgrey") +
      facet_wrap(~variable) +
      ggtitle(state()$plot_title)
    
    if (state()$no_var) {
      gg <- gg + theme(axis.text.x = element_text(angle = 90, hjust = 1))
    }
    
    return ( gg )
  })
  
}

# Return a Shiny app object
shinyApp(ui = ui, server = server)

