# Project Architecture Diagram

## System Architecture

```mermaid
architecture-beta
    group data(disk)[Data Layer]
        service csv(database)[CSV Data Source] in data
        service db(database)[Database] in data
    
    group models(server)[ML/Prediction Layer]
        service lstm(disk)[LSTM Model] in models
        service module(server)[Prediction Module] in models
        service service(server)[Prediction Service] in models
    
    group web(cloud)[Web Application]
        service main(server)[Flask App (main.py)] in web
        service plots(server)[Visualization] in web
    
    group ui(internet)[User Interface]
        service home(internet)[Home Template] in ui
        service dashboard(internet)[Dashboard] in ui
        service predictions(internet)[Predictions] in ui
    
    group eda(disk)[Analysis]
        service analysis(disk)[EDA Notebook] in eda
    
    csv:R --> L:db
    db:R --> L:module
    lstm:B --> T:service
    service:R --> L:main
    module:R --> L:main
    main:B --> T:plots
    main:R --> L:home{group}
    main:R --> L:dashboard{group}
    main:R --> L:predictions{group}
    csv:L --> L:analysis
```

## Architecture Overview

### Data Layer
- **CSV Data Source**: Raw data input from `donnees_synthetiques_tollxpress_benin_2023-2024.csv`
- **Database**: Data storage managed by `database.py`

### ML/Prediction Layer
- **LSTM Model**: Pre-trained Keras model (`lstm_heure.keras`)
- **Prediction Module**: Core prediction logic in `module_prevision.py`
- **Prediction Service**: REST service interface in `prediction_service.py`

### Web Application
- **Flask App**: Main application entry point (`main.py`)
- **Visualization**: Plotly charts (`plots_plotly.py`, `pltos.py`)

### User Interface
- **Home**: Landing page (`home.html`)
- **Dashboard**: Main dashboard view (`dashboard.html`)
- **Predictions**: Prediction results view (`predictions.html`)

### Analysis
- **EDA Notebook**: Data exploration and analysis (`EDA_ypnb.ipynb`)
