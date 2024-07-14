# Linear Regression Project with Flask

This project implements a simple Linear Regression model using Flask, focused on predicting house prices using the Boston Housing Dataset. It includes a web interface where users can input features of a house and get a predicted price.

## Purpose

This project serves as a learning exercise to understand:

- Building a machine learning model (Linear Regression) in Python.
- Creating a web application using Flask.
- Deploying a machine learning model with a Flask API.
- Basic HTML and CSS for front-end design.
- Handling dataset information presentation.

## Project Structure

The project is structured as follows:
```bash
project/
│
├── app/
│ ├── static/
│ │ └── style.css # CSS styles for the web interface
│ ├── templates/
│ │ ├── index.html # Main page for house price prediction
│ │ └── info.html # Dataset information page
│ ├── init.py # Flask application initialization
│ └── routes.py # Flask routes for rendering templates and API endpoints
│
├── run.py # Python script to run the Flask application
├── model.pkl # Serialized machine learning model (Linear Regression)
└── README.md # This README file
```

## Installation and Usage

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd project/
   ```

2. **Run the Flask application:**

Execute the `run.py` script to start the Flask web server.

```bash
python run.py
```

The application should now be accessible at http://localhost:5000.

3. **Usage:**

- Navigate to http://localhost:5000 to use the house price prediction interface.
- Click on "ℹ️ Dataset Information" to view details about the Boston Housing Dataset.

## Dataset Information

The Boston Housing Dataset provides details about various housing attributes in the Boston area. For more information about the dataset and its features, refer to info.html.

## Disclaimer

This project is developed for educational purposes to demonstrate the implementation of a machine learning model with Flask. It may not cover all production-level requirements such as security, scalability, or extensive error handling.

## Author

Created by Rushikesh Konapure
Feel free to reach out with any questions or suggestions!

