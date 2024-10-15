# [Customer Default Prediction on Loans](https://clientdefaultpred.azurewebsites.net/)

## Dataset
This project is composed by four datasets. 

### Base Cadastral
- ID_CLIENTE: The client ID
- DATA_CADASTRO: The date the client registered
- DDD: The area code of the client's phone number
- FLAG_PF: Indicates if the client is an individual ('PF') or not
- SEGMENTO_INDUSTRIAL: The industrial segment of the client
- DOMINIO_EMAIL: The email domain of the client
- PORTE: The size of the company (e.g., small, medium, large)
- CEP_2_DIG: The first two digits of the client's postal code

### Base Info
- ID_CLIENTE: The client ID
- SAFRA_REF: The reference season for the data
- RENDA_MES_ANTERIOR: The income of the previous month
- NO_FUNCIONARIOS: The number of employees in the company on the previous month

### Base Pagamentos (training and testing)
- ID_CLIENTE: The client ID
- SAFRA_REF: The reference season for the data
- DATA_EMISSAO_DOCUMENTO: The loan issue date
- DATA_VENCIMENTO: The due date to pay the loan
- VALOR_A_PAGAR: The amount to be paid
- TAXA: The interest rate of the loan
- DATA_PAGAMENTO: The date the client paid the loan (only available on the training dataset)

### Target
There's no column for a target variable, so it has to be manually made. The target needs to have the name **INADIMPLENCIA**, and it will be 1 for customers who have a delay, that is, a difference between payment and due date, greater or equal to 5 days, and 0 for all others
- INADIMPLENCIA: Defaulted the loan payment or not

## Objectives
The main objective of this project is:

**To develop a predictive model capable of generating predictions regarding the samples present in the base_pagamentos_teste.csv database. A new dataset containing the columns: ID_CLIENTE, SAFRA_REF and INADIMPLENTE must be created, with INADIMPLENTE being the probability of the customer defaulting the payment**

To achieve this objective, it was further broken down into the following technical sub-objectives:

1. To clean and join the datasets
2. To perform in-depth exploratory data analysis of the resulting dataset
3. To engineer new predictive features from the available features
4. To develop a supervised model to predict the chances of a customer default a payment
5. To put the model into production with a web app

## Main Insights

Open a web browser page and type `localhost` in the search bar. This app should load and be ready for usage. Use the datasets in the folder `Data for app usage` to test the `Predict with Dataset` function, or create your dataset based on the original data. Or, explore the `Predict with Manual Data` function, to manually input a row of data for the model to predict.

