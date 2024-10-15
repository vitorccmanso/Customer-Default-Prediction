import pandas as pd
import numpy as np
import pickle

class PredictPipeline:
    """
    A class for predicting the chances of a client defaulting a loan payment using a pre-trained model and preprocessing pipeline.

    Methods:
    - __init__: Initializes the PredictPipeline object by loading the mappings, preprocessor, and model from .pkl files
    - preprocess_dataset: Processes the input dataset, ensuring it contains the required columns
    - preprocess_data: Preprocesses the input data, including feature engineering and transformation
    - apply_mappings: Maps DDD and CEP to regions using pre-loaded mappings
    - get_feature_names: Retrieves the feature names after preprocessing is applied
    - predict: Predicts the chances of a client defaulting a loan payment
    - results_df: Saves predictions to a CSV file and returns a list of dictionaries for HTML rendering
    """
    def __init__(self):
        """
        Initializes the PredictPipeline object by loading the mapping, preprocessor and model from .pkl files
        """
        # Load mappings
        mappings_path = "app/artifacts/combined_mappings.pkl"
        with open(mappings_path, "rb") as f:
            self.mappings = pickle.load(f)

        # Load preprocessor
        preprocessor_path = "app/artifacts/preprocessor.pkl"
        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)

        # Load model
        model_path = "app/artifacts/model.pkl"
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def results_df(self, data, predictions, path):
        """
        Saves predictions to a CSV file and returns a list of dictionaries for rendering in HTML

        Parameters:
        - data (pandas.DataFrame): The original input data containing 'ID_CLIENTE' and 'SAFRA_REF'
        - predictions (list): The predicted chances of the client defaulting
        - path (str): The file path to save the CSV file

        Returns:
        - list: The results in dictionary format for HTML rendering
        """
        # Create a new DataFrame with ID_CLIENTE, SAFRA_REF, and the predictions
        id_cliente = data["ID_CLIENTE"]
        safra_ref = data["SAFRA_REF"]
        results_df = pd.DataFrame({
            "ID_CLIENTE": id_cliente,
            "SAFRA_REF": safra_ref,
            "INADIMPLENTE": predictions
        })
        # Save results to a temporary CSV file and convert DataFrame to a list of dictionaries for rendering in HTML
        results_df.to_csv(path, index=False)
        results = results_df.to_dict(orient="records")
        return results

    def apply_mappings(self, data):
        """
        Maps DDD and CEP to their corresponding regions using the pre-loaded mappings

        Parameters:
        - data (pandas.DataFrame): The input data containing 'DDD' and 'CEP_2_DIG' columns

        Returns:
        - pandas.DataFrame: The data with mapped regions for 'DDD' and 'CEP_2_DIG'
        """
        state_to_ddd = self.mappings["state_to_ddd"]
        cep_to_state = self.mappings["cep_to_state"]

        # Define regions and create a mapping from state to region
        regions = {
            "Norte": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
            "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
            "Centro-Oeste": ["DF", "GO", "MS", "MT"],
            "Sudeste": ["ES", "MG", "RJ", "SP"],
            "Sul": ["PR", "RS", "SC"]
        }
        state_to_region = {state: region for region, states in regions.items() for state in states}
        
        # Create a set of all valid regions
        valid_regions = set(regions.keys())

        # Map DDD to region using the saved mapping
        def clean_and_map_ddd(ddd):
            if ddd in valid_regions:
                return ddd
            cleaned_ddd = ''.join(filter(str.isdigit, str(ddd)))
            if cleaned_ddd:
                cleaned_ddd = int(cleaned_ddd)
                for state, ddd_list in state_to_ddd.items():
                    if cleaned_ddd in ddd_list:
                        region = state_to_region.get(state, "Unknown")
                        return region
            return "Unknown"
        data["DDD"] = data["DDD"].apply(clean_and_map_ddd)

        # Map CEP_2_DIG to region using the saved mapping
        def map_cep(cep):
            if cep in valid_regions:
                return cep
            state = cep_to_state.get(cep, "Unknown")
            return state_to_region.get(state, "Unknown")
        data["CEP_2_DIG"] = data["CEP_2_DIG"].astype(str).apply(map_cep)

        return data

    def get_feature_names(self, no_transformation_cols, cbrt_cols, one_hot_cols):
        """
        Retrieves the feature names after preprocessing is applied

        Parameters:
        - no_transformation_cols (list): List of numeric columns without transformations
        - cbrt_cols (list): List of columns where cubic root transformation is applied
        - one_hot_cols (list): List of categorical columns for which OneHotEncoding is applied

        Returns:
        - feature_names (list): List of feature names after preprocessing
        """
        numeric_features = no_transformation_cols + cbrt_cols
        one_hot_features = list(self.preprocessor.named_transformers_["cat_onehot"]["onehot"].get_feature_names_out(one_hot_cols))
        feature_names = numeric_features + one_hot_features
        return feature_names

    def preprocess_dataset(self, input_data):
        """
        Processes the input dataset, ensuring it contains the required columns

        Parameters:
        - input_data (pandas.DataFrame): The input data to be processed

        Returns:
        - pandas.DataFrame: The processed input data
        """
        # Check if the uploaded dataset contains all required columns 
        columns = ["ID_CLIENTE", "SAFRA_REF", "DATA_EMISSAO_DOCUMENTO", "DATA_VENCIMENTO", "VALOR_A_PAGAR", "TAXA", "DATA_CADASTRO", "DDD", "SEGMENTO_INDUSTRIAL", "DOMINIO_EMAIL", "PORTE", "CEP_2_DIG", "RENDA_MES_ANTERIOR", "NO_FUNCIONARIOS"]
        if not set(columns).issubset(input_data.columns):
            raise ValueError("Dataset must contain all the columns listed above")

        # Drop unnecessary columns
        input_data = input_data[columns]

        # Apply mappings for the values of DDD and CEP_2_DIG that need it
        input_data = self.apply_mappings(input_data)
        return input_data

    def preprocess_data(self, data):
        """
        Preprocesses the input data, including feature engineering and transformation

        Parameters:
        - data: The input data to be preprocessed

        Returns:
        - pandas.DataFrame: The preprocessed data
        """
        # Transform date columns into datetime format
        data["SAFRA_REF"] = pd.to_datetime(data["SAFRA_REF"])
        data["DATA_VENCIMENTO"] = pd.to_datetime(data["DATA_VENCIMENTO"])
        data["DATA_EMISSAO_DOCUMENTO"] = pd.to_datetime(data["DATA_EMISSAO_DOCUMENTO"])
        
        # Replace "Unknown" in "DATA_CADASTRO" with corresponding "DATA_EMISSAO_DOCUMENTO" values
        data["DATA_CADASTRO"] = data["DATA_CADASTRO"].replace("Unknown", np.nan)
        data["DATA_CADASTRO"] = data["DATA_CADASTRO"].fillna(data["DATA_EMISSAO_DOCUMENTO"])
        data["DATA_CADASTRO"] = pd.to_datetime(data["DATA_CADASTRO"])

        # Feature engineering
        data["month_vencimento"] = data["DATA_VENCIMENTO"].dt.month
        data["days_until_due"] = (data["DATA_VENCIMENTO"] - data["SAFRA_REF"]).dt.days
        data["days_since_registration"] = (data["DATA_EMISSAO_DOCUMENTO"] - data["DATA_CADASTRO"]).dt.days

        # Map the "DDD" and "CEP_2_DIG" columns
        data = self.apply_mappings(data)

        # Drop unused columns
        columns_to_drop = ["ID_CLIENTE", "SAFRA_REF", "DATA_EMISSAO_DOCUMENTO", "DATA_VENCIMENTO", "DATA_CADASTRO", "FLAG_PF"]
        existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
        data = data.drop(columns=existing_columns_to_drop)

        # Preprocess the data
        no_transformation_cols = ["TAXA", "month_vencimento", "NO_FUNCIONARIOS", "days_since_registration"]
        cbrt_cols = ["VALOR_A_PAGAR", "RENDA_MES_ANTERIOR", "days_until_due"]
        one_hot_cols = data.select_dtypes(include="object").columns
        data = self.preprocessor.transform(data)
        feature_names = self.get_feature_names(no_transformation_cols, cbrt_cols, one_hot_cols)

        # Create a DataFrame with the transformed data and feature names
        new_data = pd.DataFrame(data, columns=feature_names)
        return new_data

    def predict(self, data):
        """
        Predicts the chances of a client defaulting a loan payment

        Parameters:
        - data: The input data for prediction

        Returns:
        - list: The predicted chances of a client defaulting
        """
        preds_proba = self.model.predict_proba(self.preprocess_data(data))[:,1]
        return preds_proba

class CustomData:
    """ 
    A class representing custom datasets for client information

    Attributes:
    - ID_CLIENTE (int): The client ID
    - SAFRA_REF (str): The reference season for the data
    - DATA_EMISSAO_DOCUMENTO (str): The loan issue date
    - DATA_VENCIMENTO (str): The due date to pay the loan
    - VALOR_A_PAGAR (float): The amount to be paid
    - TAXA (float): The interest rate of the loan
    - DATA_CADASTRO (str): The date the client registered
    - DDD (int): The area code of the client's phone number
    - FLAG_PF (str): Indicates if the client is an individual ('PF') or not
    - SEGMENTO_INDUSTRIAL (str): The industrial segment of the client
    - DOMINIO_EMAIL (str): The email domain of the client
    - PORTE (str): The size of the company (e.g., small, medium, large)
    - CEP_2_DIG (int): The first two digits of the client's postal code
    - RENDA_MES_ANTERIOR (int): The income of the previous month
    - NO_FUNCIONARIOS (int): The number of employees in the company
    
    Methods:
    - __init__: Initializes the CustomData object with the provided attributes
    - get_data_as_dataframe: Converts the CustomData object into a pandas DataFrame
    """
    def __init__(self, ID_CLIENTE: int,
                    SAFRA_REF: str,
                    DATA_EMISSAO_DOCUMENTO: str,
                    DATA_VENCIMENTO: str,
                    VALOR_A_PAGAR: float,
                    TAXA: float,
                    DATA_CADASTRO: str,
                    DDD: int,
                    FLAG_PF: str,
                    SEGMENTO_INDUSTRIAL: str,
                    DOMINIO_EMAIL: str,
                    PORTE: str,
                    CEP_2_DIG: int,
                    RENDA_MES_ANTERIOR: int,
                    NO_FUNCIONARIOS: int):
        """
        Initializes the CustomData object with the provided attributes
        """
        self.ID_CLIENTE = ID_CLIENTE
        self.SAFRA_REF = SAFRA_REF
        self.DATA_EMISSAO_DOCUMENTO = DATA_EMISSAO_DOCUMENTO
        self.DATA_VENCIMENTO = DATA_VENCIMENTO
        self.VALOR_A_PAGAR = VALOR_A_PAGAR
        self.TAXA = TAXA
        self.DATA_CADASTRO = DATA_CADASTRO
        self.DDD = DDD
        self.FLAG_PF = FLAG_PF
        self.SEGMENTO_INDUSTRIAL = SEGMENTO_INDUSTRIAL
        self.DOMINIO_EMAIL = DOMINIO_EMAIL
        self.PORTE = PORTE
        self.CEP_2_DIG = CEP_2_DIG
        self.RENDA_MES_ANTERIOR = RENDA_MES_ANTERIOR
        self.NO_FUNCIONARIOS = NO_FUNCIONARIOS

    def get_data_as_dataframe(self):
        """
        Converts the CustomData object into a pandas DataFrame

        Returns:
        - pd.DataFrame: The CustomData object as a DataFrame
        """
        custom_data_input_dict = {
            "ID_CLIENTE": [self.ID_CLIENTE],
            "SAFRA_REF": [self.SAFRA_REF],
            "DATA_EMISSAO_DOCUMENTO": [self.DATA_EMISSAO_DOCUMENTO],
            "DATA_VENCIMENTO": [self.DATA_VENCIMENTO],
            "VALOR_A_PAGAR": [self.VALOR_A_PAGAR],
            "TAXA": [self.TAXA],
            "DATA_CADASTRO": [self.DATA_CADASTRO],
            "DDD": [self.DDD],
            "FLAG_PF": [self.FLAG_PF],
            "SEGMENTO_INDUSTRIAL": [self.SEGMENTO_INDUSTRIAL],
            "DOMINIO_EMAIL": [self.DOMINIO_EMAIL],
            "PORTE": [self.PORTE],
            "CEP_2_DIG": [self.CEP_2_DIG],
            "RENDA_MES_ANTERIOR": [self.RENDA_MES_ANTERIOR],
            "NO_FUNCIONARIOS": [self.NO_FUNCIONARIOS]
        }
        return pd.DataFrame(custom_data_input_dict)