import pandas as pd
import os

class DiscrepancyIdentifier:
    """
    A class to identify discrepancies between DDD (area code) and CEP (postal code) states in a given DataFrame
    
    Attributes:
    - state_to_ddd (dict): A dictionary mapping Brazilian states to their corresponding DDD area codes
    - cep_to_state (dict): A dictionary mapping the first two digits of CEP codes to their corresponding states

    Methods:
    - __init__: Initializes the DiscrepancyIdentifier with the provided mappings
    - get_state_from_ddd: Gets the state abbreviation corresponding to a DDD code
    - identify_discrepancies: Identifies discrepancies between DDD and CEP state codes in the given DataFrame
    """ 
    def __init__(self):
        """
        Initializes the DiscrepancyIdentifier with the provided mappings
        
        Parameters:
            state_to_ddd (dict): A mapping of states to DDD codes
            cep_to_state (dict): A mapping of CEP codes to states
        """
        self.state_to_ddd = {
            "SP": list(range(11, 20)),  # São Paulo: 11-19
            "RJ": [21, 22, 24],         # Rio de Janeiro: 21, 22, 24
            "ES": [27, 28],             # Espírito Santo: 27, 28
            "MG": [31, 32, 33, 34, 35, 37, 38],  # Minas Gerais: 31-35, 37, 38
            "BA": [71, 73, 74, 75, 77], # Bahia: 71, 73-75, 77
            "SE": [79],                 # Sergipe: 79
            "PE": [81, 87],             # Pernambuco: 81, 87
            "AL": [82],                 # Alagoas: 82
            "PB": [83],                 # Paraíba: 83
            "RN": [84],                 # Rio Grande do Norte: 84
            "CE": [85, 88],             # Ceará: 85, 88
            "PI": [86, 89],             # Piauí: 86, 89
            "MA": [98, 99],             # Maranhão: 98, 99
            "PA": [91, 93, 94],         # Pará: 91, 93, 94
            "AP": [96],                 # Amapá: 96
            "AM": [92, 97],             # Amazonas: 92, 97
            "RR": [95],                 # Roraima: 95
            "AC": [68],                 # Acre: 68
            "RO": [69],                 # Rondônia: 69
            "TO": [63],                 # Tocantins: 63
            "DF": [61],                 # Distrito Federal: 61
            "GO": [62, 64],             # Goiás: 62, 64
            "MT": [65, 66],             # Mato Grosso: 65, 66
            "MS": [67],                 # Mato Grosso do Sul: 67
            "PR": list(range(41, 47)),  # Paraná: 41-46
            "SC": [47, 48, 49],         # Santa Catarina: 47-49
            "RS": [51, 53, 54, 55]      # Rio Grande do Sul: 51, 53-55
        }
        
        self.cep_to_state = {
            # São Paulo (SP)
            "01": "SP", "02": "SP", "03": "SP", "04": "SP", "05": "SP", 
            "06": "SP", "07": "SP", "08": "SP", "09": "SP", "10": "SP", 
            "11": "SP", "12": "SP", "13": "SP", "14": "SP", "15": "SP", 
            "16": "SP", "17": "SP", "18": "SP", "19": "SP",
            
            # Rio de Janeiro (RJ)
            "20": "RJ", "21": "RJ", "22": "RJ", "23": "RJ", "24": "RJ", 
            "25": "RJ", "26": "RJ", "27": "RJ", "28": "RJ",
            
            # Espírito Santo (ES)
            "29": "ES",
            
            # Minas Gerais (MG)
            "30": "MG", "31": "MG", "32": "MG", "33": "MG", "34": "MG", 
            "35": "MG", "36": "MG", "37": "MG", "38": "MG", "39": "MG",
            
            # Bahia (BA)
            "40": "BA", "41": "BA", "42": "BA", "43": "BA", "44": "BA", 
            "45": "BA", "46": "BA", "47": "BA", "48": "BA",
            
            # Sergipe (SE)
            "49": "SE",
            
            # Pernambuco (PE)
            "50": "PE", "51": "PE", "52": "PE", "53": "PE", "54": "PE", 
            "55": "PE", "56": "PE",
            
            # Alagoas (AL)
            "57": "AL",
            
            # Paraíba (PB)
            "58": "PB",
            
            # Rio Grande do Norte (RN)
            "59": "RN",
            
            # Ceará (CE)
            "60": "CE", "61": "CE", "62": "CE", "63": "CE",
            
            # Piauí (PI)
            "64": "PI",
            
            # Maranhão (MA)
            "65": "MA",
            
            # Pará (PA)
            "66": "PA", "67": "PA", "68": "PA",
            
            # Amapá (AP)
            "68": "AP",
            
            # Amazonas (AM)
            "69": "AM",
            
            # Acre (AC)
            "69": "AC",

            # Roraima (RR)
            "69": "RR",
            
            # Distrito Federal (DF)
            "70": "DF", "71": "DF", "72": "DF", "73": "DF",
            
            # Goiás (GO)
            "72": "GO", "73": "GO", "74": "GO", "75": "GO", "76": "GO",
            
            # Tocantins (TO)
            "77": "TO",
            
            # Mato Grosso (MT)
            "78": "MT",
            
            # Rondônia (RO)
            "78": "RO",
            
            # Mato Grosso do Sul (MS)
            "79": "MS",
            
            # Paraná (PR)
            "80": "PR", "81": "PR", "82": "PR", "83": "PR", "84": "PR", 
            "85": "PR", "86": "PR", "87": "PR",
            
            # Santa Catarina (SC)
            "88": "SC", "89": "SC",
            
            # Rio Grande do Sul (RS)
            "90": "RS", "91": "RS", "92": "RS", "93": "RS", "94": "RS", 
            "95": "RS", "96": "RS", "97": "RS", "98": "RS", "99": "RS"
        }

    def get_state_from_ddd(self, ddd):
        """
        Gets the state abbreviation corresponding to a DDD code
        
        Parameters:
        - ddd (int): The DDD code to check

        Returns:
        - str: The state abbreviation, or None if not found
        """
        if pd.isna(ddd):
            return None
        for state, ddds in self.state_to_ddd.items():
            if ddd in ddds:
                return state
        return None

    def identify_discrepancies(self, df, ddd_column='DDD', cep_column='CEP_2_DIG'):
        """
        Identifies discrepancies between DDD and CEP state codes in the given DataFrame

        Parameters:
        - df (pd.DataFrame): The DataFrame containing DDD and CEP columns
        - ddd_column (str): The name of the DDD column. Defaults to 'DDD'
        - cep_column (str): The name of the CEP column. Defaults to 'CEP_2_DIG'

        Returns:
        - pd.DataFrame: A DataFrame containing rows with discrepancies
        """
        df_discrepancies = df.copy()
        
        # Convert DDD to numeric
        df_discrepancies['DDD'] = pd.to_numeric(df_discrepancies[ddd_column], errors='coerce')
           
        # Add state columns
        df_discrepancies['ESTADO_CEP'] = df_discrepancies[cep_column].map(self.cep_to_state)
        df_discrepancies['ESTADO_DDD'] = df_discrepancies['DDD'].apply(self.get_state_from_ddd)
        
        # Identify discrepancies
        return df_discrepancies[
            (df_discrepancies['ESTADO_CEP'].notna()) & 
            (df_discrepancies['ESTADO_DDD'].notna()) & 
            (df_discrepancies['ESTADO_CEP'] != df_discrepancies['ESTADO_DDD'])
        ] 

class CleaningFinalDataFrames:
    """
    A class to perform data cleaning operations on DataFrames, such as filling missing values with median values based on grouping by 'ID_CLIENTE'

    Attributes:
    - median_values (dict): A dictionary to store median values for columns, keyed by 'ID_CLIENTE'

    Methods:
    - __init__: Initializes the class with the main DataFrame where missing values need to be filled
    - calculate_medians: Calculates the median values for a specified column, grouped by 'ID_CLIENTE', from the provided DataFrame (data)
    - fill_missing_with_input: Helper method to fill missing values in a row based on median values from 'input_dict'
    - fill_missing_values: Fills missing values for the specified columns using the median values grouped by 'ID_CLIENTE', from the provided DataFrame (data)
    """
    def __init__(self):
        """
        Initializes the class with the main DataFrame where missing values need to be filled

        Parameters:
        - median_values (dict): A dictionary to store median values for columns, keyed by 'ID_CLIENTE'
        """
        self.median_values = {}

    def calculate_medians(self, data_median, column_name):
        """
        Calculates the median values for a specified column, grouped by 'ID_CLIENTE', from the provided DataFrame (data_median)

        Parameters:
        - data_median (DataFrame): The DataFrame to use for calculating medians
        - column_name (str): The column for which the median is calculated

        Returns:
        - dict: A dictionary with median values keyed by 'ID_CLIENTE'
        """
        return data_median.groupby('ID_CLIENTE')[column_name].median()

    def fill_missing_with_input(self, row, input_dict, column_name):
        """
        Helper method to fill missing values in a row based on median values from 'input_dict'

        Parameters:
        - row (Series): A row of the DataFrame
        - input_dict (dict): A dictionary containing median values keyed by 'ID_CLIENTE'
        - column_name (str): The name of the column to fill

        Returns:
        - Value: The original or filled value for the column
        """
        if pd.isnull(row[column_name]):
            return input_dict.get(row['ID_CLIENTE'], row[column_name])
        return row[column_name]

    def fill_missing_values(self, df, columns_to_fill, data_median):
        """
        Fills missing values for the specified columns using the median values grouped by 'ID_CLIENTE', from the provided DataFrame (data_median)

        Parameters:
        - df (Dataframe): The main DataFrame where missing values need to be filled
        - columns_to_fill (dict): A dictionary where keys are column names and values are the names of the corresponding columns in 'data_median' to calculate medians
        - data_median (DataFrame): The DataFrame to use for calculating median values for filling missing data

        Returns:
        - DataFrame: The DataFrame with missing values filled
        """
        for column, info_column in columns_to_fill.items():
            # Calculate median values for the corresponding info column
            self.median_values[column] = self.calculate_medians(data_median, info_column)
            
            # Apply the function to fill missing values
            df[column] = df.apply(
                self.fill_missing_with_input, axis=1, 
                input_dict=self.median_values[column], 
                column_name=column
            )
        return df

def cleaning_pipeline(data, save_folder, save_filename):
    """
    Performs a series of data cleaning operations on the input DataFrame "data" and saves the cleaned data to a CSV file

    Parameters:
    - data (DataFrame): Input DataFrame containing movie data
    - save_folder (str): Folder path where the cleaned CSV file will be saved
    - save_filename (str): Filename for the cleaned CSV file (without the extension)

    Returns:
    - None
    """
    # Fill FLAG_PF missing values and convert "X" to 1
    data["FLAG_PF"] = data["FLAG_PF"].fillna(0).replace("X", 1)
    
    # Fill missing values with "Unknown"
    columns_to_fill_unknown = ["SEGMENTO_INDUSTRIAL", "DOMINIO_EMAIL", "PORTE"]
    data[columns_to_fill_unknown] = data[columns_to_fill_unknown].fillna("Unknown")
    
    # Convert DDD to numeric
    data["DDD"] = data["DDD"].astype(str).apply(lambda x: x if x.isdigit() else "Unknown")
    
    # Define state to region mapping
    regions = {
        "Norte": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
        "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
        "Centro-Oeste": ["DF", "GO", "MS", "MT"],
        "Sudeste": ["ES", "MG", "RJ", "SP"],
        "Sul": ["PR", "RS", "SC"]
    }
    state_to_region = {state: region for region, states in regions.items() for state in states}

    # Map DDD to its corresponding state and categorize into region
    data["DDD"] = data["DDD"].apply(lambda x: state_to_region.get(DiscrepancyIdentifier().get_state_from_ddd(int(x)) if x.isdigit() else None, "Unknown"))
    
    # Map CEP_2_DIG to its corresponding state and categorize into region
    data["CEP_2_DIG"] = data["CEP_2_DIG"].map(lambda x: state_to_region.get(DiscrepancyIdentifier().cep_to_state.get(x), "Unknown"))
    
    # Save the processed DataFrame to CSV
    save_path = os.path.join(save_folder, f"{save_filename}.csv")
    data.to_csv(save_path, index=False)