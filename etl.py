import os
import json
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import flatten_close_approach_data


class ETL:
    """
    A class to perform Extract, Transform, and Load (ETL) operations for NASA's Near-Earth Object (NEO) dataset.
    
    Attributes:
        output_dir (str): Directory to store extracted and processed data.
        API_KEY (str): API key for accessing NASA's NEO API.
        unprocessed_dir (str): Directory to store raw extracted data.
        processed_dir (str): Directory to store transformed data.
        unprocessed_output_file (str, optional): Path to the most recent extracted data file.
    """
    def __init__(self, api_key, output_dir='data'):
        """
        Initializes the ETL pipeline with API key and output directory.

        Args:
            api_key (str): NASA API key.
            output_dir (str, optional): Directory to store data. Defaults to 'data'.
        """
        self.output_dir = output_dir
        self.API_KEY = api_key
        self.unprocessed_dir = os.path.join(self.output_dir, 'unprocessed')
        self.processed_dir = os.path.join(self.output_dir, 'processed')
        os.makedirs(self.unprocessed_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.unprocessed_output_file = None

    def _extract(self, start, end) -> list:
        """
        Extracts Near-Earth Object (NEO) data from NASA's API within a specified page range.

        Args:
            start (int): Start page number.
            end (int): End page number.

        Returns:
            list: List of extracted NEO data records.
        """
        near_earth_objects = []
        base_url = "https://api.nasa.gov/neo/rest/v1/neo/browse"
        stop_extraction = False
    
        def fetch_page(page_number):
            nonlocal stop_extraction
            if stop_extraction:
                return []
                
            url = f"{base_url}?page={page_number}&api_key={self.API_KEY}"
            try:
                res = requests.get(url)
                if res.status_code != 200 or 'near_earth_objects' not in res.json():
                    print(f"Error fetching page {page_number}. Stopping extraction.")
                    stop_extraction = True
                    return []
                
                print(f"Fetched page {page_number}/{end}")  # Log progress
                return res.json().get('near_earth_objects', [])
            except requests.RequestException as e:
                print(f"Request failed for page {page_number}: {e}. Stopping extraction.")
                stop_extraction = True
                return []
    
        with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust `max_workers` as needed
            futures = {executor.submit(fetch_page, page): page for page in range(start, end + 1)}
            for future in as_completed(futures):
                near_earth_objects += future.result()
                if stop_extraction:
                    break  # Stop processing further pages

        output_file = os.path.join(self.unprocessed_dir, f"extracted_{start}_{end}.json")
        self.unprocessed_output_file = output_file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(near_earth_objects, f, indent=4, ensure_ascii=False)
    
        print(f"Extracted data saved to {output_file}")
        
        return near_earth_objects
        
    def _transform(self, input_path=None):
        """
        Transforms extracted NEO data by flattening nested structures and creating separate tables.

        Args:
            input_path (str, optional): Path to the unprocessed JSON file. Defaults to None.

        Returns:
            tuple: Transformed data as Pandas DataFrames (NEO data, estimated diameter, close approach, orbital data).
        """
        print("Transforming data")
        if input_path is None:
            if self.unprocessed_output_file is None:
                raise ValueError("No unprocessed data found. Extraction process may have failed or incorrect directory specified.")
            input_path = self.unprocessed_output_file
        df = pd.read_json(input_path) # central fact table
        df = df.drop(['links'], axis=1)

        ed_df = None # estimated diameter table
        cad_df = None # close approach data table
        od_df = None # orbital data table
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Transforming data", unit="row"):
            # flatten estimated diameter
            if isinstance(row['estimated_diameter'], dict):
                ed_row = pd.json_normalize(row['estimated_diameter'])
                ed_row['obj_id'] = row['id']
                ed_df = ed_row if ed_df is None else pd.concat([ed_df, ed_row], ignore_index=True)

            # get flattened close approach data history for item
            if row['close_approach_data'] != []:
                cad_flat = flatten_close_approach_data(row['close_approach_data'], row['id'])
                cad_df = cad_flat if cad_df is None else pd.concat([cad_df, cad_flat], ignore_index=True)

            # flatten orbital data
            if isinstance(row['orbital_data'], dict):
                od_row = pd.json_normalize(row['orbital_data'], max_level=0).drop(['orbit_class'], axis=1)
                od_row['obj_id'] = row['id']
                od_df = od_row if od_df is None else pd.concat([od_df, od_row], ignore_index=True)

        df = df.drop(['estimated_diameter', 'close_approach_data', 'orbital_data'], axis=1)
        return df, ed_df, cad_df, od_df
            
    def _load(self, start, end, **kwargs):
        """
        Loads transformed data into CSV files in the processed directory.

        Args:
            start (int): Start page number for naming convention.
            end (int): End page number for naming convention.
            **kwargs: DataFrames to save, with keys as filenames.
        """
        assert all(isinstance(value, pd.DataFrame) for value in kwargs.values()), "All kwargs must be pandas DataFrames"
        print("Loading Data")
        for key, value in kwargs.items():
            output_file = os.path.join(self.processed_dir, f"processed_{start}_{end}_{key}.csv")
            value.to_csv(output_file, encoding='utf-8', index=False)
        print(f"Successfully saved processed files to {self.processed_dir}")
        
    def run(self, start, end, extract_only=False,
            unprocessed_output_file=None, 
            skip_extract=False):
        """
        Runs the complete ETL process: Extract, Transform, and Load.

        Args:
            start (int): Start page number.
            end (int): End page number.
            extract_only (bool, optional): Whether to run only the extraction step. Defaults to False.
            unprocessed_output_file (str, optional): Path to previously extracted data.
            skip_extract (bool, optional): Whether to skip extraction. Defaults to False.
        """
        try:
            if skip_extract:
                if not unprocessed_output_file:
                    raise ValueError("Please specify path to unprocessed data when skipping extraction process.")
                self.unprocessed_output_file = unprocessed_output_file
            else:
                neo_json = self._extract(start, end)

            if extract_only:
                print(f"Extraction completed. Data saved at {self.unprocessed_output_file}")
                return  # Exit early if extract_only is set

            if not os.path.exists(self.unprocessed_output_file):
                raise FileNotFoundError(f"Unprocessed file not found: {self.unprocessed_output_file}")
                
            df, ed_df, cad_df, od_df = self._transform()
            self._load(start, end, **{'neos': df, 'estimated_diameter': ed_df, 
                                      'close_orbit_data': cad_df, 'orbital_data': od_df})
        except Exception as e:
            raise e
            