import pandas as pd  

def flatten_close_approach_data(close_approach_data: list, obj_id):
    """
    Flattens the close approach data for a given Near-Earth Object (NEO).

    Args:
        data (list): List of close approach data dictionaries.
        obj_id (str): ID of the Near-Earth Object.

    Returns:
        pd.DataFrame: Flattened close approach data as a DataFrame.
    """
    cad_df = None
    for record in close_approach_data:
        row = pd.json_normalize(record)
        if cad_df is None:
            cad_df = row
        else:
            cad_df = pd.concat([cad_df, row], ignore_index=True)
    cad_df['obj_id'] = obj_id
    return cad_df
    