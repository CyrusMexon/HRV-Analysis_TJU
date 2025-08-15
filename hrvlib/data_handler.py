import pandas as pd

def load_rr_csv(file_path: str) -> list[float]:
    """Load RR intervals from a CSV file.
    TODO: Add error handling for malformed files.
    TODO: Allow selection of specific columns if file has multiple.
    """
    df = pd.read_csv(file_path)
    rr = df.iloc[:, 0].dropna().tolist()
    return rr

def load_rr_txt(file_path: str) -> list[float]:
    """Load RR intervals from a TXT file.
    TODO: Handle multiple delimiters and detect numeric columns automatically.
    """
    with open(file_path, 'r') as f:
        content = f.read().strip().replace(',', ' ')
        rr = [float(x) for x in content.split()]
    return rr

def load_rr_edf(file_path: str):
    """Load RR intervals from an EDF file.
    TODO: Implement EDF parsing using pyEDFlib or mne.io.read_raw_edf.
    """
    raise NotImplementedError("EDF loading not yet implemented.")
