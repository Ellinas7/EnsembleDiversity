"""
Converte i 16 file CSV in Excel
"""

import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("/Users/matteopascuzzo/Desktop/Results")

# Lista dei file da convertire
FILES = [
    "CoppieStaticGruppo1_Voting2su2",
    "CoppieStaticGruppo1_RecoveryBlock",
    "CoppieStaticGruppo2_Voting2su2",
    "CoppieStaticGruppo2_RecoveryBlock",
    "CoppiePercentileGruppo1_Voting2su2",
    "CoppiePercentileGruppo1_RecoveryBlock",
    "CoppiePercentileGruppo2_Voting2su2",
    "CoppiePercentileGruppo2_RecoveryBlock",
    "TripleStaticGruppo1_MajorityVoting",
    "TripleStaticGruppo1_Voting1suN",
    "TripleStaticGruppo2_MajorityVoting",
    "TripleStaticGruppo2_Voting1suN",
    "TriplePercentileGruppo1_MajorityVoting",
    "TriplePercentileGruppo1_Voting1suN",
    "TriplePercentileGruppo2_MajorityVoting",
    "TriplePercentileGruppo2_Voting1suN",
]

def convert_all():
    print("Conversione CSV -> Excel")
    print("=" * 50)
    
    for name in FILES:
        csv_path = RESULTS_DIR / f"{name}.csv"
        xlsx_path = RESULTS_DIR / f"{name}.xlsx"
        
        try:
            df = pd.read_csv(csv_path)
            df.to_excel(xlsx_path, index=False)
            print(f"✓ {name}.xlsx")
        except Exception as e:
            print(f"✗ {name}: {e}")
    
    print("\nConversione completata!")

if __name__ == "__main__":
    convert_all()