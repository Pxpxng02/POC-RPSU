import pandas as pd
import re
from rapidfuzz import fuzz

INPUT_FILE = "outputs/arxiv_cs_filtered.csv"
OUTPUT_FILE = "outputs/arxiv_cs_university.csv"

# ขยาย dictionary ของมหาวิทยาลัยไทย
UNIVERSITIES = {
    "Chulalongkorn University": [
        "Chulalongkorn", "CU", "Chulalonkorn", "Bangkok, Thailand"
    ],
    "Mahidol University": [
        "Mahidol", "Ramathibodi", "Siriraj", "Salaya"
    ],
    "Chiang Mai University": [
        "Chiang Mai", "CMU", "Chiangmai"
    ],
    "Prince of Songkla University": [
        "Prince of Songkla", "PSU", "Hat Yai", "Songkhla"
    ],
    "Khon Kaen University": [
        "Khon Kaen", "KKU"
    ]
}

# เพิ่มมหาวิทยาลัยอื่น ๆ ได้ตามต้องการ
def map_university(authors_str):
    if not isinstance(authors_str, str):
        return None
    
    best_match = None
    best_score = 0
    
    for uni, keywords in UNIVERSITIES.items():
        for kw in keywords:
            # regex exact match (กันเคส keyword โผล่ตรง ๆ)
            if re.search(r"\b" + re.escape(kw) + r"\b", authors_str, re.IGNORECASE):
                return uni
            
            # fuzzy match (กันเคสสะกดผิด / มีคำอื่นปน)
            score = fuzz.partial_ratio(kw.lower(), authors_str.lower())
            if score > 85 and score > best_score:
                best_match = uni
                best_score = score
    
    return best_match

# ฟังก์ชันหลัก
def main():
    df = pd.read_csv(INPUT_FILE)

    print("Mapping universities (regex + fuzzy)")
    df["university_match"] = df["authors"].apply(map_university)

    print("กรองเฉพาะบทความของมหาวิทยาลัยเป้าหมาย")
    df_filtered = df[df["university_match"].notna()]

    print(f"พบ {len(df_filtered)} บทความจากมหาวิทยาลัยเป้าหมาย")

    df_filtered.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"บันทึกไฟล์ {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
